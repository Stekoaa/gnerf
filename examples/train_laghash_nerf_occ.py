"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import logging
import os
import sys
import time
import warnings

import hydra
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

home_dir = os.path.expanduser('~')
project_root = os.path.join(home_dir, 'gnerf')
sys.path.append(project_root)

from datasets.nerf_synthetic import SubjectLoader
from datasets.tanks_and_temples import TanksTempleDataset
from examples.utils.general_utils import set_random_seed, TANKS_TEMPLE_SCENES, NERF_SYNTHETIC_SCENES
from examples.utils.loss_utils import calculate_loss_warmup, calculate_lod_sigma_loss, calculate_smooth_l1_loss
from examples.utils.metric_utils import calculate_psnr
from examples.utils.render_utils import render_image_with_occgrid
from nerfacc.estimators.occ_grid import OccGridEstimator
from radiance_fields.laghash import LagHashRadianceField

# Disable warnings
warnings.filterwarnings("ignore")

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../config/",
    "config_name": "synthetic_occ.yaml",
}

# A logger for this file
log = logging.getLogger(__name__)

def initialize_output():
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_path = hydra_cfg['runtime']['output_dir']
    
    log.info(f"Saving outputs in: {to_absolute_path(output_path)}")
    os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)
    
    writer = SummaryWriter(output_path, purge_step=0)
    return writer, output_path

def get_training_params(cfg):
    scene = cfg.dataset.scene
    if scene in TANKS_TEMPLE_SCENES:
        weight_decay = cfg.optimizer.weight_decay
    else:
        weight_decay = (
            1e-5 if scene in ["materials", "ficus", "drums"]
            else 1e-6
        )
    
    return {
        "max_steps": cfg.trainer.max_steps,
        "target_sample_batch_size": 1 << 18,
        "weight_decay": weight_decay,
    }

def get_occupancy_params(cfg):
    return {
        "grid_resolution": cfg.occupancy.grid_resolution,
        "grid_nlvl": cfg.occupancy.grid_nlvl,
    }

def get_render_parameters(cfg):
    return {
        "render_step_size": cfg.render.render_step_size,
        "alpha_thre": cfg.render.alpha_thre,
        "cone_angle": cfg.render.cone_angle,
    }

def get_dataset_and_scene_parameters(cfg, device):
    scene = cfg.dataset.scene
    init_batch_size = cfg.dataset.init_batch_size
    
    if scene in TANKS_TEMPLE_SCENES:
        data_path = os.path.join(cfg.dataset.data_root, scene)
        train_dataset = TanksTempleDataset(
            data_path, split="train", downsample=1, is_stack=False, num_rays=init_batch_size
        )
        test_dataset = TanksTempleDataset(
            data_path, split="test", downsample=1, is_stack=True, num_rays=None
        )
        aabb = train_dataset.scene_bbox.to(device).view(-1)
        near_plane, far_plane = train_dataset.near_far
        white_bg = train_dataset.white_bg
    else:
        train_dataset = SubjectLoader(
            subject_id=scene, root_fp=cfg.dataset.data_root,
            split="train", num_rays=init_batch_size, device=device
        )
        test_dataset = SubjectLoader(
            subject_id=scene, root_fp=cfg.dataset.data_root,
            split="test", num_rays=None, device=device
        )
        aabb = torch.tensor(cfg.scene.aabb, device=device)
        near_plane = cfg.scene.near_plane
        far_plane = cfg.scene.far_plane
        white_bg = None

    return {
        "train_dataset": train_dataset, 
        "test_dataset": test_dataset, 
        "aabb": aabb, 
        "near_plane": near_plane, 
        "far_plane": far_plane, 
        "white_bg": white_bg
    }

def initialize_estimator(aabb, grid_resolution, grid_nlvl, device):
    return OccGridEstimator(roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl).to(device)

def initialize_radiance_field(cfg, estimator, device):
    std_decay_factor = (cfg.model.std_final_factor / cfg.model.std_init_factor) ** (cfg.trainer.size_decay_every/cfg.trainer.max_steps)
    
    radiance_field = LagHashRadianceField(
        aabb=estimator.aabbs[-1], 
        num_splashes=cfg.model.num_splashes,
        # xd
        # log2_hashmap_size=cfg.model.log2_hashmap_size, 
        # max_resolution=cfg.model.max_resolution,
        n_features_per_gauss=cfg.model.n_features_per_gauss,
        n_neighbours=cfg.model.n_neighbours, 
        std_init_factor=cfg.model.std_init_factor,
        fixed_std=cfg.model.fixed_std,
        decay_factor=std_decay_factor, 
        splits=cfg.model.splits
    ).to(device)

    if cfg.model.load_model_path != "":
        state = torch.load(cfg.model.load_model_path, map_location=device)
        radiance_field.load_state_dict(state['model'])
        estimator.load_state_dict(state['occupancy'])
        log.info(f"Loaded model from {cfg.model.load_model_path}")
    
    return radiance_field

def initialize_optimizer(cfg, radiance_field, weight_decay):
    params_dict = { name : param for name, param in radiance_field.named_parameters()}
    
    gau_params, codebook_params, rest_params = [], [], []
    for name in params_dict:
        if ("means" in name) or ("stds" in name):
            gau_params.append(params_dict[name])
        elif "feats" in name:
            codebook_params.append(params_dict[name])
        else:
            rest_params.append(params_dict[name])

    gau_lr = cfg.optimizer.learning_rate * cfg.optimizer.gaussian_factor
    params = [
        {"params": gau_params, "lr": gau_lr, "eps": cfg.optimizer.eps, "weight_decay": 0.0},
        {"params": codebook_params, "lr": cfg.optimizer.learning_rate, "eps": cfg.optimizer.eps, "weight_decay": weight_decay},
        {"params": rest_params, "lr": cfg.optimizer.learning_rate, "eps": cfg.optimizer.eps, "weight_decay": weight_decay}
    ]
    
    return torch.optim.Adam(params)

def initialize_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=100),
            torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(m*cfg.trainer.max_steps) for m in cfg.scheduler.milestones], gamma=cfg.scheduler.gamma),
        ]
    )

def retrieve_image_data(img):
    render_bkgd = img["color_bkgd"]
    rays = img["rays"]
    pixels = img["pixels"]
    return render_bkgd, rays, pixels

@hydra.main(**_HYDRA_PARAMS)
def run(cfg: DictConfig):
    device = cfg.device
    set_random_seed(42)
    
    writer, output_path = initialize_output()

    if cfg.dataset.scene in TANKS_TEMPLE_SCENES or cfg.dataset.scene in NERF_SYNTHETIC_SCENES:
        train_params = get_training_params(cfg)
        max_steps, target_sample_batch_size, weight_decay = (
            train_params["max_steps"],
            train_params["target_sample_batch_size"], 
            train_params["weight_decay"]
        )
        
        occupancy_params = get_occupancy_params(cfg)
        grid_resolution, grid_nlvl = (
            occupancy_params["grid_resolution"],
            occupancy_params["grid_nlvl"]
        )

        render_params = get_render_parameters(cfg)
        render_step_size, alpha_thre, cone_angle = (
            render_params["render_step_size"],
            render_params["alpha_thre"],
            render_params["cone_angle"]
        )

        dataset_params = get_dataset_and_scene_parameters(cfg, device)
        train_dataset, test_dataset, aabb, near_plane, far_plane, white_bg = (
            dataset_params["train_dataset"],
            dataset_params["test_dataset"],
            dataset_params["aabb"],
            dataset_params["near_plane"],
            dataset_params["far_plane"],
            dataset_params["white_bg"]
        )
    else:
        error_message = f"Invalid scene: {cfg.dataset.scene}"
        logging.error(error_message)
        raise ValueError(error_message)

    estimator = initialize_estimator(aabb, grid_resolution, grid_nlvl, device)

    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    radiance_field = initialize_radiance_field(cfg, estimator, device)

    num_params = sum(p.numel() for p in radiance_field.parameters() if p.requires_grad)
    log.info(f"Number of parameters: {num_params/1e6:.2f}M")
    
    optimizer = initialize_optimizer(cfg, radiance_field, weight_decay)
    scheduler = initialize_scheduler(cfg, optimizer)
    
    # training
    log.info('Starting training')
    tic = time.time()
    for step in tqdm(range(max_steps + 1), desc="Training"):
        radiance_field.train()
        estimator.train()

        i = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[i]
        render_bkgd, rays, pixels = retrieve_image_data(data)

        def occ_eval_fn(x):
            density = radiance_field.query_density(x)
            return density * render_step_size

        # update occupancy grid
        estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )

        # render
        rgb, acc, depth, kl_div, n_rendering_samples, mip_loss = render_image_with_occgrid(
            radiance_field,
            estimator,
            rays,
            # rendering options
            near_plane=near_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )

        if n_rendering_samples == 0:
            continue

        if target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = int(len(pixels) * (target_sample_batch_size / float(n_rendering_samples)))
            train_dataset.update_num_rays(num_rays)

        # compute loss
        loss_warm_up = calculate_loss_warmup(step, max_steps)
        mip_loss = mip_loss.mean() # distortion loss
        sigma_loss, surf_loss, i = 0, 0, 0
        
        # TODO: tu coś trzeba pomajstrować
        # for idx in range(radiance_field.n_levels):
        #     resolution = radiance_field.mlp_base.encoding.resolutions[idx]
        #     stds = radiance_field.mlp_base.encoding.get_stds(idx)
        #     if stds is not None:
        #         sigma_loss += calculate_lod_sigma_loss(resolution, stds)
        #         i += 1
        if i > 0:
            sigma_loss /= i
            surf_loss = kl_div.mean()

        loss = calculate_smooth_l1_loss(rgb, pixels)
        if cfg.trainer.weight_surface:
            loss += cfg.trainer.weight_surface * loss_warm_up * surf_loss
        if cfg.trainer.weight_sigma and (not cfg.model.fixed_std):
            loss += cfg.trainer.weight_sigma * loss_warm_up * sigma_loss
        if cfg.trainer.weight_mip:
            loss += cfg.trainer.weight_mip * mip_loss

        optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        grad_scaler.scale(loss).backward()
        optimizer.step()
        scheduler.step()

        if step % cfg.trainer.log_every == 0:
            elapsed_time = time.time() - tic
            log.info(
                f"Training info: "
                f"step={step} | elapsed_time={elapsed_time:.2f}s | "
                f"whole_loss={loss:.5f} | surf_loss={surf_loss:.5f} | " 
                f"sigma_loss={sigma_loss:.5f} | n_rendering_samples={n_rendering_samples:d} | "
                f"max_depth={depth.max():.3f} | "
            )
        
        if (step % cfg.trainer.size_decay_every == cfg.trainer.size_decay_every-1) and cfg.model.fixed_std:
            radiance_field.mlp_base.encoding.update_factor()

        if step % cfg.trainer.save_every == 0:
            state_dict = {
                "steps": step,
                "model": radiance_field.state_dict(),
                "occupancy": estimator.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            
            model_output_path = f"{output_path}/model.pth"
            torch.save(state_dict, model_output_path)
            log.info(f"Model saved to {model_output_path}")
            
            # xd
            # for idx in range(radiance_field.n_levels):
            #     means = radiance_field.mlp_base.encoding.get_means(idx)
            #     if means is not None:
            #         means = means.reshape(-1, means.shape[-1])
            #         means_cloud = trimesh.PointCloud(means.cpu().detach().numpy())
            #         if step > 0:
            #             os.remove(os.path.join(output_path, f'means_lod{idx}@{step-cfg.trainer.save_every:05d}.ply'))
                    
            #         means_lod_path = os.path.join(output_path, f'means_lod{idx}@{step:05d}.ply')
            #         means_cloud.export(means_lod_path)
            #         log.info(f"Means saved to {means_lod_path}")
        
        if step % cfg.trainer.visualize_every == 0 and step > 0:
            log.info("Starting validation")
            radiance_field.eval()
            estimator.eval()

            with torch.no_grad():
                data = test_dataset[0]
                render_bkgd, rays, pixels = retrieve_image_data(data)
                rgb, _, _, _, _, _ = render_image_with_occgrid(
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=cone_angle,
                    alpha_thre=alpha_thre,
                )
                visualize = torch.concatenate([rgb, pixels], dim=1)
                writer.add_image("visual/rgb", visualize,  step, dataformats="HWC")
                
            psnr = calculate_psnr(rgb, pixels)
            log.info(f"Validation: psnr={psnr:.2f}")

    # evaluation
    log.info('Starting evaluation')
    
    radiance_field.eval()
    estimator.eval()
    psnrs = []
    with torch.no_grad():
         for i in tqdm(range(len(test_dataset)), desc='Evaluation'):
            render_bkgd, rays, pixels = retrieve_image_data(test_dataset[i])
            rgb, _, _, _, _, _ = render_image_with_occgrid(
                radiance_field,
                estimator,
                rays,
                # rendering options
                near_plane=near_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
            )
            
            psnrs.append(calculate_psnr(rgb, pixels))
            imageio.imwrite(
                f"{output_path}/test/rgb_test_{i}.png",
                (rgb.cpu().numpy() * 255).astype(np.uint8),
            )

    psnr_avg = sum(psnrs) / len(psnrs)
    logging.info(f"Evaluation: psnr_avg={psnr_avg}")
    with open(f"{output_path}/metrics.txt", "w") as fp:
        fp.write(f"PSNR:{psnr_avg:.3f}")
    writer.add_scalar("test/psnr", psnr_avg, max_steps)
    writer.close()

if __name__ == "__main__":
    run()
