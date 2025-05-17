"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings

import tyro
import imageio
import numpy as np
import torch
import yaml
import trimesh
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import nn

home_dir = os.path.expanduser('~')
project_root = os.path.join(home_dir, 'gnerf')
sys.path.append(project_root)

from dataclasses import dataclass, field
from typing import Type, Optional
from datasets.nerf_synthetic import SubjectLoader
from datasets.tanks_and_temples import TanksTempleDataset
from utils.general_utils import set_random_seed, TANKS_TEMPLE_SCENES, NERF_SYNTHETIC_SCENES
from utils.loss_utils import calculate_loss_warmup, calculate_smooth_l1_loss
from utils.metric_utils import calculate_psnr
from utils.render_utils import render_image_with_occgrid
from utils.config_utils import InstantiateConfig, convert_markup_to_ansi, CONSOLE
from nerfacc.estimators.occ_grid import OccGridEstimator
from radiance_fields.laghash import LagHashRadianceField
from pathlib import Path

# Disable warnings
warnings.filterwarnings("ignore")

# A logger for this file
log = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    # _target: Type = field(default_factory=lambda: SubjectLoader)
    # """Config class for the dataset."""
    name: str = "Synthetic"
    """Name of the dataset."""
    data_root: Path = Path("data/nerf_dataset")
    """Path to the dataset."""
    scene: str = "ficus"
    """Scene name."""
    init_batch_size: int = 1024
    """Initial batch size for training."""

@dataclass
class SceneConfig:
    aabb: list = field(default_factory=lambda: [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])
    """Axis-Aligned Bounding Box (AABB) of the scene."""
    near_plane: float = 2.0
    """Near plane distance."""
    far_plane: float = 6.0
    """Far plane distance."""

@dataclass
class RenderConfig:
    render_step_size: float = 0.005
    """Step size for rendering."""
    alpha_thre: float = 0.0
    """Alpha threshold for rendering."""
    cone_angle: float = 0.0
    """Cone angle for rendering."""

@dataclass
class ModelConfig:
    log2_hashmap_size: int = 17
    """Log2 of the size of the hashmap."""
    n_levels: int = 16
    """Number of levels in the hashmap."""
    n_neighbours: int = 16
    """Number of neighbours for the hashmap."""
    n_features_per_gauss: int = 10
    """Number of features per Gaussian."""
    max_resolution: int = 1024
    """Maximum resolution of the scene."""
    num_splashes: int = 4
    """Number of splashes in the scene."""
    fixed_std: bool = False
    """Whether to use fixed standard deviation."""
    std_init_factor: float = 50
    """Initial standard deviation factor."""
    std_final_factor: float = 5
    """Final standard deviation factor."""
    load_model_path: str = ""
    """Path to the model to load."""
    splits: list = field(default_factory=lambda: [0.875, 0.9375])
    """Splits for the model."""
    n_gausses: int = 40000
    """Number of Gaussians in the model."""

@dataclass
class OccupancyConfig:
    grid_resolution: int = 128
    """Resolution of the occupancy grid."""
    grid_nlvl: int = 1
    """Number of levels in the occupancy grid."""

@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-2
    """Learning rate for the optimizer."""
    gaussian_factor: float = 0.1
    """Gaussian factor for the optimizer."""
    weight_decay: float = 0.0
    """Weight decay for the optimizer."""
    eps: float = 1e-15
    """Epsilon for the optimizer."""

@dataclass
class SchedulerConfig:
    milestones: list = field(default_factory=lambda: [0.5, 0.75, 0.9])
    """Milestones for the learning rate scheduler."""
    gamma: float = 0.33
    """Gamma for the learning rate scheduler."""

@dataclass
class TrainerConfig:
    max_steps: int = 1000
    """Maximum number of training steps."""
    log_every: int = 200
    """Logging interval."""
    save_every: int = 100
    """Model saving interval."""
    visualize_every: int = 500
    """Visualization interval."""
    size_decay_every: int = 100
    """Size decay interval."""
    weight_surface: float = 1e-3
    """Weight for the surface loss."""
    weight_sigma: float = 1e-3
    """Weight for the sigma loss."""
    weight_mip: float = 1e-3
    """Weight for the mip loss."""

@dataclass
class ExperimentConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: Experiment)
    """Config class for the Experiment."""
    load_config: Optional[Path] = None
    """Path to config YAML file."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    """Dataset config."""
    scene: SceneConfig = field(default_factory=SceneConfig)
    """Scene config."""
    render: RenderConfig = field(default_factory=RenderConfig)
    """Render config."""
    model: ModelConfig = field(default_factory=ModelConfig)
    """Model config."""
    occupancy: OccupancyConfig = field(default_factory=OccupancyConfig)
    """Occupancy config."""
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """Optimizer config."""
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """Scheduler config."""
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    """Trainer config."""
    output_path: Path = Path("results")
    """Path to save the results."""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d_%H-%M-%S"))
    """Timestamp for the experiment."""
    device: str = "cuda"
    """Device to use for training."""


    def get_output_path(self) -> Path:
        """Get the output path for the experiment."""
        return self.output_path / self.dataset.scene / self.timestamp
    
    def save_config(self) -> None:
        """Save config to base directory"""
        base_dir: Path = self.get_output_path()
        assert base_dir is not None
        base_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = base_dir / "config.yml"
        CONSOLE.log(f"Saving config to: {config_yaml_path}")
        config_yaml_path.write_text(yaml.dump(self), "utf8")



def initialize_output(config: ExperimentConfig):
    output_path = config.get_output_path()
    
    log.info(f"Saving outputs in: {output_path}")
    os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)
    
    writer = SummaryWriter(output_path, purge_step=0)
    return writer, output_path

def get_training_params(config: ExperimentConfig):
    scene = config.dataset.scene
    if scene in TANKS_TEMPLE_SCENES:
        weight_decay = config.optimizer.weight_decay
    else:
        weight_decay = (
            1e-5 if scene in ["materials", "ficus", "drums"]
            else 1e-6
        )
    
    return {
        "max_steps": config.trainer.max_steps,
        "target_sample_batch_size": 1 << 18,
        "weight_decay": weight_decay,
    }

def get_occupancy_params(config: ExperimentConfig):
    return {
        "grid_resolution": config.occupancy.grid_resolution,
        "grid_nlvl": config.occupancy.grid_nlvl,
    }

def get_render_parameters(config: ExperimentConfig):
    return {
        "render_step_size": config.render.render_step_size,
        "alpha_thre": config.render.alpha_thre,
        "cone_angle": config.render.cone_angle,
    }

def get_dataset_and_scene_parameters(config: ExperimentConfig, device):
    scene = config.dataset.scene
    init_batch_size = config.dataset.init_batch_size
    
    if scene in TANKS_TEMPLE_SCENES:
        data_path = os.path.join(config.dataset.data_root, scene)
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
            subject_id=scene, root_fp=str(config.dataset.data_root),
            split="train", num_rays=init_batch_size, device=device
        )
        test_dataset = SubjectLoader(
            subject_id=scene, root_fp=str(config.dataset.data_root),
            split="test", num_rays=None, device=device
        )
        aabb = torch.tensor(config.scene.aabb, device=device)
        near_plane = config.scene.near_plane
        far_plane = config.scene.far_plane
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

def initialize_radiance_field(config: ExperimentConfig, estimator, device):
    std_decay_factor = (config.model.std_final_factor / config.model.std_init_factor) ** (config.trainer.size_decay_every/config.trainer.max_steps)
    
    radiance_field = LagHashRadianceField(
        aabb=estimator.aabbs[-1], 
        n_features_per_gauss=config.model.n_features_per_gauss,
        n_neighbours=config.model.n_neighbours, 
        fixed_std=config.model.fixed_std,
        decay_factor=std_decay_factor, 
        splits=config.model.splits,
        n_gausses=config.model.n_gausses
    ).to(device)

    if config.model.load_model_path != "":
        state = torch.load(config.model.load_model_path, map_location=device)
        radiance_field.load_state_dict(state['model'])
        estimator.load_state_dict(state['occupancy'])
        log.info(f"Loaded model from {config.model.load_model_path}")
    
    return radiance_field

def initialize_optimizer(config, radiance_field, weight_decay):
    params_dict = { name : param for name, param in radiance_field.named_parameters()}
    
    gau_params, codebook_params, rest_params = [], [], []
    for name in params_dict:
        if ("means" in name) or ("stds" in name):
            gau_params.append(params_dict[name])
        elif "feats" in name:
            codebook_params.append(params_dict[name])
        else:
            rest_params.append(params_dict[name])

    gau_lr = config.optimizer.learning_rate * config.optimizer.gaussian_factor
    params = [
        {"params": gau_params, "lr": gau_lr, "eps": config.optimizer.eps, "weight_decay": 0.0},
        {"params": codebook_params, "lr": config.optimizer.learning_rate, "eps": config.optimizer.eps, "weight_decay": weight_decay},
        {"params": rest_params, "lr": config.optimizer.learning_rate, "eps": config.optimizer.eps, "weight_decay": weight_decay}
    ]
    
    return torch.optim.Adam(params)

def initialize_scheduler(config, optimizer):
    return torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=100),
            torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(m*config.trainer.max_steps) for m in config.scheduler.milestones], gamma=config.scheduler.gamma),
        ]
    )

def retrieve_image_data(img):
    render_bkgd = img["color_bkgd"]
    rays = img["rays"]
    pixels = img["pixels"]
    return render_bkgd, rays, pixels


class Experiment(nn.Module):

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = config.device
        self.output_path = config.output_path

    def run(self):
        device = self.device
        set_random_seed(42)
        
        writer, output_path = initialize_output(self.config)
        self.config.save_config()

        if self.config.dataset.scene in TANKS_TEMPLE_SCENES or self.config.dataset.scene in NERF_SYNTHETIC_SCENES:
            train_params = get_training_params(self.config)
            max_steps, target_sample_batch_size, weight_decay = (
                train_params["max_steps"],
                train_params["target_sample_batch_size"], 
                train_params["weight_decay"]
            )
            
            occupancy_params = get_occupancy_params(self.config)
            grid_resolution, grid_nlvl = (
                occupancy_params["grid_resolution"],
                occupancy_params["grid_nlvl"]
            )

            render_params = get_render_parameters(self.config)
            render_step_size, alpha_thre, cone_angle = (
                render_params["render_step_size"],
                render_params["alpha_thre"],
                render_params["cone_angle"]
            )

            dataset_params = get_dataset_and_scene_parameters(self.config, device)
            train_dataset, test_dataset, aabb, near_plane, far_plane, white_bg = (
                dataset_params["train_dataset"],
                dataset_params["test_dataset"],
                dataset_params["aabb"],
                dataset_params["near_plane"],
                dataset_params["far_plane"],
                dataset_params["white_bg"]
            )
        else:
            error_message = f"Invalid scene: {self.config.dataset.scene}"
            logging.error(error_message)
            raise ValueError(error_message)

        estimator = initialize_estimator(aabb, grid_resolution, grid_nlvl, device)

        grad_scaler = torch.cuda.amp.GradScaler(2**10)
        radiance_field = initialize_radiance_field(self.config, estimator, device)

        num_params = sum(p.numel() for p in radiance_field.parameters() if p.requires_grad)
        log.info(f"Number of parameters: {num_params/1e6:.2f}M")
        
        optimizer = initialize_optimizer(self.config, radiance_field, weight_decay)
        scheduler = initialize_scheduler(self.config, optimizer)
        
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
            if self.config.trainer.weight_surface:
                loss += self.config.trainer.weight_surface * loss_warm_up * surf_loss
            if self.config.trainer.weight_sigma and (not self.config.model.fixed_std):
                loss += self.config.trainer.weight_sigma * loss_warm_up * sigma_loss
            if self.config.trainer.weight_mip:
                loss += self.config.trainer.weight_mip * mip_loss

            optimizer.zero_grad()
            # do not unscale it because we are using Adam.
            grad_scaler.scale(loss).backward()
            optimizer.step()
            scheduler.step()

            if step % self.config.trainer.log_every == 0:
                elapsed_time = time.time() - tic
                log.info(
                    f"Training info: "
                    f"step={step} | elapsed_time={elapsed_time:.2f}s | "
                    f"whole_loss={loss:.5f} | surf_loss={surf_loss:.5f} | " 
                    f"sigma_loss={sigma_loss:.5f} | n_rendering_samples={n_rendering_samples:d} | "
                    f"max_depth={depth.max():.3f} | "
                )
            
            if (step % self.config.trainer.size_decay_every == self.config.trainer.size_decay_every-1) and self.config.model.fixed_std:
                radiance_field.mlp_base.encoding.update_factor()

            if step % self.config.trainer.save_every == 0:
                state_dict = {
                    "steps": step,
                    "model": radiance_field.state_dict(),
                    "occupancy": estimator.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                
                model_output_path = f"{output_path}/model.pth"
                torch.save(state_dict, model_output_path)
                log.info(f"Model saved to {model_output_path}")

                means = radiance_field.mlp_base.encoding.get_means()
                means = means.reshape(-1, means.shape[-1])
                means_cloud = trimesh.PointCloud(means.cpu().detach().numpy())
                if step > 0:
                    os.remove(os.path.join(output_path, f'means@{step-self.config.trainer.save_every:05d}.ply'))
                
                means_lod_path = os.path.join(output_path, f'means@{step:05d}.ply')
                means_cloud.export(means_lod_path)
                log.info(f"Means saved to {means_lod_path}")

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


def entrypoint():
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    
    config = tyro.cli(tyro.conf.SuppressFixed[tyro.conf.FlagConversionOff[ExperimentConfig]], description=convert_markup_to_ansi(__doc__))
    
    # Create an instance of the Experiment class
    experiment: Experiment = config.setup()
    assert isinstance(experiment, Experiment), "Experiment class not found in config"
    experiment.run()

if __name__ == "__main__":
    entrypoint()
