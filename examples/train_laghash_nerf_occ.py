"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from __future__ import annotations

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
import viser
import viser.transforms as vtf
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from datasets.utils import Rays
import io
from PIL import Image

home_dir = os.path.expanduser('~')
project_root = os.path.join(home_dir, 'gnerf')
sys.path.append(project_root)

from dataclasses import dataclass, field
from typing import Type, Optional
from datasets.nerf_synthetic import SubjectLoaderConfig, SubjectLoader
from datasets.tanks_and_temples import TanksTempleDataset
from utils.general_utils import set_random_seed, TANKS_TEMPLE_SCENES, NERF_SYNTHETIC_SCENES
from utils.loss_utils import calculate_loss_warmup, calculate_smooth_l1_loss
from utils.render_utils import render_image_with_occgrid, retrieve_image_data
from utils.config_utils import InstantiateConfig, convert_markup_to_ansi, CONSOLE
from nerfacc.estimators.occ_grid import OccGridEstimator
from radiance_fields.laghash import LagHashRadianceFieldConfig, LagHashRadianceField
from pathlib import Path
from configs.base_configs import BaseDatasetConfig, BaseDataset

# Disable warnings
warnings.filterwarnings("ignore")


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
    std_init_factor: float = 50
    """Initial standard deviation factor."""
    std_final_factor: float = 5
    """Final standard deviation factor."""
    size_decay_every: int = 100
    """Size decay interval."""
    weight_surface: float = 1e-3
    """Weight for the surface loss."""
    weight_sigma: float = 1e-3
    """Weight for the sigma loss."""
    weight_mip: float = 1e-3
    """Weight for the mip loss."""
    target_sample_batch_size: int = 1 << 18
    """Target sample batch size."""
    render_step_size: float = 0.005
    """Step size for rendering."""
    cone_angle: float = 0.0
    """Cone angle for rendering."""
    alpha_thre: float = 0.0
    """Alpha threshold for rendering."""

@dataclass
class ExperimentConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: Experiment)
    """Config class for the Experiment."""
    load_config: Optional[Path] = None
    """Path to config YAML file."""
    dataset: BaseDatasetConfig = field(default_factory=SubjectLoaderConfig)
    """Dataset config."""
    model: LagHashRadianceFieldConfig = field(default_factory=LagHashRadianceFieldConfig)
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
        "weight_decay": weight_decay,
    }

# def get_dataset_and_scene_parameters(config: ExperimentConfig, device):
#     scene = config.dataset.scene
#     init_batch_size = config.dataset.init_batch_size
    
#     if scene in TANKS_TEMPLE_SCENES:
#         data_path = os.path.join(config.dataset.data_root, scene)
#         train_dataset = TanksTempleDataset(
#             data_path, split="train", downsample=1, is_stack=False, num_rays=init_batch_size
#         )
#     else:
#         train_dataset: BaseDataset = config.dataset.setup(split="train", num_rays=config.dataset.init_batch_size, device=device)

#     return {
#         "train_dataset": train_dataset, 
#     }

def initialize_radiance_field(config: ExperimentConfig, estimator: OccGridEstimator, device):
    
    std_decay_factor = (config.trainer.std_final_factor / config.trainer.std_init_factor) ** (config.trainer.size_decay_every / config.trainer.max_steps)
    radiance_field: LagHashRadianceField = config.model.setup(std_decay_factor=std_decay_factor, device=device).to(device)

    if config.model.load_model_path != "":
        state = torch.load(config.model.load_model_path, map_location=device)
        radiance_field.load_state_dict(state['model'])
        estimator.load_state_dict(state['occupancy'])
        CONSOLE.log(f"Loaded model from {config.model.load_model_path}")
    
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

def initialize_scheduler(config: ExperimentConfig, optimizer):
    return torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=100),
            torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(m * config.trainer.max_steps) for m in config.scheduler.milestones], gamma=config.scheduler.gamma),
        ]
    )


class Experiment(nn.Module):

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.output_path = config.get_output_path()

        # Viser
        self.viser_server = viser.ViserServer(port=8080)
        self.start_button: viser.GuiButtonHandle = self.viser_server.add_button("Start Training")
        self.ready = False
        self.pause_training = False

        @self.start_button.on_click
        def on_start_button_click(_):
            self.start_button.disabled == True
        
        # Store client references
        clients = set()

        @self.viser_server.on_client_connect
        def handle_connect(client: viser.ClientHandle):
            clients.add(client)

            @client.camera.on_update
            def _(_: viser.CameraHandle) -> None:

                def get_camera_state(client: viser.ClientHandle):
                    R = vtf.SO3(wxyz=client.camera.wxyz)
                    R = R @ vtf.SO3.from_x_radians(np.pi)
                    R = torch.tensor(R.as_matrix(), dtype=torch.float32, device="cuda")
                    pos = torch.tensor(client.camera.position, dtype=torch.float32, device="cuda")
                    c2w = torch.concatenate([R, pos[:, None]], dim=1)
                    return pos, c2w
            
                def get_intrinsic_matrix(client: viser.ClientHandle, width: int, height: int):
                    # Get camera parameters
                    hfov_rad = client.camera.fov
            
                    # Compute focal length fx (assuming pinhole camera model)
                    fx = (width / 2) / np.tan(hfov_rad / 2)
                    fy = fx * height / width  # maintain aspect ratio (assuming square pixels)
            
                    cx = width / 2
                    cy = height / 2
            
                    # Intrinsic matrix
                    K = np.array([
                        [fx,  0, cx],
                        [0,  fy, cy],
                        [0,   0,  1]
                    ])
            
                    return K

                if not self.ready:
                    return
                # self.last_move_time = time.time()
                with self.viser_server.atomic():
                    with torch.no_grad():
                        self.ready = False
                        self.pause_training = True
                        position, c2w = get_camera_state(client)
                        # Set width and height based on aspect ratio
                        width = 200
                        aspect = client.camera.aspect
                        height = int(width / aspect) if aspect > 0 else width
                        opengl_camera = True

                        K = get_intrinsic_matrix(client, width, height)

                        # generate rays
                        x, y = torch.meshgrid(
                            torch.arange(width, device="cuda"),
                            torch.arange(height, device="cuda"),
                            indexing="xy",
                        )
                        x = x.flatten()
                        y = y.flatten()

                        camera_dirs = F.pad(
                            torch.stack(
                                [
                                    (x - K[0, 2] + 0.5) / K[0, 0],
                                    (y - K[1, 2] + 0.5)
                                    / K[1, 1]
                                    * (-1.0 if opengl_camera else 1.0),
                                ],
                                dim=-1,
                            ),
                            (0, 1),
                            value=(-1.0 if opengl_camera else 1.0),
                        )  # [num_rays, 3]

                        # [n_cams, height, width, 3]
                        directions = (camera_dirs @ c2w[:3, :3].T)
                        origins = torch.broadcast_to(c2w[:3, 3], directions.shape)
                        viewdirs = directions / torch.linalg.norm(
                            directions, dim=-1, keepdims=True
                        )

                        origins = torch.reshape(origins, (width, height, 3))
                        viewdirs = torch.reshape(viewdirs, (width, height, 3))

                        rays = Rays(origins=origins, viewdirs=viewdirs)

                        self.radiance_field.eval()
                        self.estimator.eval()

                        rgb, _, _, _, _, _ = render_image_with_occgrid(
                            self.radiance_field,
                            self.estimator,
                            rays,
                            # rendering options
                            near_plane=config.dataset.near_plane,
                            render_step_size=config.trainer.render_step_size,
                            render_bkgd=torch.ones(3, device="cuda"),
                            cone_angle=config.trainer.cone_angle,
                            alpha_thre=config.trainer.alpha_thre,
                        )

                        self.radiance_field.train()
                        self.estimator.train()
                        np_image = rgb.detach().cpu().numpy()

                        print(np_image.shape)

                        np_image = np_image.reshape(height, width, 3)

                        # print(f"Camera intrinsic matrix: {K}")
                        print(f"Camera position: {position}")
                        # print(f"Camera rotation: {c2w}")
                        print("-----------------------------------------------------------")

                        client.scene.set_background_image(np_image)

                        self.pause_training = False

        @self.viser_server.on_client_disconnect
        def handle_disconnect(client: viser.ClientHandle):
            clients.remove(client)

    def run(self):
        set_random_seed(42)
        
        CONSOLE.log(f"Saving outputs in: {self.output_path}")
        os.makedirs(os.path.join(self.output_path, 'test'), exist_ok=True)
        self.config.save_config()

        # Wait for the user to click the start button in viser
        while not self.start_button.value:
            print("Waiting for the start button to be clicked...")
            time.sleep(1)

        if self.config.dataset.scene in TANKS_TEMPLE_SCENES or self.config.dataset.scene in NERF_SYNTHETIC_SCENES:
            train_params = get_training_params(self.config)
            weight_decay = (
                train_params["weight_decay"]
            )

            train_dataset: BaseDataset = self.config.dataset.setup(split="train", num_rays=self.config.dataset.init_batch_size, device=self.device)
        else:
            error_message = f"Invalid scene: {self.config.dataset.scene}"
            raise ValueError(error_message)

        self.estimator = OccGridEstimator(roi_aabb=self.config.model.aabb, resolution=self.config.model.grid_resolution, levels=self.config.model.grid_nlvl).to(self.device)

        grad_scaler = torch.cuda.amp.GradScaler(2**10)
        self.radiance_field = initialize_radiance_field(self.config, self.estimator, self.device)

        num_params = sum(p.numel() for p in self.radiance_field.parameters() if p.requires_grad)
        CONSOLE.log(f"Number of parameters: {num_params/1e6:.2f}M")
        
        optimizer = initialize_optimizer(self.config, self.radiance_field, weight_decay)
        scheduler = initialize_scheduler(self.config, optimizer)
        
        # training
        CONSOLE.log('Starting training')
        tic = time.time()
        for step in tqdm(range(self.config.trainer.max_steps + 1), desc="Training"):
            self.radiance_field.train()
            self.estimator.train()

            while self.pause_training:
                print("Training_paused...")
                time.sleep(1)

            i = torch.randint(0, len(train_dataset), (1,)).item()
            data = train_dataset[i]
            render_bkgd, rays, pixels = retrieve_image_data(data)

            def occ_eval_fn(x):
                density = self.radiance_field.query_density(x)
                return density * self.config.trainer.render_step_size

            # update occupancy grid
            self.estimator.update_every_n_steps(
                step=step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=1e-2,
            )

            # render
            rgb, acc, depth, kl_div, n_rendering_samples, mip_loss = render_image_with_occgrid(
                self.radiance_field,
                self.estimator,
                rays,
                # rendering options
                near_plane=self.config.dataset.near_plane,
                render_step_size=self.config.trainer.render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=self.config.trainer.cone_angle,
                alpha_thre=self.config.trainer.alpha_thre,
            )

            if n_rendering_samples == 0:
                continue

            if self.config.trainer.target_sample_batch_size > 0:
                # dynamic batch size for rays to keep sample batch size constant.
                num_rays = int(len(pixels) * (self.config.trainer.target_sample_batch_size / float(n_rendering_samples)))
                train_dataset.update_num_rays(num_rays)

            # compute loss
            loss_warm_up = calculate_loss_warmup(step, self.config.trainer.max_steps)
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
                CONSOLE.log(
                    f"Training info: "
                    f"step={step} | elapsed_time={elapsed_time:.2f}s | "
                    f"whole_loss={loss:.5f} | surf_loss={surf_loss:.5f} | " 
                    f"sigma_loss={sigma_loss:.5f} | n_rendering_samples={n_rendering_samples:d} | "
                    f"max_depth={depth.max():.3f} | "
                )
            
            if (step % self.config.trainer.size_decay_every == self.config.trainer.size_decay_every-1) and self.config.model.fixed_std:
                self.radiance_field.mlp_base.encoding.update_factor()

            if step % self.config.trainer.save_every == 0:
                state_dict = {
                    "steps": step,
                    "model": self.radiance_field.state_dict(),
                    "occupancy": self.estimator.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                
                model_output_path = f"{self.output_path}/model.pth"
                torch.save(state_dict, model_output_path)
                CONSOLE.log(f"Model saved to {model_output_path}")

                means = self.radiance_field.mlp_base.encoding.get_means()
                means = means.reshape(-1, means.shape[-1])
                means_cloud = trimesh.PointCloud(means.cpu().detach().numpy())
                if step > 0:
                    os.remove(os.path.join(self.output_path, f'means@{step-self.config.trainer.save_every:05d}.ply'))
                
                means_lod_path = os.path.join(self.output_path, f'means@{step:05d}.ply')
                means_cloud.export(means_lod_path)
                CONSOLE.log(f"Means saved to {means_lod_path}")

            means = self.radiance_field.mlp_base.encoding.get_means()
            means = means.reshape(-1, means.shape[-1])
            means_cloud = trimesh.PointCloud(means.cpu().detach().numpy())

            color_coeffs = np.random.uniform(0.4, 1.0, size=(means_cloud.vertices.shape[0]))
            self.viser_server.scene.add_point_cloud(
                "/means",
                points=means_cloud.vertices,
                colors=np.tile((0, 0, 255), means_cloud.vertices.shape[0]).reshape(-1, 3) * color_coeffs[:, None],
                point_size=0.001,
                point_shape="circle"
            )

            self.ready = True


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
