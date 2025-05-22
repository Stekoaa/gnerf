"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from __future__ import annotations

import os
import sys
import time

import tyro
import numpy as np
import torch
import yaml
import trimesh
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
from scipy.ndimage import distance_transform_edt

home_dir = os.path.expanduser('~')
project_root = os.path.join(home_dir, 'gnerf')
sys.path.append(project_root)

from dataclasses import dataclass, field
from typing import Type, Optional
from datasets.nerf_synthetic import SubjectLoaderConfig
from datasets.tanks_and_temples import TanksTempleDataset
from utils.general_utils import set_random_seed, TANKS_TEMPLE_SCENES, NERF_SYNTHETIC_SCENES
from utils.loss_utils import calculate_loss_warmup, calculate_smooth_l1_loss
from utils.render_utils import render_image_with_occgrid, retrieve_image_data
from utils.config_utils import InstantiateConfig, convert_markup_to_ansi, CONSOLE
from nerfacc.estimators.occ_grid import OccGridEstimator
from radiance_fields.laghash import LagHashRadianceFieldConfig, LagHashRadianceField
from pathlib import Path
from configs.base_configs import BaseDatasetConfig, BaseDataset
from viewer import ViewerConfig, Viewer


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
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    """Viewer config."""
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


# def get_training_params(config: ExperimentConfig):
#     scene = config.dataset.scene
#     if scene in TANKS_TEMPLE_SCENES:
#         weight_decay = config.optimizer.weight_decay
#     else:
#         weight_decay = (
#             1e-5 if scene in ["materials", "ficus", "drums"]
#             else 1e-6
#         )
    
#     return {
#         "weight_decay": weight_decay,
#     }

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


def denormalize_points(points: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
    num_dim = points.shape[-1]
    aabb_min, aabb_max = torch.split(aabb, num_dim)
    return points * (aabb_max - aabb_min) + aabb_min


def initialize_optimizer(config: ExperimentConfig, radiance_field, weight_decay):
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

    def run(self):
        set_random_seed(42)
        
        CONSOLE.log(f"Saving outputs in: {self.output_path}")
        os.makedirs(os.path.join(self.output_path, 'test'), exist_ok=True)
        self.config.save_config()

        if self.config.dataset.scene in TANKS_TEMPLE_SCENES or self.config.dataset.scene in NERF_SYNTHETIC_SCENES:
            train_dataset: BaseDataset = self.config.dataset.setup(split="train", num_rays=self.config.dataset.init_batch_size, device=self.device)
            weight_decay = train_dataset.get_weight_decay()
        else:
            error_message = f"Invalid scene: {self.config.dataset.scene}"
            raise ValueError(error_message)

        self.estimator = OccGridEstimator(roi_aabb=self.config.model.aabb, resolution=self.config.model.grid_resolution, levels=self.config.model.grid_nlvl).to(self.device)

        grad_scaler = torch.cuda.amp.GradScaler(2**10)
        std_decay_factor = (self.config.trainer.std_final_factor / self.config.trainer.std_init_factor) ** (self.config.trainer.size_decay_every / self.config.trainer.max_steps)
        self.radiance_field: LagHashRadianceField = self.config.model.setup(std_decay_factor=std_decay_factor, device=self.device).to(self.device)

        num_params = sum(p.numel() for p in self.radiance_field.parameters() if p.requires_grad)
        CONSOLE.log(f"Number of parameters: {num_params/1e6:.2f}M")
        
        optimizer = initialize_optimizer(self.config, self.radiance_field, weight_decay)
        scheduler = initialize_scheduler(self.config, optimizer)

        self.viewer: Viewer = self.config.viewer.setup(radiance_field = self.radiance_field, 
                                                       estimator = self.estimator, 
                                                       near_plane = self.config.dataset.near_plane, 
                                                       render_step_size = self.config.trainer.render_step_size, 
                                                       cone_angle = self.config.trainer.cone_angle, 
                                                       alpha_thre = self.config.trainer.alpha_thre,
                                                       device = self.device)
        
        # Wait for the user to click the start button in viser
        while not self.viewer.start_button.value:
            print("Waiting for the start button to be clicked...")
            time.sleep(1)

        # training
        CONSOLE.log('Starting training')
        tic = time.time()
        self.distance_field = None
        for step in tqdm(range(self.config.trainer.max_steps + 1), desc="Training"):
            self.radiance_field.train()
            self.estimator.train()

            while self.viewer.pause_training:
                print("Training_paused...")
                time.sleep(1)

            i = torch.randint(0, len(train_dataset), (1,)).item()
            data = train_dataset[i]
            render_bkgd, rays, pixels = retrieve_image_data(data)

            def occ_eval_fn(x):
                density = self.radiance_field.query_density(x)
                if step > -1:
                    self.distance_field = distance_transform_edt(~self.estimator.binaries.squeeze(0).detach().cpu().numpy(), sampling=3/128)
                    self.distance_field = torch.tensor(self.distance_field, dtype=torch.float32).squeeze(0)
                    self.distance_field = self.distance_field.to(self.device)
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
            loss = torch.tensor(0.0, device=self.device)
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

            if self.distance_field is not None:
                def points_to_grid_coords(points, aabb):
                    xyz_min, xyz_max = aabb[:3], aabb[3:]
                    # Normalize to [0, 1]
                    normalized = (points - xyz_min) / (xyz_max - xyz_min)
                    # Convert to [-1, 1] for grid_sample
                    return normalized * 2 - 1
                

                def trilinear_interpolation(distance_field, points, aabb):
                    """
                    Args:
                        distance_field: (D, H, W) tensor on same device as points.
                        points: (N, 3) tensor in world coordinates.
                        aabb: (6,) tensor [min_x, min_y, min_z, max_x, max_y, max_z].

                    Returns:
                        distances: (N,) interpolated values at points.
                    """
                    D, H, W = distance_field.shape
                    device = points.device
                    dtype = points.dtype

                    xyz_min, xyz_max = aabb[:3], aabb[3:]
                    grid_size = torch.tensor([W, H, D], device=device, dtype=dtype)
                    voxel_size = (xyz_max - xyz_min) / grid_size

                    # Normalize to grid space
                    grid_coords = (points - xyz_min) / voxel_size  # shape (N, 3)

                    # Clamp to avoid indexing outside
                    min_val = torch.zeros(3, device=device, dtype=dtype)
                    max_val = grid_size - 1 - 1e-6
                    grid_coords = torch.clamp(grid_coords, min_val, max_val)

                    # Get integer and fractional parts
                    idx0 = grid_coords.floor().long()  # (N, 3)
                    d = grid_coords - idx0.float()     # (N, 3)
                    idx1 = idx0 + 1

                    x0, y0, z0 = idx0.unbind(dim=1)
                    x1, y1, z1 = idx1.unbind(dim=1)
                    dx, dy, dz = d.unbind(dim=1)

                    def get_vals(x, y, z):
                        x = torch.clamp(x, 0, W - 1)
                        y = torch.clamp(y, 0, H - 1)
                        z = torch.clamp(z, 0, D - 1)
                        return distance_field[z, y, x]

                    c000 = get_vals(x0, y0, z0)
                    c100 = get_vals(x1, y0, z0)
                    c010 = get_vals(x0, y1, z0)
                    c110 = get_vals(x1, y1, z0)
                    c001 = get_vals(x0, y0, z1)
                    c101 = get_vals(x1, y0, z1)
                    c011 = get_vals(x0, y1, z1)
                    c111 = get_vals(x1, y1, z1)

                    # Trilinear interpolation
                    c00 = c000 * (1 - dx) + c100 * dx
                    c01 = c001 * (1 - dx) + c101 * dx
                    c10 = c010 * (1 - dx) + c110 * dx
                    c11 = c011 * (1 - dx) + c111 * dx

                    c0 = c00 * (1 - dy) + c10 * dy
                    c1 = c01 * (1 - dy) + c11 * dy

                    c = c0 * (1 - dz) + c1 * dz  # (N,)

                    return c

                # Normalize points to aabb for grid_sample
                means = self.radiance_field.mlp_base.encoding.get_means()
                means = denormalize_points(means, self.config.model.aabb)
                # Rotate means 90 degrees around y axis
                rot = torch.tensor([[0, 0, 1],
                                    [0, 1, 0],
                                    [-1, 0, 0]], dtype=means.dtype, device=means.device)
                means = means @ rot.T
                grid_coords = points_to_grid_coords(means, self.config.model.aabb)
                grid_coords = grid_coords.view(1, -1, 1, 1, 3)  # shape (1, N, 1, 1, 3)

                # Sample the distance field
                aabb = self.config.model.aabb.to(means.device)
                distances = trilinear_interpolation(self.distance_field, means, aabb)
                loss += distances.mean() * 1e-2

            loss += calculate_smooth_l1_loss(rgb, pixels)
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
            means = denormalize_points(means, self.config.model.aabb)

            means = means.reshape(-1, means.shape[-1])
            means_cloud = trimesh.PointCloud(means.cpu().detach().numpy())

            color_coeffs = np.random.uniform(0.4, 1.0, size=(means_cloud.vertices.shape[0]))
            self.viewer.server.scene.add_point_cloud(
                "/means",
                points=means_cloud.vertices,
                colors=np.tile((0, 0, 255), means_cloud.vertices.shape[0]).reshape(-1, 3) * color_coeffs[:, None],
                point_size=0.002,
                point_shape="circle"
            )

            # Step 1: Generate voxel grid indices
            occ_grid = self.estimator.binaries.bool().squeeze(0)
            res = occ_grid.shape[0]
            device = occ_grid.device

            grid_coords = torch.stack(torch.meshgrid(
                torch.arange(res, device=device),
                torch.arange(res, device=device),
                torch.arange(res, device=device),
                indexing='ij'
            ), dim=-1).reshape(-1, 3)  # (res^3, 3)

            # Step 2: Select occupied voxels
            occupied_indices = grid_coords[occ_grid.view(-1)]  # (N, 3)

            # Step 3: Convert to world coordinates
            aabb_min = aabb[:3]
            aabb_max = aabb[3:]
            voxel_size = (aabb_max - aabb_min) / res
            occupied_centers = aabb_min + (occupied_indices + 0.5) * voxel_size  # (N, 3)

            # Step 4: Convert to NumPy and visualize using trimesh + your viewer
            occupied_cloud = trimesh.PointCloud(occupied_centers.cpu().numpy())
            self.viewer.server.scene.add_point_cloud(
                "/occupied_voxels",
                points=occupied_cloud.vertices,
                colors=np.tile((255, 0, 0), occupied_cloud.vertices.shape[0]).reshape(-1, 3),  # red color
                point_size=0.003,
                point_shape="circle"
            )

            self.viewer.ready = True


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
