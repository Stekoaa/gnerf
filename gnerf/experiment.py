"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from __future__ import annotations

import os
import sys
import time
import torch
import yaml
import trimesh
import numpy as np

home_dir = os.path.expanduser('~')
project_root = os.path.join(home_dir, 'gnerf')
sys.path.append(project_root)

from dataclasses import dataclass, field
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from typing import Type, Optional
from tqdm import tqdm
from torch import nn
from torch.cuda.amp import GradScaler

from nerfacc.estimators.occ_grid import OccGridEstimator

from gnerf.configs.base_configs import BaseDatasetConfig, BaseDataset
from gnerf.optimization.optimizers import OptimizerConfig, initialize_optimizer
from gnerf.optimization.schedulers import SchedulerConfig, initialize_scheduler
from gnerf.radiance_fields.laghash import LagHashRadianceFieldConfig, LagHashRadianceField
from gnerf.utils.config_utils import InstantiateConfig, CONSOLE
from gnerf.utils.general_utils import TANKS_TEMPLE_SCENES, NERF_SYNTHETIC_SCENES
from gnerf.utils.general_utils import points_to_grid_coords, trilinear_interpolation, denormalize_points
from gnerf.utils.loss_utils import calculate_loss_warmup, calculate_smooth_l1_loss
from gnerf.utils.render_utils import render_image_with_occgrid, retrieve_image_data
from gnerf.viewer import ViewerConfig, Viewer


@dataclass
class ExperimentConfig(InstantiateConfig):

    load_config: Optional[Path] = None
    """Path to config YAML file."""
    dataset: BaseDatasetConfig = field(default_factory=BaseDatasetConfig)
    """Dataset config."""
    model: LagHashRadianceFieldConfig = field(default_factory=LagHashRadianceFieldConfig)
    """Occupancy config."""
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """Optimizer config."""
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """Scheduler config."""
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    """Viewer config."""
    output_path: Path = Path("results")
    """Path to save the results."""
    timestamp: Optional[str] = None
    """Timestamp for the experiment."""
    device: str = "cuda"
    """Device to use for training."""

    def set_timestamp(self) -> None:
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

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

    def set_device(self, device: str) -> None:
        """Set the device for the experiment."""
        self.device = device


@dataclass
class TrainerConfig(ExperimentConfig):
    """Configuration for the Trainer."""

    _target: Type = field(default_factory=lambda: Trainer)
    """Config class for the Experiment."""
    random_seed: Optional[int] = None
    """Random seed for reproducibility."""
    pause_on_start: bool = True
    """Pause the training until the user clicks the start button in the viewer."""
    max_steps: int = 2000
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
    surface_loss_swithch_step: int = 500
    """Step to switch to surface loss from attraction to spread on the surface."""
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


class Trainer(nn.Module):

    def __init__(self, config: TrainerConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.output_path = config.get_output_path()

    def setup(self):

        # Setup the dataset
        if self.config.dataset.scene in TANKS_TEMPLE_SCENES or self.config.dataset.scene in NERF_SYNTHETIC_SCENES:
            self.train_dataset: BaseDataset = self.config.dataset.setup(split="train", num_rays=self.config.dataset.init_batch_size, device=self.device)
            weight_decay = self.config.optimizer.weight_decay
            if self.config.optimizer.weight_decay is None:
                weight_decay = self.train_dataset.get_weight_decay()
        else:
            error_message = f"Invalid scene: {self.config.dataset.scene}"
            raise ValueError(error_message)

        # Prepare estimator and model
        self.estimator = OccGridEstimator(roi_aabb=self.config.model.aabb, resolution=self.config.model.grid_resolution, levels=self.config.model.grid_nlvl).to(self.device)

        self.grad_scaler = GradScaler(2**10)
        std_decay_factor = (self.config.std_final_factor / self.config.std_init_factor) ** (self.config.size_decay_every / self.config.max_steps)
        self.radiance_field: LagHashRadianceField = self.config.model.setup(std_decay_factor=std_decay_factor, device=self.device).to(self.device)

        num_params = sum(p.numel() for p in self.radiance_field.parameters() if p.requires_grad)
        CONSOLE.log(f"Number of parameters: {num_params/1e6:.2f}M")
        
        self.optimizer = initialize_optimizer(self.config, self.radiance_field, weight_decay)
        self.scheduler = initialize_scheduler(self.config, self.optimizer)

        # Initialize the viewer
        self.viewer: Viewer = self.config.viewer.setup(radiance_field = self.radiance_field, 
                                                       estimator = self.estimator, 
                                                       near_plane = self.config.dataset.near_plane, 
                                                       render_step_size = self.config.render_step_size, 
                                                       cone_angle = self.config.cone_angle, 
                                                       alpha_thre = self.config.alpha_thre,
                                                       device = self.device)
        
        # Wait for the user to click the start button in viser
        if self.config.pause_on_start:
            while not self.viewer.start_button.value:
                print("Waiting for the start button to be clicked...")
                time.sleep(1)

    def train(self):

        # Training
        CONSOLE.log('Starting training')
        tic = time.time()
        self.distance_field = None
        for step in tqdm(range(self.config.max_steps + 1), desc="Training"):
            self.radiance_field.train()
            self.estimator.train()

            while self.viewer.pause_training:
                print("Training_paused...")
                time.sleep(1)

            i = torch.randint(0, len(self.train_dataset), (1,)).item()
            data = self.train_dataset[i]
            render_bkgd, rays, pixels = retrieve_image_data(data)

            def occ_eval_fn(x):
                density = self.radiance_field.query_density(x)
                if step > -1:
                    self.distance_field = distance_transform_edt(~self.estimator.binaries.squeeze(0).detach().cpu().numpy(), sampling=3/128)
                    self.distance_field = torch.tensor(self.distance_field, dtype=torch.float32).squeeze(0)
                    self.distance_field = self.distance_field.to(self.device)
                return density * self.config.render_step_size

            # update occupancy grid
            self.estimator.update_every_n_steps(
                step=step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=1e-2,
            )

            # render
            rgb, acc, depth, kl_div, n_rendering_samples, mip_loss, weighted_squared_gausses_distance = render_image_with_occgrid(
                self.radiance_field,
                self.estimator,
                rays,
                # rendering options
                near_plane=self.config.dataset.near_plane,
                render_step_size=self.config.render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=self.config.cone_angle,
                alpha_thre=self.config.alpha_thre,
            )

            if n_rendering_samples == 0:
                continue

            if self.config.target_sample_batch_size > 0:
                # dynamic batch size for rays to keep sample batch size constant.
                num_rays = int(len(pixels) * (self.config.target_sample_batch_size / float(n_rendering_samples)))
                self.train_dataset.update_num_rays(num_rays)

            # compute loss
            loss = torch.tensor(0.0, device=self.device)
            loss_warm_up = calculate_loss_warmup(step, self.config.max_steps)
            mip_loss = mip_loss.mean() # distortion loss
            sigma_loss, surf_loss, i = 0, 0, 0
            
            # TODO: tu coś trzeba pomajstrować
            # for idx in range(radiance_field.n_levels):
            #     resolution = radiance_field.mlp_base.encoding.resolutions[idx]
            #     stds = radiance_field.mlp_base.encoding.get_stds(idx)
            #     if stds is not None:
            #         sigma_loss += calculate_lod_sigma_loss(resolution, stds)
            #         i += 1

            if self.distance_field is not None:
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

            loss += calculate_smooth_l1_loss(rgb, pixels)
            if self.config.weight_surface:
                if step > self.config.surface_loss_swithch_step:
                    surf_loss = weighted_squared_gausses_distance.sum() * self.config.weight_surface
                else:
                    surf_loss = distances.mean() * 1e-2
                loss += surf_loss
            if self.config.weight_sigma and (not self.config.model.fixed_std):
                loss += self.config.weight_sigma * loss_warm_up * sigma_loss
            if self.config.weight_mip:
                loss += self.config.weight_mip * mip_loss

            self.optimizer.zero_grad()
            self.grad_scaler.scale(loss).backward() # Do not unscale it because we are using Adam.
            self.optimizer.step()
            self.scheduler.step()

            if step % self.config.log_every == 0:
                elapsed_time = time.time() - tic
                CONSOLE.log(
                    f"Training info: "
                    f"step={step} | elapsed_time={elapsed_time:.2f}s | "
                    f"whole_loss={loss:.5f} | surf_loss={surf_loss:.5f} | " 
                    f"sigma_loss={sigma_loss:.5f} | n_rendering_samples={n_rendering_samples:d} | "
                    f"max_depth={depth.max():.3f} | "
                )
            
            if (step % self.config.size_decay_every == self.config.size_decay_every-1) and self.config.model.fixed_std:
                self.radiance_field.mlp_base.encoding.update_factor()

            if step % self.config.save_every == 0:
                state_dict = {
                    "steps": step,
                    "model": self.radiance_field.state_dict(),
                    "occupancy": self.estimator.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                
                model_output_path = f"{self.output_path}/model.pth"
                torch.save(state_dict, model_output_path)
                CONSOLE.log(f"Model saved to {model_output_path}")

                means = self.radiance_field.mlp_base.encoding.get_means()
                means = means.reshape(-1, means.shape[-1])
                means_cloud = trimesh.PointCloud(means.cpu().detach().numpy())
                if step > 0:
                    os.remove(os.path.join(self.output_path, f'means@{step-self.config.save_every:05d}.ply'))
                
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