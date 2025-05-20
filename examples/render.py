
import os
import logging
import numpy as np
import imageio
import torch
import tyro
import yaml

from dataclasses import dataclass, field
from tqdm import tqdm
from pathlib import Path
from typing import Type, Optional

from nerfacc.estimators.occ_grid import OccGridEstimator
from utils.config_utils import InstantiateConfig, convert_markup_to_ansi
from train_laghash_nerf_occ import ExperimentConfig, Experiment, OptimizerConfig, SchedulerConfig, TrainerConfig
from utils.render_utils import render_image_with_occgrid, retrieve_image_data
from utils.metric_utils import calculate_psnr
import trimesh

# A logger for this file
log = logging.getLogger(__name__)

@dataclass
class RendererConfig(InstantiateConfig):
    """Configuration for the Renderer."""
    _target: Type = field(default_factory=lambda: Renderer)
    """Target class for the Renderer."""
    load_config: Optional[Path] = None
    """Path to the configuration file."""


class Renderer:
        
    def __init__(self, config: RendererConfig):
        self.render_config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def render(self):
        # evaluation
        assert self.render_config.load_config is not None, "No config file provided"
        config = yaml.load(self.render_config.load_config.read_text(), Loader=yaml.Loader)
        assert isinstance(config, ExperimentConfig), "Invalid config file"

        log.info('Starting evaluation')

        # Load the model
        std_decay_factor = (config.trainer.std_final_factor / config.trainer.std_init_factor) ** (config.trainer.size_decay_every / config.trainer.max_steps)
        radiance_field = config.model.setup(std_decay_factor=std_decay_factor, device=self.device).to(self.device)

        model_state_dict = torch.load(config.get_output_path() / "model.pth", map_location=self.device)

        # # TODO Code for model editing
        # means = model_state_dict["model"]["mlp_base.encoding.means"]

        # for mean in means:
        #     if mean[0] > 0.5:
        #         mean[0] += 0.1

        # model_state_dict["model"]["mlp_base.encoding.means"] = means

        radiance_field.load_state_dict(model_state_dict['model'])

        for key, value in model_state_dict['model'].items():
            print(f"{key}: {value.shape}")

        for key, value in model_state_dict['occupancy'].items():
            print(f"{key}: {value.shape}")

        radiance_field.eval()

        # Define estimator
        estimator = OccGridEstimator(roi_aabb=config.model.aabb, resolution=config.model.grid_resolution, levels=config.model.grid_nlvl).to(self.device)
        model_state_dict['occupancy']['binaries'].fill_(True)
        estimator.load_state_dict(model_state_dict['occupancy'])
        estimator.eval()

        # Load the dataset
        test_dataset = config.dataset.setup(split="test", device=self.device)
        
        psnrs = []
        with torch.no_grad():
            for i in tqdm(range(len(test_dataset)), desc='Evaluation'):
                render_bkgd, rays, pixels = retrieve_image_data(test_dataset[i])
                rgb, _, _, _, _, _ = render_image_with_occgrid(
                    radiance_field,
                    estimator,
                    rays,
                    # rendering options
                    near_plane=config.dataset.near_plane,
                    render_step_size=config.trainer.render_step_size,
                    render_bkgd=render_bkgd,
                    cone_angle=config.trainer.cone_angle,
                    alpha_thre=config.trainer.alpha_thre,
                )
                
                psnrs.append(calculate_psnr(rgb, pixels))
                imageio.imwrite(
                    config.get_output_path() / f"test/rgb_test_{i}.png",
                    (rgb.cpu().numpy() * 255).astype(np.uint8),
                )

        psnr_avg = sum(psnrs) / len(psnrs)
        logging.info(f"Evaluation: psnr_avg={psnr_avg}")
        with open(config.get_output_path() / f"metrics.txt", "w") as fp:
            fp.write(f"PSNR:{psnr_avg:.3f}")


def entrypoint():
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    
    config = tyro.cli(tyro.conf.SuppressFixed[tyro.conf.FlagConversionOff[RendererConfig]], description=convert_markup_to_ansi(__doc__))
    
    # Create an instance of the Experiment class
    renderer: Renderer = config.setup()
    assert isinstance(renderer, Renderer), "Experiment class not found in config"
    renderer.render()

if __name__ == "__main__":
    entrypoint()