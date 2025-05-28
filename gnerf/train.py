from __future__ import annotations

import os
import sys
import tyro
import torch
import traceback
import random
import numpy as np

home_dir = os.path.expanduser('~')
project_root = os.path.join(home_dir, 'gnerf')
sys.path.append(project_root)
sys.path.append("/workspace/gnerf/gnerf/lagrangian_hash/knn")

from gnerf.utils.config_utils import convert_markup_to_ansi
from gnerf.configs.method_configs import AnnotatedBaseConfigUnion
from gnerf.experiment import Trainer, TrainerConfig
from gnerf.utils.config_utils import CONSOLE


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_loop(config: TrainerConfig):
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    if config.random_seed is None:
        _set_random_seed(config.random_seed)
    trainer: Trainer = config.setup()
    trainer.setup()
    trainer.train()


def main(config: TrainerConfig):
    """Main function."""

    # Check if cuda is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare conifig
    config.set_device(device)
    config.set_timestamp()
    config.save_config()
    CONSOLE.log(f"Saving outputs in: {config.get_output_path()}")
    config.save_config()

    try:
        train_loop(config=config)
    except KeyboardInterrupt:
        # Print the stack trace
        CONSOLE.print(traceback.format_exc())


def entrypoint():
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    
    config = tyro.cli(AnnotatedBaseConfigUnion, description=convert_markup_to_ansi(__doc__))
    
    # Create an instance of the Experiment class
    assert isinstance(config, TrainerConfig), "TrainerConfig class not found in config"
    main(config)

if __name__ == "__main__":
    entrypoint()
