from __future__ import annotations

import os
import sys
import tyro

home_dir = os.path.expanduser('~')
project_root = os.path.join(home_dir, 'gnerf')
sys.path.append(project_root)

from utils.config_utils import convert_markup_to_ansi
from gnerf.configs.method_configs import AnnotatedBaseConfigUnion
from gnerf.experiment import Experiment


def entrypoint():
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    
    config = tyro.cli(AnnotatedBaseConfigUnion, description=convert_markup_to_ansi(__doc__))
    
    # Create an instance of the Experiment class
    experiment: Experiment = config.setup()
    assert isinstance(experiment, Experiment), "Experiment class not found in config"
    experiment.run()

if __name__ == "__main__":
    entrypoint()
