import tyro

from typing import Dict, Union

from gnerf.experiment import TrainerConfig
from gnerf.datasets.nerf_synthetic import NeRFSyntheticDatasetConfig
from gnerf.radiance_fields.laghash import LagHashRadianceFieldConfig
from gnerf.lagrangian_hash.knn.knn_algorithms import TorchKNNConfig, FaissKNNConfig, FaissIVFKNNConfig, OptixKNNConfig


method_configs: Dict[str, Union[TrainerConfig]] = {}
descriptions = {
    "gnerf": "Hybrid representation of gaussian splatting and NeRF allowing for easy editing of the scene.",
}


method_configs["gnerf"] = TrainerConfig(
    random_seed=42,
    dataset=NeRFSyntheticDatasetConfig(),
    model=LagHashRadianceFieldConfig(
        knn_algorithm=FaissIVFKNNConfig()
    )
)


AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]