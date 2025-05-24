from __future__ import annotations

from torch.utils.data import Dataset

from dataclasses import dataclass, field
from typing import Type
from pathlib import Path

from utils.config_utils import InstantiateConfig

@dataclass
class BaseDatasetConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: BaseDataset)
    """Base class for dataset configuration."""
    name: str = "Synthetic"
    """Name of the dataset."""
    data_root: Path = Path("data/nerf_dataset")
    """Path to the dataset."""
    scene: str = "ficus"
    """Scene name."""
    near_plane: float = 2.0
    """Near clipping plane distance."""
    far_plane: float = 6.0
    """Far clipping plane distance."""
    init_batch_size: int = 1024
    """Initial batch size for training."""


class BaseDataset(Dataset):
    """Base class for datasets."""

    def __init__(self, config: BaseDatasetConfig):
        super().__init__()
        self.config: BaseDatasetConfig = config

    def get_weight_decay(self) -> float:
        """Get the weight decay value."""
        raise NotImplementedError("get_weight_decay() not implemented in BaseDataset")