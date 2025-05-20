"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Type, Literal
from configs.base_configs import BaseDatasetConfig, BaseDataset
from pathlib import Path

from .utils import Rays


def _load_renderings(root_fp: Path, scene: str, split: str):
    """Load images from disk."""
    data_dir = root_fp / scene
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        rgba = imageio.imread(fname)
        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal


@dataclass
class SubjectLoaderConfig(BaseDatasetConfig):
    """Configuration for the SubjectLoader."""

    _target: Type = field(default_factory=lambda: SubjectLoader)
    """Target class for the SubjectLoader."""
    scene: Literal["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"] = "ficus"
    """Scene name."""
    width: int = 800
    """Width of the images."""
    height: int = 800
    """Height of the images."""
    opengl_camera: bool = True
    """Use OpenGL camera convention."""


class SubjectLoader(BaseDataset):
    """Single subject data loader for training and evaluation."""

    def __init__(self, 
                 config: SubjectLoaderConfig,
                 split: Literal["train", "val", "trainval"] = "train",
                 color_bkgd_aug: str = "white",
                 num_rays: int = None,
                 batch_over_images: bool = True,
                 device: torch.device = torch.device("cpu")
                ):
        super().__init__(config)

        assert color_bkgd_aug in ["white", "black", "random"]
        self.num_rays = num_rays
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        if split == "trainval":
            _images_train, _camtoworlds_train, _focal_train = _load_renderings(
                self.config.data_root, self.config.scene, "train"
            )
            _images_val, _camtoworlds_val, _focal_val = _load_renderings(
                self.config.data_root, self.config.scene, "val"
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.focal = _focal_train
        else:
            self.images, self.camtoworlds, self.focal = _load_renderings(
                self.config.data_root, self.config.scene, split
            )
        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = torch.tensor(
            [
                [self.focal, 0, self.config.width / 2.0],
                [0, self.focal, self.config.height / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        self.images = self.images.to(device)
        self.camtoworlds = self.camtoworlds.to(device)
        self.K = self.K.to(device)
        assert self.images.shape[1:3] == (self.config.height, self.config.width)
        self.g = torch.Generator(device=device)
        self.g.manual_seed(42)

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(
                    3, device=self.images.device, generator=self.g
                )
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                    generator=self.g,
                )
            else:
                image_id = [index] * num_rays
            x = torch.randint(
                0,
                self.config.width,
                size=(num_rays,),
                device=self.images.device,
                generator=self.g,
            )
            y = torch.randint(
                0,
                self.config.height,
                size=(num_rays,),
                device=self.images.device,
                generator=self.g,
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.config.width, device=self.images.device),
                torch.arange(self.config.height, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.config.opengl_camera else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.config.opengl_camera else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, 4))
        else:
            origins = torch.reshape(origins, (self.config.height, self.config.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.config.height, self.config.width, 3))
            rgba = torch.reshape(rgba, (self.config.height, self.config.width, 4))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }
    
    def get_weight_decay(self) -> float:
        """Get the weight decay for the dataset."""
        
        weight_decay = (
            1e-5 if self.config.scene in ["materials", "ficus", "drums"]
            else 1e-6
        )

        return weight_decay
