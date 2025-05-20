from __future__ import annotations

import viser
import torch
import numpy as np
import viser.transforms as vtf
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Type

from datasets.utils import Rays
from utils.config_utils import InstantiateConfig
from utils.render_utils import render_image_with_occgrid


@dataclass
class ViewerConfig(InstantiateConfig):
    """Viewer configuration class."""

    _target: Type = field(default_factory=lambda: Viewer)
    """Viewer class."""
    port: int = 8080
    """Port for the viewer."""
    host: str = "localhost"
    """Host for the viewer."""
    width: int = 200
    """Width of the image rendered in the viewer."""


class Viewer:

    def __init__(self, config: ViewerConfig, radiance_field = None, estimator = None, near_plane = None, render_step_size = None, cone_angle = None, alpha_thre = None, device = "cpu"):

        self.config = config

        # Viser
        self.server = viser.ViserServer(port=8080)
        self.start_button: viser.GuiButtonHandle = self.server.add_button("Start Training")
        self.ready = False
        self.pause_training = False
        self.device = device

        # Store client references
        clients = set()

        @self.start_button.on_click
        def on_start_button_click(_):
            self.start_button.disabled == True

        @self.server.on_client_connect
        def handle_connect(client: viser.ClientHandle):
            clients.add(client)

            @client.camera.on_update
            def _(_: viser.CameraHandle) -> None:

                if not self.ready:
                    return
                # self.last_move_time = time.time()
                with self.server.atomic():
                    with torch.no_grad():
                        self.ready = False
                        self.pause_training = True
                        
                        width = config.width
                        aspect = client.camera.aspect
                        height = int(width / aspect) if aspect > 0 else width

                        # generate rays
                        rays = self.get_rays(client, width, height)

                        radiance_field.eval()
                        estimator.eval()

                        rgb, _, _, _, _, _ = render_image_with_occgrid(
                            radiance_field,
                            estimator,
                            rays,
                            near_plane=near_plane,
                            render_step_size=render_step_size,
                            render_bkgd=torch.ones(3, device=device),
                            cone_angle=cone_angle,
                            alpha_thre=alpha_thre
                        )

                        radiance_field.train()
                        estimator.train()

                        # Set the background image
                        np_image = rgb.detach().cpu().numpy()
                        np_image = np_image.reshape(height, width, 3)
                        client.scene.set_background_image(np_image)

                        self.pause_training = False

        @self.server.on_client_disconnect
        def handle_disconnect(client: viser.ClientHandle):
            clients.remove(client)

    def get_camera_state(self, client: viser.ClientHandle):
        R = vtf.SO3(wxyz=client.camera.wxyz)
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = torch.tensor(R.as_matrix(), dtype=torch.float32, device=self.device)
        pos = torch.tensor(client.camera.position, dtype=torch.float32, device=self.device)
        c2w = torch.concatenate([R, pos[:, None]], dim=1)
        return pos, c2w

    def get_intrinsic_matrix(self, client: viser.ClientHandle, width: int, height: int):
        # Get camera parameters
        vfov_rad = client.camera.fov

        # Compute focal length fx (assuming pinhole camera model)
        fx = fy = (height / 2) / np.tan(vfov_rad / 2)

        cx = width / 2
        cy = height / 2

        # Intrinsic matrix
        K = np.array([
            [fx,  0, cx],
            [0,  fy, cy],
            [0,   0,  1]
        ])

        return K
    
    def get_rays(self, client: viser.ClientHandle, width: int, height: int, opengl_camera=True) -> Rays:
        # Get camera parameters
        _, c2w = self.get_camera_state(client)
        K = self.get_intrinsic_matrix(client, width, height)

        # Create rays
        x, y = torch.meshgrid(
            torch.arange(width, device=self.device),
            torch.arange(height, device=self.device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()

        camera_dirs = F.pad(torch.stack([(x - K[0, 2] + 0.5) / K[0, 0],
                                            (y - K[1, 2] + 0.5) / K[1, 1] * (-1.0 if opengl_camera else 1.0)], dim=-1), 
                                            (0, 1), value=(-1.0 if opengl_camera else 1.0))

        directions = (camera_dirs @ c2w[:3, :3].T)
        origins = torch.broadcast_to(c2w[:3, 3], directions.shape)
        viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)

        origins = torch.reshape(origins, (width, height, 3))
        viewdirs = torch.reshape(viewdirs, (width, height, 3))

        return Rays(origins=origins, viewdirs=viewdirs)
