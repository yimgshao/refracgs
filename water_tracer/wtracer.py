from typing import Tuple
import logging
import os
from enum import IntEnum
import torch
import torch.utils.cpp_extension

from threedgrut.utils.timer import CudaTimer
from threedgrut.datasets.protocols import Batch

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
#

_wtracer_plugin = None


def load_wtracer_plugin(conf):
    global _wtracer_plugin
    if _wtracer_plugin is None:
        try:
            from . import libwtracer_cc as _C  # type: ignore
        except ImportError:
            from .setup_wtracer import setup_wtracer

            setup_wtracer(conf)
            import libwtracer_cc as _C  # type: ignore
        _wtracer_plugin = _C

# ----------------------------------------------------------------------------
#

class WTracer:
    def __init__(self, conf, device='cuda'):
        self.device = device
        self.conf = conf
        load_wtracer_plugin(conf)
        self.tracer_wrapper = _wtracer_plugin.WaterTracer(
            os.path.dirname(__file__),
            torch.utils.cpp_extension.CUDA_HOME
        )
        print("wtracer loaded.")

    def trace(self, rays_ori: torch.Tensor, rays_dir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rays_ori: torch.Tensor
            rays_dir: torch.Tensor

        Returns:
            hit_i: ...
            hit_t: ...
        """
        # safty check
        assert rays_ori.shape[-1] == 3 and rays_ori.shape[-1] == 3
        assert rays_ori.device.type == 'cuda' and rays_ori.device.type == 'cuda'
        assert rays_ori.shape == rays_dir.shape

        rays_shape = rays_ori.shape
        rays_ori = rays_ori.reshape(-1, 3)
        rays_dir = rays_dir.reshape(-1, 3)

        # lanch cpp code
        hit_i, hit_t = self.tracer_wrapper.trace(rays_ori, rays_dir)

        hit_i = hit_i.reshape(rays_shape[:-1])
        hit_t = hit_t.reshape(rays_shape[:-1])
        return hit_i, hit_t
        

    def build_bvh(self, vertices: torch.Tensor, triangles: torch.Tensor):
        """
        ...
        """
        # safty check
        assert vertices.shape[-1] == 3 and triangles.shape[-1] == 3
        assert vertices.dim() == 2 and triangles.dim() == 2
        assert vertices.device.type == 'cuda' and triangles.device.type == 'cuda'

        # lanch cpp code
        self.tracer_wrapper.build_bvh(vertices, triangles)

    def subdivide(self, vertices: torch.Tensor, triangles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
