import torch
from torch import nn

from omegaconf import DictConfig

from threedgrut.datasets.protocols import Batch


class BasicSurface(nn.Module):
    """ Basic class for water surface """
    
    def __init__(self, conf: DictConfig, device='cuda'):
        super().__init__()
        self.conf = conf
        self.device = device

    def forward(self, batch: Batch, iter: int) -> Batch:
        """
        Calculate the refraction effect of light on the water surface and deflect the light.
        In basic water surface, no refraction effect on the rays.

        Batch:  
            rays_ori: torch.Tensor  # [B, H, W, 3]
            rays_dir: torch.Tensor  # [B, H, W, 3]
            T_to_world: torch.Tensor  # [B, 4, 4]
            intrinsics: list  # [fx, fy, cx, cy]
        """
        return batch
    
    def switch_to_render_mode(self):
        pass
    
    def save_geometry(self, path: str):
        pass
    