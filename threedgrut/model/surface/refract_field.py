from typing import Tuple
import torch
from torch import nn
import time

from omegaconf import DictConfig

from threedgrut.datasets.protocols import Batch
from threedgrut.model.surface.base import BasicSurface
from threedgrut.model.surface.utils import world_to_ndc, ndc_to_world
from threedgrut.model.surface.utils import least_squares_normal, refract_rays, get_embedder


class RefractFieldSurface(BasicSurface):
    """ Reimplement of NDC space MLP based water surface in NeRFrac """
    def __init__(self, conf: DictConfig, device='cuda'):
        super().__init__(conf=conf, device=device)
        self.eta = conf.surface.eta
        self.init_depth = conf.surface.init_depth
        self.multires = conf.surface.network.multires
        self.intrinsics = None
        self.surface_points = None
        self.surface_normals = None

        # set network activation
        self._activation = nn.ReLU()

        input_dims = 5
        self.embed_fn = None
        if self.multires > 0:
            embed_fn, input_ch = get_embedder(self.multires, input_dims)
            self.embed_fn = embed_fn
            input_dims = input_ch
        
        # create network
        self._network = self.create_network(input_dims).to(self.device)

    def forward(self, batch: Batch, iter: int, save_surface=False) -> Batch:
        # Save intrinsics for ndc space transform
        if self.intrinsics is None:
            self.intrinsics = batch.intrinsics

        # Convert from world space to ndc space
        world_ori, world_dir = batch.rays_ori, batch.rays_dir
        ndc_ori, ndc_dir = world_to_ndc(world_ori, world_dir, self.intrinsics)

        # Put rays into surface network
        dt = self.query_network(ndc_ori, ndc_dir)

        # We use a warm-up iter
        if iter < 1000:
            dt = dt.detach()
        
        # Calculate the intersection of each ray with the surface
        init_t = (ndc_ori[..., 2] - self.init_depth) / (-ndc_dir[..., 2])  # check
        init_t = init_t.unsqueeze(-1)
        xs_ndc = ndc_ori + (init_t + dt) * ndc_dir

        # Convert back to world space and calculate normal
        xs_world = ndc_to_world(xs_ndc, self.intrinsics)
        xs_normals = least_squares_normal(xs_world)
        if save_surface:
            # save intersect points and normals if needed
            self.surface_points = xs_world.detach()
            self.surface_normals = xs_normals.detach()
        
        # Refract Rays
        refracted_dir = refract_rays(world_dir, xs_normals, eta=self.eta)

        batch.rays_ori, batch.rays_dir = xs_world, refracted_dir
        
        return batch
    
    def create_network(self, in_dim):
        layers = []
        for i in range(self.conf.surface.network.depth):
            if i == self.conf.surface.network.skip:
                layers.append(nn.Linear(in_dim + 2, self.conf.surface.network.width))
            else:
                layers.append(nn.Linear(in_dim, self.conf.surface.network.width))
            in_dim = self.conf.surface.network.width

        # output layer
        layers.append(nn.Linear(in_dim, 1))

        # init the final layer to follow NeRFrac
        layers[-1].apply(lambda m: nn.init.uniform_(m.weight, a=-1e-5, b=1e-5))

        return nn.ModuleList(layers)

    def query_network(self, ndc_ori, ndc_dir):
        h = torch.cat((ndc_dir, ndc_ori[..., :2]), dim=-1)
        if self.embed_fn is not None:
            h = self.embed_fn(h)
        for i in range(len(self._network) - 1):
            if i == self.conf.surface.network.skip:
                h = torch.cat((ndc_ori[..., :2], h), dim=-1)
            h = self._network[i](h)
            h = self._activation(h)
        h = self._network[-1](h)
        return h
    
    def save_geometry(self, path: str):
        """ Export surface geometry """
        pass
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """ Save state dict and intrinsics """
        state = super().state_dict(destination, prefix, keep_vars)
        state[prefix + "intrinsics"] = self.intrinsics
        return state

    def load_state_dict(self, state_dict, strict=True):
        """ Load state dict and intrinsics """
        if "intrinsics" in state_dict:
            self.intrinsics = state_dict["intrinsics"]
            del state_dict["intrinsics"]
        super().load_state_dict(state_dict, strict)

