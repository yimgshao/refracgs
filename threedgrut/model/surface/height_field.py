from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from omegaconf import DictConfig
import time

import open3d as o3d

from threedgrut.datasets.protocols import Batch
from threedgrut.model.surface.base import BasicSurface
from .utils import refract_rays, generate_mesh, get_embedder
from .utils import fast_intersect_and_normal, hit_extract, mesh_subdivide, sub_trace
from .utils import compute_vertex_normals, phong_intersect_normal
from water_tracer import WTracer


class HeightFieldSurface(BasicSurface):
    """ Height field water surface """   
    def __init__(self, conf: DictConfig, device='cuda'):
        super().__init__(conf, device=device)
        self.eta = conf.surface.eta
        self.recursive_iter = conf.surface.recursive_iter
        self.render_mode = False
        self._hf_net = HFNetwork(conf).to(device)
        self.wtracer = WTracer(conf)
        self.surface_path = None

        if self.surface_path is not None:
            self.vertices, self.triangles = self.load_geometry(self.surface_path)
            self.wtracer.build_bvh(self.vertices, self.triangles)
        else:
            self.vertices, self.triangles = generate_mesh(conf.surface.bounds, conf.surface.step)
            self.vertices, self.triangles = self.vertices.to(self.device), self.triangles.to(self.device)


    def forward(self, batch: Batch, iter: int) -> Batch:
        """
        Calculate the refraction effect of light on the water surface and deflect the light.

        Batch:  
            rays_ori: torch.Tensor  # [B, H, W, 3]
            rays_dir: torch.Tensor  # [B, H, W, 3]
            T_to_world: torch.Tensor  # [B, 4, 4]
            intrinsics: list  # [fx, fy, cx, cy]
        """
        rays_ori, rays_dir = batch.rays_ori, batch.rays_dir

        # recursive tracing
        hit_i, hit_t, vertices, triangles = self.recursive_tracing(rays_ori, rays_dir, self.recursive_iter)

        # save hit points
        if False:
            hit_points = rays_ori + hit_t.unsqueeze(-1) * rays_dir
            np.save(f"intersect/{time.time()}.npy", hit_points.reshape(-1, 3).cpu().numpy())

        # warmup iter
        if iter < 1000:
            vertices[..., 2] = vertices[..., 2].detach()

        if not self.render_mode and iter < 7500:
            hit_vertices = vertices[triangles[hit_i]]  # (..., 3, 3)
            refract_ori, hit_normal = fast_intersect_and_normal(rays_ori, rays_dir, hit_vertices)
        else:
            if self.render_mode:
                vertice_normals = self.normals
            else:
                vertice_normals = compute_vertex_normals(vertices, triangles)

            refract_ori, hit_normal = phong_intersect_normal(rays_ori, rays_dir, vertices, triangles, vertice_normals, hit_i)

        refract_dir = refract_rays(rays_dir, hit_normal, self.eta)
        refract_ori = torch.where(hit_t.unsqueeze(-1) > 0., refract_ori, rays_ori)
        refract_dir = torch.where(hit_t.unsqueeze(-1) > 0., refract_dir, rays_dir)
        batch.rays_ori, batch.rays_dir = refract_ori, refract_dir
        return batch

    def recursive_tracing(self, rays_ori: torch.Tensor, rays_dir: torch.Tensor, iter=3):
        if self.render_mode or self.surface_path is not None:
            vertices, triangles = self.vertices, self.triangles
            hit_i, hit_t = self.wtracer.trace(rays_ori, rays_dir)
            
        else:
            vertices, triangles = self.query_network(self.vertices, requires_grad=(iter==0)), self.triangles
            self.wtracer.build_bvh(vertices, triangles)
            hit_i, hit_t = self.wtracer.trace(rays_ori, rays_dir)
            for i in range(iter):
                # extract mesh
                vertices, triangles, hit_i, hit_t = hit_extract(vertices, triangles, hit_i, hit_t)

                # subdivide
                vertices, triangles = mesh_subdivide(vertices, triangles)
                triangles = triangles.int()

                # query network for more accurate z
                vertices = self.query_network(vertices, requires_grad=(i==iter-1))

                # sub trace
                sub_hit_i, sub_hit_t = sub_trace(vertices, triangles, rays_ori, rays_dir, hit_i, hit_t)
                triangles = triangles.view(-1, 3)
                hit_i = hit_i * 4 + sub_hit_i
                hit_t = sub_hit_t

        return hit_i, hit_t, vertices, triangles

    def query_network(self, vertices, requires_grad=True):
        xy = vertices[..., :2]
        with torch.set_grad_enabled(requires_grad):
            z = self._hf_net(xy)
        return torch.cat([xy, z], dim=-1)   
     
    def batch_query_network(self, vertices, requires_grad=True, batch_size=1024 * 8):
        results = []
        num_batches = (vertices.shape[0] + batch_size - 1) // batch_size
        for i in range(num_batches):
            batch = vertices[i * batch_size : (i + 1) * batch_size]
            result = self.query_network(batch, requires_grad=requires_grad)
            results.append(result)

        return torch.cat(results, dim=0)

    def subdivide_mesh(self, vertices: torch.Tensor, triangles: torch.Tensor, subdivide_num=3) -> Tuple[torch.Tensor, torch.Tensor]:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(triangles.cpu().numpy())

        mesh = mesh.subdivide_midpoint(number_of_iterations=subdivide_num)
        new_vertices = torch.from_numpy(np.asarray(mesh.vertices)).to(vertices.device).to(torch.float32)
        new_triangles = torch.from_numpy(np.asarray(mesh.triangles)).to(triangles.device).to(torch.int32)

        return new_vertices, new_triangles
    
    def switch_to_render_mode(self):
        self.render_mode = True
        if self.surface_path is None:
            self.vertices, self.triangles = generate_mesh(self.conf.surface.bounds, self.conf.surface.step / 2**self.recursive_iter)
            self.vertices, self.triangles = self.vertices.to(self.device), self.triangles.to(self.device)
            self.vertices = self.query_network(self.vertices, requires_grad=False)
        else:
            self.vertices, self.triangles = self.load_geometry(self.surface_path)

        self.normals = compute_vertex_normals(self.vertices, self.triangles)
        self.wtracer.build_bvh(self.vertices, self.triangles)

    def save_geometry(self, path):
        vertices, triangles = self.query_network(self.vertices), self.triangles
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices.detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(triangles.detach().cpu().numpy())
        o3d.io.write_triangle_mesh(path, mesh)

    def load_geometry(self, path):
        mesh = o3d.io.read_triangle_mesh(path)
        vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=self.device)
        triangles = torch.tensor(np.asarray(mesh.triangles), dtype=torch.int32, device=self.device)
        return vertices, triangles
    

# MLP + Positional Encoding
class HFNetwork(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.init_depth = conf.surface.init_depth

        input_dims = 2
        output_dims = 1
        multires = conf.surface.network.multires
        internal_dims = conf.surface.network.internal_dims
        hidden = conf.surface.network.hidden
        final_dim_init = conf.surface.network.final_dim_init
        self.embed_fn = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims)
            self.embed_fn = embed_fn
            input_dims = input_ch

        net = (torch.nn.Linear(input_dims, internal_dims, bias=False), torch.nn.LeakyReLU())
        for i in range(hidden-1):
            net = net + (torch.nn.Linear(internal_dims, internal_dims, bias=False), torch.nn.LeakyReLU())
            
        net = net + (torch.nn.Linear(internal_dims, output_dims, bias=False),)
        net[-1].apply(lambda m: nn.init.uniform_(m.weight, a=-final_dim_init, b=final_dim_init))
        self.net = torch.nn.Sequential(*net)

    def forward(self, p):
        if self.embed_fn is not None:
            p = self.embed_fn(p)
        out = self.net(p) + self.init_depth
        return out
  