from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from plyfile import PlyData
import numpy as np
import time


def camera_to_world(
    rays_ori: torch.Tensor,
    rays_dir: torch.Tensor,
    c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert rays from camera space to world space.

    Inputs shape:
    rays_ori: (B, H, W, 3), rays_dir: (B, H, W, 3), c2w: (B, 4, 4)

    Returns shape:
    world_ori: (B, H, W, 3), world_dir: (B, H, W, 3)
    """
    B, H, W, _ = rays_ori.shape

    # Flatten rays
    rays_ori = rays_ori.view(B, -1, 3)
    rays_dir = rays_dir.view(B, -1, 3)

    # Transform rays to world space
    R = c2w[:, :3, :3]  # (B, 3, 3)
    T = c2w[:, :3, 3]   # (B, 3)
    world_ori = torch.bmm(rays_ori, R.transpose(1, 2)) + T.unsqueeze(1)
    world_dir = torch.bmm(rays_dir, R.transpose(1, 2))

    world_ori = world_ori.view(B, H, W, 3)
    world_dir = world_dir.view(B, H, W, 3)

    world_dir = F.normalize(world_dir, dim=-1)

    return world_ori, world_dir


def world_to_camera(
    rays_ori: torch.Tensor,
    rays_dir: torch.Tensor,
    c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert rays from world space to camera space.

    Inputs shape:
    rays_ori: (B, H, W, 3), rays_dir: (B, H, W, 3), c2w: (B, 4, 4)

    Returns shape:
    camera_ori: (B, H, W, 3), camera_dir: (B, H, W, 3)
    """
    B, H, W, _ = rays_ori.shape

    # Flatten rays
    rays_ori = rays_ori.view(B, -1, 3)  # (B, H * W, 3)
    rays_dir = rays_dir.view(B, -1, 3)

    # Transform rays to world space
    R = c2w[:, :3, :3]  # (B, 3, 3)
    T = -c2w[:, :3, 3]   # (B, 3)
    camera_ori = torch.bmm(rays_ori + T.unsqueeze(1), R)
    camera_dir = torch.bmm(rays_dir, R)

    camera_ori = camera_ori.view(B, H, W, 3)
    camera_dir = camera_dir.view(B, H, W, 3)

    return camera_ori, camera_dir


def world_to_ndc(
    world_ori: torch.Tensor,
    world_dir: torch.Tensor, 
    intrinsics: list,
    near=1., far=100.
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert rays from world space to NDC space.

    Inputs shape:
    world_ori: (B, H, W, 3), world_dir: (B, H, W, 3), intrinsics: [fx, fy, cx, cy]
    near, far: scalar depth bounds

    Returns shape:
    ndc_ori: (B, H, W, 3), ndc_dir: (B, H, W, 3)
    """
    B, H, W, _ = world_ori.shape
    fx, fy, cx, cy = intrinsics

    # Shift ray origin to near plane
    t = - (near + world_ori[..., 2]) / world_dir[..., 2]
    rays_ori_near = world_ori + t.unsqueeze(-1) * world_dir

    # Project to NDC space
    ori_x, ori_y, ori_z = rays_ori_near[..., 0], rays_ori_near[..., 1], rays_ori_near[..., 2]
    dir_x, dir_y, dir_z = world_dir[..., 0], world_dir[..., 1], world_dir[..., 2]

    ndc_ori_x = - (2 * fx / W) * (ori_x / ori_z)
    ndc_ori_y = - (2 * fy / H) * (ori_y / ori_z)
    ndc_ori_z = 1 + (2 * near / ori_z)
    ndc_dir_x = - (2 * fx / W) * (dir_x / dir_z - ori_x / ori_z)
    ndc_dir_y = - (2 * fy / H) * (dir_y / dir_z - ori_y / ori_z)
    ndc_dir_z = - (2 * near) / ori_z

    ndc_ori = torch.stack([ndc_ori_x, ndc_ori_y, ndc_ori_z], dim=-1)
    ndc_dir = torch.stack([ndc_dir_x, ndc_dir_y, ndc_dir_z], dim=-1)
    ndc_dir = F.normalize(ndc_dir, dim=-1)

    return ndc_ori, ndc_dir


def ndc_to_world(
    points_ndc: torch.Tensor,
    intrinsics: list
) -> torch.Tensor:
    """
    Convert points from NDC space to world space.

    Inputs shape:
    points_ndc: (B, H, W, 3), intrinsics: [fx, fy, cx, cy]

    Returns shape:
    points_world: (B, H, W, 3)
    """
    B, H, W, _ = points_ndc.shape
    fx, fy, cx, cy = intrinsics

    z = 2 / (points_ndc[..., 2] - 1)
    x = - (W * points_ndc[..., 0] * z) / (2 * fx)
    y = - (H * points_ndc[..., 1] * z) / (2 * fy)
    points_world = torch.stack([x, y, z], dim=-1)

    return points_world


def least_squares_normal(points: torch.Tensor, mode='lstsq') -> torch.Tensor:
    """
    Select each point and its eight adjacent points, and fit a plane.
    Obtaining the normal vector of the plane.

    Inputs shape:
    points: (B, H, W, 3)
    mode: Which method should be used to calculate normals, svd or qr.

    Returns shape:
    normals: (B, H, W, 3)
    """
    B, H, W, C = points.shape
    assert C == 3, f'Input shape must be (B, H, W, 3), but the final dim is {C}, not 3.'

    points = points.permute(0, 3, 1, 2)  # (B, 3, H, W)
    padded = F.pad(points, (1, 1, 1, 1), mode='replicate')  # (B, 3, H+2, W+2)

    patches = padded.unfold(2, 3, 1).unfold(3, 3, 1)  # (B, 3, H, W, 3, 3)
    patches = patches.contiguous().view(B, 3, H, W, 9)  # (B, 3, H, W, 9)
    patches = patches.permute(0, 2, 3, 4, 1)  # (B, H, W, 9, 3)

    mean = patches.mean(dim=3, keepdim=True)
    centered = patches - mean  # (B, H, W, 9, 3)

    if mode == 'svd':
        # It may cause NaN in backward!!
        # SVD decomposition
        # Compute covariance matrix: (B, H, W, 3, 3)
        cov = centered.transpose(-2, -1) @ centered  # (B, H, W, 3, 3)

        # assert not torch.isnan(cov).any(), "NaN in covariance matrix"
        # assert not torch.isinf(cov).any(), "Inf in covariance matrix"
        # print("Covariance min:", cov.min().item(), "; max: ", cov.max().item())

        # SVD: cov = U @ S @ Vh
        _, _, Vh = torch.linalg.svd(cov)

        # The last row of Vh (corresponding to smallest singular value) is the normal
        normals = Vh[..., -1]  # (B, H, W, 3)
        normals = F.normalize(normals, dim=-1)

    elif mode == 'qr':
        # QR decomposition
        pts_xy, pts_z = torch.split(centered, [2, 1], dim=-1)
        Q, R = torch.linalg.qr(pts_xy)
        w = torch.inverse(R) @ Q.transpose(-2, -1) @ pts_z
        w = w.squeeze(-1)  # (B, H, W, 2)

        # Got normals
        ones = torch.ones_like(w[..., :1])
        normals = torch.cat([-w, ones], dim=-1)  # (B, H, W, 3)
        normals = F.normalize(normals, dim=-1)

    elif mode == 'eigh':
        # Warning! torch.linalg.eigh may cause NaN
        cov = centered.transpose(-2, -1) @ centered  # (B, H, W, 3, 3)
        _, eigvecs = torch.linalg.eigh(cov)
        normals = eigvecs[..., 0]  # Smallest eigenvalue direction
        normals = F.normalize(normals, dim=-1)

    elif mode == 'sobel':
        # Use Sobel filters to estimate gradients
        dx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=points.dtype, device=points.device).view(1, 1, 3, 3)
        dy = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=points.dtype, device=points.device).view(1, 1, 3, 3)

        dzdx = F.conv2d(points, dx.expand(3, 1, 3, 3), padding=1, groups=3)
        dzdy = F.conv2d(points, dy.expand(3, 1, 3, 3), padding=1, groups=3)

        dzdx = dzdx.permute(0, 2, 3, 1)  # (B, H, W, 3)
        dzdy = dzdy.permute(0, 2, 3, 1)

        normals = torch.cross(dzdx, dzdy, dim=-1)
        normals = F.normalize(normals, dim=-1)

    elif mode == 'lstsq':
        # Least squares fitting using torch.linalg.lstsq
        # Fit z = ax + by + c, then normal is [-a, -b, 1]
        pts_xy = centered[..., :2]  # (B, H, W, 9, 2)
        pts_z = centered[..., 2:]   # (B, H, W, 9, 1)

        ones = torch.ones_like(pts_xy[..., :1])  # (B, H, W, 9, 1)
        A = torch.cat([pts_xy, ones], dim=-1)    # (B, H, W, 9, 3)

        # Reshape for lstsq: (B*H*W, 9, 3) and (B*H*W, 9, 1)
        A_flat = A.view(-1, 9, 3)
        b_flat = pts_z.view(-1, 9, 1)

        # Solve least squares
        solution = torch.linalg.lstsq(A_flat, b_flat).solution  # (B*H*W, 3, 1)

        # Extract a, b -> normal = [-a, -b, 1]
        ab = solution[:, :2, 0]  # (B*H*W, 2)
        ones = torch.ones_like(ab[:, :1])  # (B*H*W, 1)
        normal = torch.cat([-ab, ones], dim=-1)  # (B*H*W, 3)
        normal = F.normalize(normal, dim=-1)

        # Reshape back to (B, H, W, 3)
        normals = normal.view(B, H, W, 3)
    
    else:
        raise NotImplementedError('Unrecognizable mode type!')

    return normals


def refract_rays(rays_dir: torch.Tensor, normals: torch.Tensor, eta=1.33) -> torch.Tensor:
    """
    Refraction conforms to the Snell's Law, which dictates how much a light ray “bends” when air into water.

    Inputs shape:
    rays_dir: (B, H, W, 3)
    normals: (B, H, W, 3)
    eta: refractive index of water

    Returns shape:
    refracted_dir: (B, H, W, 3)
    """
    B, H, W, C = rays_dir.shape
    assert C == 3, f'Input shape must be (B, H, W, 3), but the final dim is {C}, not 3.'

    # Flip normals if they point in the same direction as rays_dir
    flip_mask = torch.sum(normals * rays_dir, dim=-1) > 0
    normals = torch.where(flip_mask.unsqueeze(-1), -normals, normals)

    eta_inv = 1. / eta
    c1 = - torch.sum(normals * rays_dir, dim=-1)  # (B, H, W)
    c2 = torch.sqrt(1. - eta_inv**2 * (1. - c1**2))

    c1 = c1.unsqueeze(-1)  # (B, H, W, 1)
    c2 = c2.unsqueeze(-1)

    refracted_dir = eta_inv * (rays_dir + c1 * normals) - c2 * normals
    refracted_dir = F.normalize(refracted_dir, dim=-1)
    return refracted_dir

def calculate_intersect_and_normal(rays_ori, rays_dir, hit_i, vertices, triangles):
    # Gain vertices of all triangles
    triangle_points = vertices[triangles]  # (N, 3, 3)
    A = triangle_points[:, 0, :]  # (N, 3)
    B = triangle_points[:, 1, :]
    C = triangle_points[:, 2, :]

    # Calculate normals of every triangles
    e1 = B - A
    e2 = C - A
    normals_raw = torch.cross(e1, e2, dim=-1)  # (N, 3)
    normals = F.normalize(normals_raw, dim=-1)

    # Gain vertices and normals for all rays
    hit_triangle_points = vertices[triangles[hit_i]]  # (..., 3, 3)
    hit_A = hit_triangle_points[..., 0, :]  # (..., 3)
    hit_normals = normals[hit_i]  # (..., 3)

    # Calculate intersection points
    ray_to_plane = hit_A - rays_ori  # (..., 3)
    numerator = (ray_to_plane * hit_normals).sum(dim=-1)  # (...,)
    denominator = (rays_dir * hit_normals).sum(dim=-1)  # (...,)

    t = numerator / denominator  # (...,)
    hit_points = rays_ori + t.unsqueeze(-1) * rays_dir  # (..., 3)

    return hit_points, hit_normals

def fast_intersect_and_normal(rays_ori, rays_dir, hit_vertices):
    v0, v1, v2 = hit_vertices[..., 0, :], hit_vertices[..., 1, :], hit_vertices[..., 2, :]

    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = torch.cross(rays_dir, edge2, dim=-1)
    det = (edge1 * pvec).sum(dim=-1)

    inv_det = 1.0 / det
    tvec = rays_ori - v0
    qvec = torch.cross(tvec, edge1, dim=-1)
    t = inv_det * (edge2 * qvec).sum(dim=-1)

    hit_points = rays_ori + t.unsqueeze(-1) * rays_dir  # (..., 3)
    hit_normals = F.normalize(torch.cross(edge1, edge2, dim=-1), dim=-1)

    return hit_points, hit_normals

def compute_vertex_normals(vertices, triangles):
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0

    face_normals = torch.cross(edge1, edge2, dim=1)

    vertex_normals = torch.zeros_like(vertices)

    for i in range(3):
        vertex_normals.index_add_(0, triangles[:, i], face_normals)

    vertex_normals = F.normalize(vertex_normals, p=2, dim=1)

    return vertex_normals

def phong_intersect_normal(rays_ori, rays_dir, vertices, triangles, normals, hit_i):
    B, W, H, _ = rays_ori.shape
    N = B * W * H

    # Flatten the ray data
    rays_ori_flat = rays_ori.view(N, 3)
    rays_dir_flat = rays_dir.view(N, 3)
    hit_i_flat = hit_i.view(N)

    # Vertex indices of the triangle hit by each ray
    tri = triangles[hit_i_flat]  # (N, 3)
    v0 = vertices[tri[:, 0]]
    v1 = vertices[tri[:, 1]]
    v2 = vertices[tri[:, 2]]

    # Möller–Trumbore ray-triangle intersection
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = torch.cross(rays_dir_flat, edge2, dim=-1)
    det = (edge1 * pvec).sum(dim=-1)
    inv_det = 1.0 / det

    tvec = rays_ori_flat - v0
    u = (tvec * pvec).sum(dim=-1) * inv_det
    qvec = torch.cross(tvec, edge1, dim=-1)
    v = (rays_dir_flat * qvec).sum(dim=-1) * inv_det
    w = 1.0 - u - v

    t = (edge2 * qvec).sum(dim=-1) * inv_det
    hit_points = rays_ori_flat + t.unsqueeze(-1) * rays_dir_flat

    # Fetch the vertex normals of the hit triangles
    n0 = normals[tri[:, 0]]
    n1 = normals[tri[:, 1]]
    n2 = normals[tri[:, 2]]

    # Interpolate the normal and normalize
    hit_normals = F.normalize(w.unsqueeze(-1) * n0 + u.unsqueeze(-1) * n1 + v.unsqueeze(-1) * n2, dim=-1)

    # Reshape back to the original shape
    hit_points = hit_points.view(B, W, H, 3)
    hit_normals = hit_normals.view(B, W, H, 3)

    return hit_points, hit_normals

@torch.no_grad()
def hit_extract(vertices, triangles, hit_i, hit_t):
    # set hit_i = 0 for missed rays
    hit_i[hit_t < 0.] = 0

    # flatten hit_i
    hit_i = hit_i.view(-1)

    # extract hit mesh
    hit_triangle_id, new_hit_i = torch.unique(hit_i, return_inverse=True)
    hit_triangles = triangles[hit_triangle_id]
    hit_vert_id, new_hit_triangles = torch.unique(hit_triangles.view(-1), return_inverse=True)
    new_hit_triangles = new_hit_triangles.view(-1, 3)
    new_hit_vertices = vertices[hit_vert_id]

    # reshape back to ray shape
    new_hit_i = new_hit_i.view(hit_t.shape)

    return new_hit_vertices, new_hit_triangles, new_hit_i, hit_t

@torch.no_grad()
def mesh_subdivide(vertices: torch.Tensor, triangles: torch.Tensor):
    """
    Parameters:
        vertices: (N, 3)
        triangles: (M, 3)

    Returns:
        sub_vertices: (N + E, 3)
        sub_triangles: (M, 4, 3)
    """
    # N: num vertices, M: num triangles
    N = vertices.shape[0]
    M = triangles.shape[0]

    # Calculate edges for every triangles
    e01 = torch.stack([triangles[:, 0], triangles[:, 1]], dim=-1)  # (M, 2)
    e12 = torch.stack([triangles[:, 1], triangles[:, 2]], dim=-1)
    e20 = torch.stack([triangles[:, 2], triangles[:, 0]], dim=-1)
    edges = torch.cat([e01, e12, e20], dim=0)  # (3M, 2)

    # Remove duplicate edges
    edges = torch.stack([edges.min(dim=-1)[0], edges.max(dim=-1)[0]], dim=-1)
    edges_unique, inverse_indices = torch.unique(edges, dim=0, return_inverse=True)

    # Calculate new vertices and add them into vertices set
    midpoints = (vertices[edges_unique[:, 0]] + vertices[edges_unique[:, 1]]) * 0.5  # (E, 3)
    sub_vertices = torch.cat([vertices, midpoints], dim=0)  # (N + E, 3)

    # Calculate new vertices indices for every triangle
    m01, m12, m20 = (inverse_indices[:M] + N,
                     inverse_indices[M:2 * M] + N,
                     inverse_indices[2 * M:] + N)
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    # Calculate subdivided triangles
    sub_triangles = torch.stack([
        torch.stack([v0, m01, m20], dim=1),
        torch.stack([v1, m12, m01], dim=1),
        torch.stack([v2, m20, m12], dim=1),
        torch.stack([m01, m12, m20], dim=1)
    ], dim=1)

    return sub_vertices, sub_triangles

@torch.no_grad()
def sub_trace(vertices, triangles, rays_ori, rays_dir, hit_i, hit_t):
    """
    Parameters:
        vertices: (N, 3)
        triangles: (M, 4, 3)
        rays_ori: (B, H, W, 3)
        rays_dir: (B, H, W, 3)
        hit_i: (B, H, W)
        hit_t: (B, H, W)

    Returns:
        new_hit_i: (B, H, W)
        new_hit_t: (B, H, W)
    """
    sub_triangles = triangles[hit_i]  # (B, H, W, 4, 3)
    rays_ori_expand = rays_ori.unsqueeze(-2)  # (B, H, W, 1, 3)
    rays_dir_expand = rays_dir.unsqueeze(-2)  # (B, H, W, 1, 3)

    # Moller–Trumbore
    v0 = vertices[sub_triangles[..., 0]]  # (B, H, W, 4, 3)
    v1 = vertices[sub_triangles[..., 1]]
    v2 = vertices[sub_triangles[..., 2]]

    eps = 1e-8
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = torch.cross(rays_dir_expand, edge2, dim=-1)
    det = (edge1 * pvec).sum(dim=-1)

    mask_parallel = (det.abs() < eps)
    inv_det = torch.where(mask_parallel, torch.zeros_like(det), 1.0 / det)

    tvec = rays_ori_expand - v0
    u = torch.sum(tvec * pvec, dim=-1) * inv_det
    qvec = torch.cross(tvec, edge1, dim=-1)
    v = torch.sum(rays_dir_expand * qvec, dim=-1) * inv_det
    t = torch.sum(edge2 * qvec, dim=-1) * inv_det

    valid = (t > eps) & (u >= 0) & (v >= 0) & (u + v <= 1)
    t_valid = torch.where(valid, t, torch.full_like(t, float('inf')))
    min_t, min_i = torch.min(t_valid, dim=-1)
    # print(torch.isinf(min_t).sum().item())
    fallback_t, fallback_i = torch.min(t, dim=-1)

    final_t = torch.where(torch.isinf(min_t), fallback_t, min_t)
    final_i = torch.where(torch.isinf(min_t), fallback_i, min_i)

    return final_i, final_t


def generate_mesh(bounds, step=0.002):
    x = torch.arange(bounds[0], bounds[1] + step, step)
    y = torch.arange(bounds[2], bounds[3] + step, step)

    xx, yy = torch.meshgrid(x, y, indexing='ij')  # shape: [H, W]
    zz = torch.zeros_like(xx) - 1.
    verts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    rows, cols = xx.shape
    triangles = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            idx0 = i * cols + j
            idx1 = idx0 + 1
            idx2 = idx0 + cols
            idx3 = idx2 + 1
            triangles.append([idx0, idx2, idx1])
            triangles.append([idx1, idx2, idx3])

    triangles = torch.tensor(triangles, dtype=torch.int32)

    return verts, triangles

def load_mesh(ply_path):
    plydata = PlyData.read(ply_path)

    vertex_data = plydata['vertex']
    vertices = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=-1)
    vertices_tensor = torch.tensor(vertices, dtype=torch.float32)

    trangles_data = plydata['face']
    trangles = np.stack(trangles_data['vertex_indices'], axis=0)
    trangles_tensor = torch.tensor(trangles, dtype=torch.int32)
    
    return vertices_tensor, trangles_tensor


# Positional Encoding from https://github.com/yenchenlin/nerf-pytorch/blob/1f064835d2cca26e4df2d7d130daa39a8cee1795/run_nerf_helpers.py
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims=2):
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

