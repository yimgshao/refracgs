import torch
import torch.nn.functional as F
import numpy as np

from omegaconf import DictConfig, OmegaConf

from threedgrut.datasets.protocols import Batch
from threedgrut.utils.transforms import *

def transform_points_to_world(points: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    B, W, H, _ = points.shape

    points_homo = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (B, W, H, 4)
    points_flat = points_homo.view(B, -1, 4)  # (B, N, 4)
    points_world_flat = (c2w[:, None, :, :] @ points_flat.unsqueeze(-1)).squeeze(-1)[..., :3]  # (B, N, 3)
    points_world = points_world_flat.view(B, W, H, 3)

    return points_world

def gen_virtul_batch(batch: Batch, trans_noise=1.0, deg_noise=15.0, device='cuda') -> Batch:
    """
    Batch:  
        rays_ori: torch.Tensor  # [B, H, W, 3]
        rays_dir: torch.Tensor  # [B, H, W, 3]
        T_to_world: torch.Tensor  # [B, 4, 4]
        intrinsics: list  # [fx, fy, cx, cy]
    """
    B, H, W, _ = batch.rays_ori.shape
    translation_perturbation = (torch.rand(B, 3) - 0.5) * 2 * trans_noise
    rotation_perturbation_deg = (torch.rand(B, 3) - 0.5) * 2 * deg_noise
    rotation_perturbation_rad = torch.deg2rad(rotation_perturbation_deg)
    rx, ry, rz = rotation_perturbation_rad[..., 0], rotation_perturbation_rad[..., 1], rotation_perturbation_rad[..., 2]

    sin_rx, sin_ry, sin_rz = torch.sin(rx), torch.sin(ry), torch.sin(rz)
    cos_rx, cos_ry, cos_rz = torch.cos(rx), torch.cos(ry), torch.cos(rz)

    Rx = torch.stack([
        torch.stack([torch.ones(B), torch.zeros(B), torch.zeros(B)], dim=1),
        torch.stack([torch.zeros(B), cos_rx, -sin_rx], dim=1),
        torch.stack([torch.zeros(B), sin_rx, cos_rx], dim=1)
    ], dim=1).to(device)

    Ry = torch.stack([
        torch.stack([cos_ry, torch.zeros(B), sin_ry], dim=1),
        torch.stack([torch.zeros(B), torch.ones(B), torch.zeros(B)], dim=1),
        torch.stack([-sin_ry, torch.zeros(B), cos_ry], dim=1)
    ], dim=1).to(device)

    Rz = torch.stack([
        torch.stack([cos_rz, -sin_rz, torch.zeros(B)], dim=1),
        torch.stack([sin_rz, cos_rz, torch.zeros(B)], dim=1),
        torch.stack([torch.zeros(B), torch.zeros(B), torch.ones(B)], dim=1)
    ], dim=1).to(device)
    R_perturbation = torch.bmm(Rz, torch.bmm(Ry, Rx))

    c2w = torch.zeros_like(batch.T_to_world)
    c2w[..., :3, :3] = torch.bmm(batch.T_to_world[..., :3, :3], R_perturbation)
    c2w[..., :3, 3] = batch.T_to_world[..., :3, 3] + translation_perturbation
    c2w[..., 3, 3] = torch.ones(B)

    return Batch(batch.rays_ori.clone(), batch.rays_dir.clone(), c2w, intrinsics=batch.intrinsics)

def depth_normal_consistency(
    rays_ori: torch.Tensor, rays_dir: torch.Tensor, c2w: torch.Tensor, depth: torch.Tensor, normal: torch.Tensor
) -> torch.Tensor:
    """
    Parameters:
        rays_ori, rays_dir: (B, H, W, 3)
        c2w: (B, 4, 4)
        depth: (B, H, W, 1)
        normal: (B, H, W, 3)
    """
    points = rays_ori + depth * rays_dir
    points_world = transform_points_to_world(points, c2w)
    dx = points_world[:, 2:, 1:-1, :] - points_world[:, :-2, 1:-1, :]  # (B, H-2, W-2, 3)
    dy = points_world[:, 1:-1, 2:, :] - points_world[:, 1:-1, :-2, :]  # (B, H-2, W-2, 3)
    normal_from_depth = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    normal_from_render = normal[:, 1:-1, 1:-1, :]

    return (1 - (normal_from_depth * normal_from_render).sum(dim=-1)).mean()

def multiview_consistency(
    conf: DictConfig,
    cam_rays_ori: torch.Tensor,
    cam_rays_dir: torch.Tensor,
    intrinsics: list,
    render_outputs: dict,
    step=0
) -> torch.Tensor:
    pixel_noise, d_mask = multiview_reprojection(
        cam_rays_ori,
        cam_rays_dir,
        render_outputs["depth"],
        render_outputs["neighbor_depth"],
        render_outputs["pose"],
        render_outputs["neighbor_pose"],
        intrinsics
    )

    if conf.loss.multiview_consistency_use_geo_occ_aware:
        d_mask = d_mask & (pixel_noise < conf.loss.multiview_consistency_pixel_noise_th)
        weights = (1.0 / torch.exp(pixel_noise)).detach()
        weights[~d_mask] = 0
    else:
        weights = torch.ones_like(pixel_noise)
        weights[~d_mask] = 0

    multiview_loss = multiview_geo_loss(pixel_noise, d_mask, weights, conf.loss.multiview_consistency_geo_weight)
    return multiview_loss

def multiview_reprojection(
    cam_rays_ori: torch.Tensor,
    cam_rays_dir: torch.Tensor,
    reference_depth: torch.Tensor,
    neighboring_depth: torch.Tensor,
    reference_pose: torch.Tensor, 
    neighboring_pose: torch.Tensor,
    intrinsics: list,
) -> torch.Tensor:
    """
    Parameters:
        rays_ori, rays_dir: (B, H, W, 3)
        reference_depth, neighboring_depth: (B, H, W, 1)
        reference_pose, neighboring_pose: (B, 4, 4)
        intrinsics: [fx, fy, cx, cy]
    """
    B, H, W, _ = cam_rays_ori.shape

    ref_points_cam = cam_rays_ori + reference_depth * cam_rays_dir
    ref_points = points_camera_to_world(ref_points_cam, reference_pose)
    ref_points_nei = points_world_to_camera(ref_points, neighboring_pose)

    ref_project_to_nei = project_points(ref_points_nei, intrinsics)
    d_mask = (ref_project_to_nei[..., 0] > 0) & (ref_project_to_nei[..., 0] < W) &\
           (ref_project_to_nei[..., 1] > 0) & (ref_project_to_nei[..., 1] < H) & (ref_points_nei[..., 2] > 0.1)
    ref_project_to_nei[..., 0] = ref_project_to_nei[..., 0] * 2 / (W - 1) - 1
    ref_project_to_nei[..., 1] = ref_project_to_nei[..., 1] * 2 / (H - 1) - 1

    depth_view = neighboring_depth.permute(0, 3, 1, 2)  # -> (B, 1, H, W)
    grid = ref_project_to_nei  # (B, H, W, 2)
    map_z = torch.nn.functional.grid_sample(
        input=depth_view,
        grid=grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )  # → (B, 1, H, W)
    map_z = map_z.squeeze(1)  # (B, H, W)

    pts_in_nearest_cam = ref_points_nei / (ref_points_nei[..., 2:3].clamp(min=1e-6))  # (B, H, W, 3)
    pts_in_nearest_cam = pts_in_nearest_cam * map_z[..., None]  # (B, H, W, 3)

    pts_world = points_camera_to_world(pts_in_nearest_cam, neighboring_pose)
    pts_in_view_cam = points_world_to_camera(pts_world, reference_pose)
    pts_projections = project_points(pts_in_view_cam, intrinsics)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=cam_rays_ori.device),
        torch.arange(W, device=cam_rays_ori.device),
        indexing='ij'
    )
    pixels = torch.stack([grid_x, grid_y], dim=-1).float()  # (H, W, 2)
    pixels = pixels[None].expand(B, -1, -1, -1)  # (B, H, W, 2)
    pixel_noise = torch.norm(pts_projections - pixels, dim=-1)  # (B, H, W)

    return pixel_noise, d_mask

def multiview_geo_loss(pixel_noise, d_mask, weights, geo_weight=0.1):
    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
    return geo_loss

def multiview_pho_loss(
    d_mask: torch.Tensor,
    weights: torch.Tensor,
    pixels: torch.Tensor,
    gt_image_gray: torch.Tensor,
    render_pkg: dict,
    viewpoint_cam,
    nearest_cam,
    patch_size: int,
    sample_num: int,
    ncc_weight: float
):
    """
    Patch-based photometric consistency loss using LNCC.
    """
    loss = 0.0
    total_patch_size = patch_size * patch_size

    with torch.no_grad():
        # Step 1: sample valid pixels
        d_mask = d_mask.reshape(-1)
        valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
        if d_mask.sum() > sample_num:
            num_valid = d_mask.sum()
            index = torch.randperm(num_valid, device=d_mask.device)[:sample_num]
            valid_indices = valid_indices[index]

        weights = weights.reshape(-1)[valid_indices]
        pixels = pixels.reshape(-1, 2)[valid_indices]

        # Step 2: construct reference patch grid
        offsets = patch_offsets(patch_size, pixels.device)
        ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()

        H, W = gt_image_gray.squeeze().shape
        pixels_patch = ori_pixels_patch.clone()
        pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
        pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0

        ref_gray_val = F.grid_sample(
            gt_image_gray.unsqueeze(1),
            pixels_patch.view(1, -1, 1, 2),
            align_corners=True
        ).reshape(-1, total_patch_size)

        # Step 3: compute homography
        ref_to_neareast_r = nearest_cam.world_view_transform[:3, :3].T @ viewpoint_cam.world_view_transform[:3, :3]
        ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3, :3] + nearest_cam.world_view_transform[3, :3]

        ref_local_n = render_pkg["rendered_normal"].permute(1, 2, 0).reshape(-1, 3)[valid_indices]
        ref_local_d = render_pkg["rendered_distance"].reshape(-1)[valid_indices]

        H_ref_to_neareast = ref_to_neareast_r[None] - \
            torch.matmul(
                ref_to_neareast_t[None, :, None].expand(ref_local_d.shape[0], 3, 1),
                ref_local_n[:, :, None].permute(0, 2, 1)
            ) / ref_local_d[..., None, None]

        H_ref_to_neareast = torch.matmul(
            nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3),
            H_ref_to_neareast
        )
        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)

        # Step 4: warp patch to nearest view
        grid = patch_warp(H_ref_to_neareast.reshape(-1, 3, 3), ori_pixels_patch)
        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0

        _, nearest_image_gray = nearest_cam.get_image()
        sampled_gray_val = F.grid_sample(
            nearest_image_gray[None],
            grid.reshape(1, -1, 1, 2),
            align_corners=True
        ).reshape(-1, total_patch_size)

    # Step 5: compute LNCC loss
    ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
    mask = ncc_mask.reshape(-1)
    ncc = ncc.reshape(-1) * weights
    ncc = ncc[mask].squeeze()

    if mask.sum() > 0:
        ncc_loss = ncc_weight * ncc.mean()
        loss += ncc_loss

    return loss


def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask

def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing='xy')[::-1], dim=-1).view(1, -1, 2)

def patch_warp(H, uv):
    B, P = uv.shape[:2]
    H = H.view(B, 3, 3)
    ones = torch.ones((B,P,1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)

    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid_tmp = grid_tmp.reshape(B, P, 3)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid