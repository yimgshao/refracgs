import torch
import torch.nn.functional as F

def normals_world_to_camera(normals_world: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
        normals_world: (B, H, W, 3)
        T_to_world: (B, 4, 4)
    
    Returns:
        normals_camera: (B, H, W, 3)
    """
    B, H, W, _ = normals_world.shape

    # Extract rotation matrix R_cam_to_world and invert it to get R_world_to_cam
    R_cam_to_world = c2w[:, :3, :3]  # (B, 3, 3)
    R_world_to_cam = R_cam_to_world.transpose(1, 2)  # (B, 3, 3)

    # Flatten spatial dimensions for batch matrix multiplication
    normals_flat = normals_world.reshape(B, -1, 3)  # (B, H*W, 3)

    # Apply rotation: camera_normal = R_world_to_cam @ world_normal
    normals_cam_flat = torch.bmm(normals_flat, R_world_to_cam)  # (B, H*W, 3)

    # Normalize to unit length
    normals_cam_flat = F.normalize(normals_cam_flat, dim=-1)

    # Reshape back to (B, H, W, 3)
    normals_camera = normals_cam_flat.view(B, H, W, 3)

    return normals_camera

def points_camera_to_world(points_camera: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
        points_camera: (B, H, W, 3)
        c2w: (B, 4, 4)
    
    Returns:
        points_world: (B, H, W, 3)
    """
    B, H, W, _ = points_camera.shape
    assert c2w.shape == (B, 4, 4), f"Expected c2w shape ({B}, 4, 4), but got {c2w.shape}"

    R = c2w[:, :3, :3]
    T = c2w[:, :3, 3]

    points_flat = points_camera.view(B, -1, 3)  # (B, N, 3)
    points_world_flat = torch.matmul(points_flat, R.transpose(-1, -2)) + T[:, None, :]  # (B, N, 3)
    points_world = points_world_flat.view(B, H, W, 3)

    return points_world

def points_world_to_camera(points_world: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
        points_world: (B, H, W, 3)
        c2w: (B, 4, 4)
    
    Returns:
        points_camera: (B, H, W, 3)
    """
    B, H, W, _ = points_world.shape
    assert c2w.shape == (B, 4, 4), f"Expected c2w shape ({B}, 4, 4), but got {c2w.shape}"

    R = c2w[:, :3, :3]
    T = c2w[:, :3, 3]

    points_flat = points_world.view(B, -1, 3)  # (B, N, 3)
    points_camera_flat = torch.matmul(points_flat - T[:, None, :], R)
    points_camera = points_camera_flat.view(B, H, W, 3)

    return points_camera

def vectors_camera_to_world(vectors_camera: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
        vectors_camera: (B, H, W, 3)
        c2w: (B, 4, 4)
    
    Returns:
        vectors_world: (B, H, W, 3)
    """
    B, H, W, _ = vectors_camera.shape
    assert c2w.shape == (B, 4, 4), f"Expected c2w shape ({B}, 4, 4), but got {c2w.shape}"

    R = c2w[:, :3, :3]

    vectors_flat = vectors_camera.view(B, -1, 3)  # (B, N, 3)
    vectors_world_flat = torch.matmul(vectors_flat, R.transpose(-1, -2))  # (B, N, 3)
    vectors_world = vectors_world_flat.view(B, H, W, 3)

    vectors_world = F.normalize(vectors_world, dim=-1)

    return vectors_world

def vectors_world_to_camera(vectors_world: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """
    Parameters:
        vectors_world: (B, H, W, 3)
        c2w: (B, 4, 4)
    
    Returns:
        vectors_camera: (B, H, W, 3)
    """
    B, H, W, _ = vectors_world.shape
    assert c2w.shape == (B, 4, 4), f"Expected c2w shape ({B}, 4, 4), but got {c2w.shape}"

    R = c2w[:, :3, :3]

    vectors_flat = vectors_world.view(B, -1, 3)  # (B, N, 3)
    vectors_camera_flat = torch.matmul(vectors_flat, R)  # (B, N, 3)
    vectors_camera = vectors_camera_flat.view(B, H, W, 3)

    vectors_camera = F.normalize(vectors_camera, dim=-1)

    return vectors_camera

def project_points(points: torch.Tensor, intrinsics: list) -> torch.Tensor:
    """
    Parameters:
        points: (B, H, W, 3)
        intrinsics: [fx, fy, cx, cy]
    
    Returns:
        projected_points: (B, H, W, 2)
    """
    fx, fy, cx, cy = intrinsics
    B, H, W, _ = points.shape

    # Extract x, y, z
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2].clamp(min=1e-6)  # avoid division by zero

    # Project to image plane
    u = x * fx / z + cx
    v = y * fy / z + cy

    # Stack into (u, v)
    projected_points = torch.stack([u, v], dim=-1)  # (B, H, W, 2)

    return projected_points
