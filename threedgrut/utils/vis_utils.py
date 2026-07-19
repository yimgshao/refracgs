import numpy as np
import torch
import matplotlib

def compute_error_map(pred: torch.Tensor, gt: torch.Tensor, mode='l1', normalize=True, colormap='hot') -> torch.Tensor:
    assert pred.shape == gt.shape, "pred_image and gt_image mush have the same shape"

    if mode == 'l2':
        error = (pred - gt) ** 2
    elif mode == 'l1':
        error = torch.abs(pred - gt)
    else:
        raise ValueError(f"mode must be l1 or l2: {mode}")

    error_gray = error.mean(dim=-1)  # [B, H, W]

    if normalize:
        B = error_gray.shape[0]
        for b in range(B):
            min_val = error_gray[b].min()
            max_val = error_gray[b].max()
            error_gray[b] = (error_gray[b] - min_val) / (max_val - min_val + 1e-8)

    # apply colormap
    cmap = matplotlib.colormaps[colormap]
    error_map = []
    for b in range(error_gray.shape[0]):
        color_mapped = cmap(error_gray[b].cpu().numpy())[..., :3]
        error_map.append(torch.tensor(color_mapped, dtype=torch.float32))

    error_map = torch.stack(error_map, dim=0)  # [B, H, W, 3]

    return error_map