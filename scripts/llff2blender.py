"""Convert an LLFF-format dataset (poses_bounds.npy + images) to the
NeRF/D-NeRF blender-style json dataset format (transforms_train/test/val.json
+ train/test image folders), as used by threedgrut's NeRF dataloader.

Input layout (per scene):
    <input_dir>/poses_bounds.npy      (N, 17) float array: N x [3x5 pose | 2 bounds]
    <input_dir>/images/               full-resolution images
    <input_dir>/images_<ds>/          optional pre-downsampled images (factor <ds>)

Output layout (per scene):
    <output_dir>/train/<name>.png
    <output_dir>/test/<name>.png
    <output_dir>/transforms_train.json
    <output_dir>/transforms_test.json
    <output_dir>/transforms_val.json    (same frame as test)

Conversion recipe (verified to reproduce the reference dataset bit-exactly):
  1. LLFF pose columns [c0 c1 c2 t] are permuted to the NeRF c2w convention
     [right, up, back] = [c1, -c0, c2] with translation t.
  2. Poses are recentered with NeRF's recenter_poses(): a rigid transform
     built from the average view direction and average up vector
     (viewmatrix of the mean pose), so the mean camera center is at the origin.
  3. Translations are scaled by 4 / (3 * min_bound), where min_bound is the
     minimum over all near/far bounds in poses_bounds.npy (i.e. the scaled
     near plane becomes 4/3).
  4. camera_angle_x = 2 * atan(W / (2 * focal)) from the LLFF hwf entry
     (invariant to image downsampling).
  5. Split: the middle frame (index N//2 of the name-sorted list) goes to
     test/val, all others to train.

Usage:
    # Single scene
    python scripts/llff2blender.py -i data/nerfrac_llff/real_plant -o data/nerfrac/real_plant

    # Batch: process every scene subdirectory
    python scripts/llff2blender.py -i data/nerfrac_llff -o data/nerfrac --batch
"""

import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image


# ----------------------------------------------------------------------------
# Pose helpers (LLFF <-> NeRF conventions), following the original NeRF code.


def normalize(v):
    return v / np.linalg.norm(v)


def viewmatrix(z, up, pos):
    """Look-at matrix with columns [vec0, vec1, vec2, pos]."""
    vec2 = normalize(z)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, pos], 1)


def poses_avg(poses):
    """Average pose of a set of (N, 3, 4) c2w matrices (NeRF convention)."""
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    return viewmatrix(vec2, up, center)


def recenter_poses(poses):
    """Apply the inverse of the average pose to all poses (NeRF convention)."""
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    return poses_


def llff_to_nerf_poses(poses_bounds):
    """(N, 17) LLFF poses_bounds -> (N, 3, 5) poses in NeRF [right, up, back]
    convention, keeping the hwf column."""
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    # [c0 c1 c2 t hwf] -> [c1, -c0, c2, t, hwf]
    return np.stack(
        [poses[:, :, 1], -poses[:, :, 0], poses[:, :, 2], poses[:, :, 3], poses[:, :, 4]],
        axis=2,
    )


# ----------------------------------------------------------------------------
# Conversion


def convert_scene(scene_in, scene_out, downsample=4, test_index=None, resize=False):
    poses_bounds_path = os.path.join(scene_in, "poses_bounds.npy")
    images_dir = os.path.join(scene_in, "images")
    if not os.path.exists(poses_bounds_path):
        raise FileNotFoundError(f"poses_bounds.npy not found in {scene_in}")

    poses_bounds = np.load(poses_bounds_path)
    image_names = sorted(
        f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    n = len(image_names)
    if poses_bounds.shape[0] != n:
        raise ValueError(f"{scene_in}: {n} images but {poses_bounds.shape[0]} poses")

    # Poses: LLFF -> NeRF convention -> recenter -> scale translations
    poses = llff_to_nerf_poses(poses_bounds)
    poses = recenter_poses(poses)
    min_bound = poses_bounds[:, 15:].min()
    scale = 4.0 / (3.0 * min_bound)
    poses[:, :3, 3] *= scale

    # Intrinsics from the LLFF hwf entry (full resolution)
    hwf = poses[0, :3, 4]
    height, width, focal = hwf
    camera_angle_x = 2.0 * np.arctan(width / (2.0 * focal))

    # Split: middle frame to test/val, the rest to train
    if test_index is None:
        test_index = n // 2
    splits = {"train": [], "test": []}
    for i, name in enumerate(image_names):
        splits["test" if i == test_index else "train"].append((i, name))

    # Images: copy from images_<ds> if available, otherwise copy full resolution
    # (use --resize to force on-the-fly downsampling when images_<ds> is missing)
    ds_dir = os.path.join(scene_in, f"images_{downsample}")
    use_ds = downsample > 1 and os.path.isdir(ds_dir)
    for split in ("train", "test"):
        os.makedirs(os.path.join(scene_out, split), exist_ok=True)
    for split, items in splits.items():
        for _, name in items:
            dst = os.path.join(scene_out, split, name)
            if use_ds:
                shutil.copyfile(os.path.join(ds_dir, name), dst)
            elif resize:
                img = Image.open(os.path.join(images_dir, name))
                img = img.resize(
                    (img.width // downsample, img.height // downsample),
                    Image.LANCZOS,
                )
                img.save(dst)
            else:
                shutil.copyfile(os.path.join(images_dir, name), dst)

    # Json files
    def make_json(items, split):
        frames = []
        for i, name in items:
            c2w = np.eye(4)
            c2w[:3, :4] = poses[i, :3, :4]
            frames.append(
                {
                    "time": 0.0,
                    "rotation": 0.0,
                    "transform_matrix": c2w.tolist(),
                    "file_path": f"./{split}/{name}",
                }
            )
        return {"camera_angle_x": camera_angle_x, "frames": frames}

    outputs = {
        "transforms_train.json": make_json(splits["train"], "train"),
        "transforms_test.json": make_json(splits["test"], "test"),
        "transforms_val.json": make_json(splits["test"], "test"),
    }
    for fname, data in outputs.items():
        with open(os.path.join(scene_out, fname), "w") as f:
            json.dump(data, f, indent=4)

    print(
        f"Converted {scene_in} -> {scene_out}: "
        f"{len(splits['train'])} train / {len(splits['test'])} test, "
        f"scale={scale:.6f}, camera_angle_x={camera_angle_x:.6f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert an LLFF dataset (poses_bounds.npy) to the blender-style json format."
    )
    parser.add_argument("--input_dir", "-i", required=True, help="Scene directory, or dataset root with --batch")
    parser.add_argument("--output_dir", "-o", required=True, help="Output scene directory, or root with --batch")
    parser.add_argument("--batch", action="store_true", help="Process every subdirectory of input_dir as a scene")
    parser.add_argument("--downsample", type=int, default=4, help="Image downsample factor (default 4; uses images_<ds>/ if present)")
    parser.add_argument("--test_index", type=int, default=None, help="Index (name-sorted) of the test frame; default N//2")
    parser.add_argument("--resize", action="store_true", help="Resize images on the fly when images_<ds>/ is missing (default: copy full resolution)")
    args = parser.parse_args()

    if args.batch:
        for scene in sorted(os.listdir(args.input_dir)):
            scene_in = os.path.join(args.input_dir, scene)
            if not os.path.isdir(scene_in):
                continue
            convert_scene(scene_in, os.path.join(args.output_dir, scene), args.downsample, args.test_index, args.resize)
    else:
        convert_scene(args.input_dir, args.output_dir, args.downsample, args.test_index, args.resize)


if __name__ == "__main__":
    main()
