import os
import argparse
import numpy as np
import open3d as o3d

def merge_npy_to_ply(npy_dir, output_ply_path):
    all_points = []

    # Iterate over all .npy files in the directory
    for fname in os.listdir(npy_dir):
        if fname.endswith(".npy"):
            fpath = os.path.join(npy_dir, fname)
            points = np.load(fpath)
            if points.ndim != 2 or points.shape[1] != 3:
                print(f"Skipping {fname}: shape {points.shape} is not (N, 3)")
                continue
            all_points.append(points)

    if not all_points:
        print("No valid .npy files found.")
        return

    # Merge all point clouds
    merged_points = np.concatenate(all_points, axis=0)

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)

    # Save as a .ply file
    o3d.io.write_point_cloud(output_ply_path, pcd)
    print(f"✅ Saved merged point cloud to: {output_ply_path}")
    print(f"🔢 Total points: {len(merged_points)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple .npy point clouds into a single .ply file.")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Directory containing .npy files")
    parser.add_argument("--output_path", "-o", type=str, default='./pcd.ply', help="Path to save the merged .ply file")

    args = parser.parse_args()
    merge_npy_to_ply(args.input_dir, args.output_path)

