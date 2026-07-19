# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --- Modification Notice ---
# Modified by Yiming Shao
# Description: Add render train mode and export water surface geometry while rendering

import os
from pathlib import Path
import time

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import open3d as o3d

import threedgrut.datasets as datasets
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.logger import logger
from threedgrut.utils.timer import CudaTimer
from threedgrut.utils.misc import create_summary_writer
from threedgrut.utils.transforms import normals_world_to_camera
from threedgrut.utils import vis_utils


class Renderer:
    def __init__(
        self, model, conf, global_step, out_dir, path="", save_gt=True, writer=None, compute_extra_metrics=True, mode='test'
    ) -> None:

        if path:  # Replace the path to the test data
            conf.path = path

        self.model = model
        self.out_dir = out_dir
        self.save_gt = save_gt
        self.path = path
        self.conf = conf
        self.global_step = global_step
        self.writer = writer
        self.compute_extra_metrics = compute_extra_metrics

        if mode == 'train' or mode == 'mesh':
            self.dataset, self.dataloader = self.create_train_dataloader(conf)
        elif mode == 'test' or mode == 'air':
            self.dataset, self.dataloader = self.create_test_dataloader(conf)
        else:
            raise NotImplementedError('Unrecognizable mode type!')

        if conf.model.background.color == "black":
            self.bg_color = torch.zeros((3,), dtype=torch.float32, device="cuda")
        elif conf.model.background.color == "white":
            self.bg_color = torch.ones((3,), dtype=torch.float32, device="cuda")
        elif conf.model.background.color == "random":
            self.bg_color = torch.rand((3,), dtype=torch.float32, device="cuda")
        else:
            assert False, f"{conf.model.background.color} is not a supported background color."

    def create_train_dataloader(self, conf):
        """Create the train dataloader for the given configuration."""
        from threedgrut.datasets.utils import configure_dataloader_for_platform

        dataset = datasets.make_train(name=conf.dataset.type, config=conf)
        
        # Configure DataLoader arguments for the current platform
        dataloader_kwargs = configure_dataloader_for_platform({
            'num_workers': 8,
            'batch_size': 1,
            'shuffle': False,
            'collate_fn': None,
        })
        
        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
        return dataset, dataloader
    
    def create_test_dataloader(self, conf):
        """Create the test dataloader for the given configuration."""
        from threedgrut.datasets.utils import configure_dataloader_for_platform

        dataset = datasets.make_test(name=conf.dataset.type, config=conf)
        
        # Configure DataLoader arguments for the current platform
        dataloader_kwargs = configure_dataloader_for_platform({
            'num_workers': 8,
            'batch_size': 1,
            'shuffle': False,
            'collate_fn': None,
        })
        
        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
        return dataset, dataloader

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path, out_dir, path="", save_gt=True, writer=None, model=None, computes_extra_metrics=True, mode='test', surface=None
    ):
        """Loads checkpoint for test path.
        If path is stated, it will override the test path in checkpoint.
        If model is None, it will be loaded base on the
        """

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        global_step = checkpoint["global_step"]

        conf = checkpoint["config"]
        # overrides
        if conf["render"]["method"] == "3dgrt":
            conf["render"]["particle_kernel_density_clamping"] = True
            # conf["render"]["min_transmittance"] = 0.03
        conf["render"]["enable_kernel_timings"] = True

        object_name = Path(conf.path).stem
        experiment_name = conf["experiment_name"]
        writer, out_dir, run_name = create_summary_writer(conf, object_name, out_dir, experiment_name, use_wandb=False)

        if model is None:
            # Initialize the model and the optix context
            model = MixtureOfGaussians(conf)
            # Initialize the parameters from checkpoint
            model.init_from_checkpoint(checkpoint)
        model.build_acc()
        model.surface.surface_path = surface

        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=computes_extra_metrics,
            mode=mode
        )

    @classmethod
    def from_preloaded_model(
        cls, model, out_dir, path="", save_gt=True, writer=None, global_step=None, compute_extra_metrics=False
    ):
        """Loads checkpoint for test path."""

        conf = model.conf
        if global_step is None:
            global_step = ""
        model.build_acc()
        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=compute_extra_metrics,
        )

    @torch.no_grad()
    def render_all(self):
        """Render all the images in the test dataset and log the metrics."""

        # Criterions that we log during training
        criterions = {"psnr": PeakSignalNoiseRatio(data_range=1).to("cuda")}

        if self.compute_extra_metrics:
            criterions |= {
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda"),
                "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda"),
            }

        # switch to render mode to reduce inference time spend
        self.model.surface.switch_to_render_mode()

        output_path_renders = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "renders")
        os.makedirs(output_path_renders, exist_ok=True)

        output_path_air = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "air")
        os.makedirs(output_path_air, exist_ok=True)

        output_path_depth = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "depth")
        os.makedirs(output_path_depth, exist_ok=True)

        output_path_normals = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "normals")
        os.makedirs(output_path_normals, exist_ok=True)

        output_path_error_map = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "error_map")
        os.makedirs(output_path_error_map, exist_ok=True)

        if self.save_gt:
            output_path_gt = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "gt")
            os.makedirs(output_path_gt, exist_ok=True)

        # save water surface geometry
        output_path_surface = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "surface")
        os.makedirs(output_path_surface, exist_ok=True)
        self.model.surface.save_geometry(os.path.join(output_path_surface, 'water_surface.ply'))

        psnr = []
        ssim = []
        lpips = []
        inference_time = []
        test_images = []

        best_psnr = -1.0
        worst_psnr = 2**16 * 1.0

        best_psnr_img = None
        best_psnr_img_gt = None

        worst_psnr_img = None
        worst_psnr_img_gt = None

        logger.start_progress(task_name="Rendering", total_steps=len(self.dataloader), color="orange1")

        for iteration, batch in enumerate(self.dataloader):
            # Get the GPU-cached batch
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)

            # The first iter is much slower, which will cause inaccurate timing
            if iteration == 0:
                _ = self.model(gpu_batch)

            # Compute the outputs of a single batch
            inference_timer = CudaTimer()
            inference_timer.start()
            outputs = self.model(gpu_batch)
            inference_timer.end()
            outputs_air = self.model(gpu_batch, remove_surface=True)

            pred_rgb_full = outputs["pred_rgb"]
            pred_air_rgb_full = outputs_air["pred_rgb"]
            pred_depth_full = outputs_air["pred_dist"].expand(-1, -1, -1, 3)
            pred_depth_full = (pred_depth_full - pred_depth_full.min()) / (pred_depth_full.max() - pred_depth_full.min())
            pred_normals_full = normals_world_to_camera(outputs_air["pred_normals"], gpu_batch.T_to_world) / 2 + 0.5
            rgb_gt_full = gpu_batch.rgb_gt

            error_map = vis_utils.compute_error_map(pred_rgb_full, rgb_gt_full)
            
            # The values are already alpha composited with the background
            torchvision.utils.save_image(
                pred_rgb_full.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_renders, "{0:05d}".format(iteration) + ".png"),
            )
            torchvision.utils.save_image(
                pred_air_rgb_full.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_air, "{0:05d}".format(iteration) + ".png"),
            )
            torchvision.utils.save_image(
                pred_depth_full.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_depth, "{0:05d}".format(iteration) + ".png"),
            )
            torchvision.utils.save_image(
                pred_normals_full.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_normals, "{0:05d}".format(iteration) + ".png"),
            )
            torchvision.utils.save_image(
                error_map.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_error_map, "{0:05d}".format(iteration) + ".png"),
            )

            pred_img_to_write = pred_rgb_full[-1].clip(0, 1.0)
            gt_img_to_write = rgb_gt_full[-1].clip(0, 1.0)

            if self.writer is not None:
                test_images.append(pred_img_to_write)

            if self.save_gt:
                torchvision.utils.save_image(
                    rgb_gt_full.squeeze(0).permute(2, 0, 1),
                    os.path.join(output_path_gt, "{0:05d}".format(iteration) + ".png"),
                )

            # Compute the loss
            psnr_single_img = criterions["psnr"](outputs["pred_rgb"], gpu_batch.rgb_gt).item()
            psnr.append(psnr_single_img)  # evaluation on valid rays only
            logger.info(f"Frame {iteration}, PSNR: {psnr[-1]}")

            if psnr_single_img > best_psnr:
                best_psnr = psnr_single_img
                best_psnr_img = pred_img_to_write
                best_psnr_img_gt = gt_img_to_write

            if psnr_single_img < worst_psnr:
                worst_psnr = psnr_single_img
                worst_psnr_img = pred_img_to_write
                worst_psnr_img_gt = gt_img_to_write

            # evaluate on full image
            ssim.append(
                criterions["ssim"](
                    pred_rgb_full.permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )
            lpips.append(
                criterions["lpips"](
                    pred_rgb_full.clip(0, 1).permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )

            # Record the time
            print("refraction time: ", outputs["refrac_time_ms"])
            inference_time.append(inference_timer.timing())
            print("full inference time:", inference_time[-1])

            logger.log_progress(task_name="Rendering", advance=1, iteration=f"{str(iteration)}", psnr=psnr[-1])

        logger.end_progress(task_name="Rendering")

        mean_psnr = np.mean(psnr)
        mean_ssim = np.mean(ssim)
        mean_lpips = np.mean(lpips)
        std_psnr = np.std(psnr)
        mean_inference_time = np.mean(inference_time)

        table = dict(
            mean_psnr=mean_psnr,
            mean_ssim=mean_ssim,
            mean_lpips=mean_lpips,
            std_psnr=std_psnr,
        )

        if self.conf.render.enable_kernel_timings:
            table["mean_inference_time"] = f"{'{:.2f}'.format(mean_inference_time)}" + " ms/frame"

        logger.log_table(f"⭐ Test Metrics - Step {self.global_step}", record=table)

        if self.writer is not None:
            self.writer.add_scalar("psnr/test", mean_psnr, self.global_step)
            self.writer.add_scalar("ssim/test", mean_ssim, self.global_step)
            self.writer.add_scalar("lpips/test", mean_lpips, self.global_step)
            #self.writer.add_scalar("time/inference/test", mean_inference_time, self.global_step)

            if len(test_images) > 0:
                self.writer.add_images(
                    "image/pred/test",
                    torch.stack(test_images),
                    self.global_step,
                    dataformats="NHWC",
                )

            if best_psnr_img is not None:
                self.writer.add_images(
                    "image/best_psnr/test",
                    torch.stack([best_psnr_img, best_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

            if worst_psnr_img is not None:
                self.writer.add_images(
                    "image/worst_psnr/test",
                    torch.stack([worst_psnr_img, worst_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

        return mean_psnr, std_psnr, mean_inference_time
    
    @torch.no_grad()
    def render_air(self):
        """Render water removal images in the test dataset and log the metrics."""

        # Criterions that we log during training
        criterions = {"psnr": PeakSignalNoiseRatio(data_range=1).to("cuda")}

        if self.compute_extra_metrics:
            criterions |= {
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda"),
                "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda"),
            }

        # switch to render mode to reduce inference time spend
        self.model.surface.switch_to_render_mode()

        output_path_renders = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "renders")
        os.makedirs(output_path_renders, exist_ok=True)

        output_path_depth = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "depth")
        os.makedirs(output_path_depth, exist_ok=True)

        output_path_error_map = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "error_map")
        os.makedirs(output_path_error_map, exist_ok=True)

        if self.save_gt:
            output_path_gt = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "gt")
            os.makedirs(output_path_gt, exist_ok=True)

        psnr = []
        ssim = []
        lpips = []
        inference_time = []
        test_images = []

        best_psnr = -1.0
        worst_psnr = 2**16 * 1.0

        best_psnr_img = None
        best_psnr_img_gt = None

        worst_psnr_img = None
        worst_psnr_img_gt = None

        logger.start_progress(task_name="Rendering", total_steps=len(self.dataloader), color="orange1")

        for iteration, batch in enumerate(self.dataloader):
            # Get the GPU-cached batch
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)

            # The first iter is much slower, which will cause inaccurate timing
            if iteration == 0:
                _ = self.model(gpu_batch)

            # Compute the outputs of a single batch
            outputs = self.model(gpu_batch, remove_surface=True)

            pred_rgb_full = outputs["pred_rgb"]
            pred_depth_full = outputs["pred_dist"].expand(-1, -1, -1, 3)
            pred_depth_full = (pred_depth_full - pred_depth_full.min()) / (pred_depth_full.max() - pred_depth_full.min())
            rgb_gt_full = gpu_batch.rgb_gt

            error_map = vis_utils.compute_error_map(pred_rgb_full, rgb_gt_full)
            
            # The values are already alpha composited with the background
            torchvision.utils.save_image(
                pred_rgb_full.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_renders, "{0:05d}".format(iteration) + ".png"),
            )
            torchvision.utils.save_image(
                pred_depth_full.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_depth, "{0:05d}".format(iteration) + ".png"),
            )
            torchvision.utils.save_image(
                error_map.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_error_map, "{0:05d}".format(iteration) + ".png"),
            )

            pred_img_to_write = pred_rgb_full[-1].clip(0, 1.0)
            gt_img_to_write = rgb_gt_full[-1].clip(0, 1.0)

            if self.writer is not None:
                test_images.append(pred_img_to_write)

            if self.save_gt:
                torchvision.utils.save_image(
                    rgb_gt_full.squeeze(0).permute(2, 0, 1),
                    os.path.join(output_path_gt, "{0:05d}".format(iteration) + ".png"),
                )

            # Compute the loss
            psnr_single_img = criterions["psnr"](outputs["pred_rgb"], gpu_batch.rgb_gt).item()
            psnr.append(psnr_single_img)  # evaluation on valid rays only
            logger.info(f"Frame {iteration}, PSNR: {psnr[-1]}")

            if psnr_single_img > best_psnr:
                best_psnr = psnr_single_img
                best_psnr_img = pred_img_to_write
                best_psnr_img_gt = gt_img_to_write

            if psnr_single_img < worst_psnr:
                worst_psnr = psnr_single_img
                worst_psnr_img = pred_img_to_write
                worst_psnr_img_gt = gt_img_to_write

            # evaluate on full image
            ssim.append(
                criterions["ssim"](
                    pred_rgb_full.permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )
            lpips.append(
                criterions["lpips"](
                    pred_rgb_full.clip(0, 1).permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )

            logger.log_progress(task_name="Rendering", advance=1, iteration=f"{str(iteration)}", psnr=psnr[-1])

        logger.end_progress(task_name="Rendering")

        mean_psnr = np.mean(psnr)
        mean_ssim = np.mean(ssim)
        mean_lpips = np.mean(lpips)
        std_psnr = np.std(psnr)
        mean_inference_time = np.mean(inference_time)

        table = dict(
            mean_psnr=mean_psnr,
            mean_ssim=mean_ssim,
            mean_lpips=mean_lpips,
            std_psnr=std_psnr,
        )

        logger.log_table(f"⭐ Test Metrics - Step {self.global_step}", record=table)

        if self.writer is not None:
            self.writer.add_scalar("psnr/test", mean_psnr, self.global_step)
            self.writer.add_scalar("ssim/test", mean_ssim, self.global_step)
            self.writer.add_scalar("lpips/test", mean_lpips, self.global_step)

            if len(test_images) > 0:
                self.writer.add_images(
                    "image/pred/test",
                    torch.stack(test_images),
                    self.global_step,
                    dataformats="NHWC",
                )

            if best_psnr_img is not None:
                self.writer.add_images(
                    "image/best_psnr/test",
                    torch.stack([best_psnr_img, best_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

            if worst_psnr_img is not None:
                self.writer.add_images(
                    "image/worst_psnr/test",
                    torch.stack([worst_psnr_img, worst_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

        return mean_psnr, std_psnr, mean_inference_time
    
    @torch.no_grad()
    def render_mesh(self):
        # switch to render mode to reduce inference time spend
        self.model.surface.switch_to_render_mode()

        output_path = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "mesh")
        os.makedirs(output_path, exist_ok=True)

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= 0.004,
            sdf_trunc=0.02,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for iteration, batch in enumerate(self.dataloader):
            # get gpu batch and camera params
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)
            B, H, W, _ = gpu_batch.rgb_gt.shape
            fx, fy, cx, cy = gpu_batch.intrinsics
            c2w = gpu_batch.T_to_world.squeeze().cpu().numpy()

            # render clear
            outputs = self.model(gpu_batch, remove_surface=True)
            pred_rgb_full = outputs["pred_rgb"].squeeze(0)  # (H, W, 3)
            pred_depth_full = outputs["pred_dist"].squeeze(0)  # (H, W, 1)

            intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, cx = cx, cy = cy, fx = fx, fy = fy)
            extrinsic = np.linalg.inv(c2w)
            cam_o3d = o3d.camera.PinholeCameraParameters()
            cam_o3d.extrinsic = extrinsic
            cam_o3d.intrinsic = intrinsic

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(pred_rgb_full.detach().cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(pred_depth_full.detach().cpu().numpy(), order="C")),
                depth_trunc = 3, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )
            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)
            print(f'Image {iteration} processed')
        
        # Extract mesh using TSDF
        mesh = volume.extract_triangle_mesh()
        mesh_post = self.post_process_mesh(mesh, cluster_to_keep=100)
        o3d.io.write_triangle_mesh(os.path.join(output_path, f'mesh.ply'), mesh_post)

    def post_process_mesh(self, mesh, cluster_to_keep=100):
        """
        Post-process a mesh to filter out floaters and disconnected parts
        """
        import copy
        print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
        mesh_0 = copy.deepcopy(mesh)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        cluster_area = np.asarray(cluster_area)
        n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
        n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        mesh_0.remove_unreferenced_vertices()
        mesh_0.remove_degenerate_triangles()
        print("num vertices raw {}".format(len(mesh.vertices)))
        print("num vertices post {}".format(len(mesh_0.vertices)))
        return mesh_0

