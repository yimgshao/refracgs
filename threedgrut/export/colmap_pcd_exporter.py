from pathlib import Path

import numpy as np
import torch

from threedgrut.export.base import ExportableModel, ModelExporter
from threedgrut.utils.logger import logger

class ColmapPCDExporter(ModelExporter):
    @torch.no_grad()
    def export(self, model: ExportableModel, output_path: Path, dataset=None, conf=None, **kwargs) -> None:
        logger.info(f"exporting colmap point cloud to {output_path}...")
        positions = model.get_positions().detach().cpu().numpy()
        colors = model.get_features_albedo().detach().cpu().numpy()
        with open(output_path, 'w') as f:
            for i in range(positions.shape[0]):
                x, y, z = positions[i]
                r, g, b = colors[i]
                error = 1.0
                track = ''
                line = f"{i} {x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {error:.6f} {track}\n"
                f.write(line)
