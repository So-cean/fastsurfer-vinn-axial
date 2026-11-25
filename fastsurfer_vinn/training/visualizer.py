# -*- coding: utf-8 -*-
"""Validation visualizer — overlay PNG + TensorBoard per epoch."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch


class Visualizer:
    """Visualize one validation sample per epoch.

    Picks the first batch from ``val_loader`` once.  Each epoch it slices
    the middle axial plane, computes prediction, and saves a 3-panel figure
    (MRI | GT seg | overlay) as PNG under ``{output_dir}/visuals/``.

    If a ``writer`` (SummaryWriter) is provided, the image is also logged
    to TensorBoard under ``val/overlay``.
    """

    def __init__(self, val_loader, device, output_dir, num_classes=35):
        self.val_loader = val_loader
        self.device = device
        self.out_dir = Path(output_dir) / "visuals"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        # 35 discrete colors: 20 from tab20 + 15 from tab20b
        colors = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors
        self.cmap = mcolors.ListedColormap(colors[:num_classes])
        self._ref_batch = None

    def _get_ref(self):
        if self._ref_batch is None:
            for t1, seg in self.val_loader:
                self._ref_batch = (t1[:1].to(self.device), seg[:1].to(self.device))
                break
        return self._ref_batch

    @torch.no_grad()
    def __call__(self, model, epoch, writer=None):
        model.eval()
        t1, seg_gt = self._get_ref()

        seg_pred = model(t1)
        if seg_pred.dim() == 6:  # 3D deep supervision
            seg_pred = seg_pred[:, 0]
        elif seg_pred.dim() == 5:  # 2D deep supervision
            seg_pred = seg_pred[:, 0]
        seg_pred = seg_pred.argmax(dim=1)

        if t1.ndim == 4:  # 2D: (B, C, H, W)
            mri  = t1[0, 0].cpu().numpy()
            gt   = seg_gt[0].cpu().numpy()
            pred = seg_pred[0].cpu().numpy()
        else:  # 3D: (B, C, D, H, W)
            z = t1.shape[-1] // 2
            mri  = t1[0, 0, :, :, z].cpu().numpy()
            gt   = seg_gt[0, :, :, z].cpu().numpy()
            pred = seg_pred[0, :, :, z].cpu().numpy()
        mri = (mri - mri.min()) / (mri.max() - mri.min() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(mri, cmap="gray")
        axes[0].set_title("MRI")
        axes[1].imshow(gt, cmap=self.cmap, vmin=0, vmax=self.num_classes - 1)
        axes[1].set_title("GT")
        axes[2].imshow(pred, cmap=self.cmap, vmin=0, vmax=self.num_classes - 1)
        axes[2].set_title("Pred")
        for ax in axes:
            ax.axis("off")

        path = self.out_dir / f"epoch_{epoch:04d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if writer is not None:
            from PIL import Image
            pil = Image.open(path)
            writer.add_image("val/overlay",
                             np.array(pil).transpose(2, 0, 1) / 255.0,
                             epoch, dataformats="CHW")
