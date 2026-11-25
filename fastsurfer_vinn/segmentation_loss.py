"""Anatomical consistency loss using a frozen 35-class FastSurferVINN2D."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .models.vinn import FastSurferVINN2D


# ── Dice + CE loss (copied from fastsurfer3d/utils/loss.py) ─────────

def dice_ce_loss(logits, target, ignore_index=0, dice_weight=0.5,
                 ce_weight=0.5, class_weight=None, smooth=1e-6, voxel_weight=None):
    """Combined Dice + Cross-Entropy loss.

    Args:
        logits: (B, C, H, W) predicted logits.
        target: (B, H, W) integer labels.
        ignore_index: Class index to ignore in loss (default 0 = background).
        dice_weight / ce_weight: Relative weighting.
        class_weight: Per-class weight for CE (1-D tensor, length C).
        smooth: Laplace smoothing for Dice.
        voxel_weight: Per-pixel weight for CE (B, H, W).

    Returns:
        (combined_loss, intersection_per_class, union_per_class)
    """
    n_classes = logits.shape[1]

    # CE
    ce = F.cross_entropy(logits, target, weight=class_weight,
                         ignore_index=ignore_index, reduction='none')
    if voxel_weight is not None:
        ce = ce * voxel_weight
    ce = ce.mean()

    # Dice (per-class, excluding ignore_index)
    target_one_hot = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    if ignore_index >= 0 and ignore_index < n_classes:
        target_one_hot[:, ignore_index] = 0.0
    pred_soft = F.softmax(logits, dim=1)
    inter = (pred_soft * target_one_hot).sum(dim=(0, 2, 3))
    union = (pred_soft + target_one_hot).sum(dim=(0, 2, 3))
    dice = (2.0 * inter + smooth) / (union + smooth)
    dice_loss = 1.0 - dice[1:].mean()  # exclude background (class 0)

    total = dice_weight * dice_loss + ce_weight * ce
    return total, inter.detach(), union.detach()


# ── VINNSegLoss ─────────────────────────────────────────────────────

class VINNSegLoss(nn.Module):
    """Frozen FastSurferVINN2D segmentation consistency loss.

    The default soft KL compares reference and prediction class distributions.
    Hard mode compares the prediction with reference argmax labels using Dice + CE.

    Args:
        ckpt_path: Path to a ``FastSurferVINN2D`` checkpoint (best.pth).
        device: Torch device.
        scale_factor: 1 mm divided by the in-plane voxel spacing in mm.
        mode: ``"soft"`` for KL divergence or ``"hard"`` for Dice + CE.
    """

    def __init__(self, ckpt_path: str, device: torch.device | None = None,
                 scale_factor: float = 2.0, *, mode: str = "soft"):
        super().__init__()
        if mode not in ("soft", "hard"):
            raise ValueError(f"Expected mode 'soft' or 'hard', got {mode!r}.")
        self.mode = mode
        self._device = device
        self._scale_factor = scale_factor

        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = state.get("model", state)
        if any(k.startswith("module.") for k in sd):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

        self.vinn = FastSurferVINN2D(in_ch=1, out_ch=35)
        self.vinn.load_state_dict(sd)
        for p in self.vinn.parameters():
            p.requires_grad = False
        self.vinn.eval()
        if self._device is not None:
            self.vinn.to(self._device)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"VINNSegLoss loaded: {sum(p.numel() for p in self.vinn.parameters()):,} params "
                  f"from {ckpt_path}")

    def train(self, mode: bool = True) -> "VINNSegLoss":
        super().train(mode)
        self.vinn.eval()
        return self

    def to(self, device):
        self._device = device
        self.vinn.to(device)
        return super().to(device)

    def forward(self, pred: torch.Tensor, ref: torch.Tensor,
                scale_factor: float | None = None) -> torch.Tensor:
        """Compute VINN segmentation consistency loss.

        Assumes inputs are already in N(0,1) for brain voxels (as produced
        by the ``norm_nonzero`` data pipeline).  No extra per-image
        normalisation is applied.

        Args:
            pred: Generated image (B, 1, H, W).
            ref: Aligned reference image (B, 1, H, W) at the same spacing.
            scale_factor: Override the constructor's default scale for this pair.

        Returns:
            Scalar consistency loss. Soft KL includes all classes and averages
            over batch and pixels. Hard Dice + CE uses foreground targets.
        """
        if pred.ndim != 4 or pred.shape[1] != 1 or pred.shape != ref.shape:
            raise ValueError("Prediction and reference must have the same (B, 1, H, W) shape.")
        if pred.device != ref.device:
            raise ValueError("Prediction and reference must be on the same device.")

        if pred.device != self._device:
            self.vinn.to(pred.device)
            self._device = pred.device

        sf = scale_factor or self._scale_factor

        with torch.no_grad():
            seg_ref = self.vinn(ref, scale_factor=sf)
            target = (F.softmax(seg_ref.float(), dim=1) if self.mode == "soft"
                      else seg_ref.argmax(dim=1))
        seg_pred = self.vinn(pred, scale_factor=sf)
        if self.mode == "soft":
            log_pred = F.log_softmax(seg_pred.float(), dim=1)
            return F.kl_div(log_pred, target, reduction="none").sum(dim=1).mean()
        loss, _, _ = dice_ce_loss(seg_pred, target, ignore_index=0)
        return loss

    def predict(self, x: torch.Tensor,
                scale_factor: float | None = None) -> torch.Tensor:
        """Run VINN inference for visualisation — returns (B, H, W) argmax.

        Assumes input is already in N(0,1) for brain voxels (as produced
        by the ``norm_nonzero`` data pipeline).  No extra per-image
        normalisation is applied.

        Args:
            x: Image tensor (B, 1, H, W) in any float dtype (bf16/fp32).
            scale_factor: VINN scale (default ``self._scale_factor``).

        Returns:
            Integer label map ``(B, H, W)`` (0–34).
        """
        if x.device != self._device:
            self.vinn.to(x.device)
            self._device = x.device
        sf = scale_factor or self._scale_factor
        with torch.no_grad():
            return self.vinn(x.float(), scale_factor=sf).argmax(dim=1)
