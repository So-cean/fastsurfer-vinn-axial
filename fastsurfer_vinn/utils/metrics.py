"""Confusion-matrix based Dice for segmentation training.

All computation stays on NPU (async sums, no per-batch sync).
Conversion to CPU only at epoch end.
"""

import torch

CLASS_NAMES = ["BG", "WM", "Cortex", "Vent+CSF", "DeepGM", "Cereb+BS"]


def confusion_matrix(pred_logits, target, n_classes=6):
    """Compute per-class TP / PredCount / GTCount on the current device.

    No sync, no item() — all torch ops stay async on NPU/GPU.

    Args:
        pred_logits: (B, C, H, W) logits
        target:      (B, H, W) int labels

    Returns:
        tensor (n_classes*3,)  = [tp_0..tp_5, pc_0..pc_5, gc_0..gc_5]
    """
    pred_c = pred_logits.argmax(dim=1)
    conf = torch.zeros(n_classes * 3, device=pred_c.device)
    for c in range(n_classes):
        p = (pred_c == c)
        t = (target == c)
        conf[c] = (p & t).sum().float()
        conf[n_classes + c] = p.sum().float()
        conf[2 * n_classes + c] = t.sum().float()
    return conf


def dice_from_confusion(conf):
    """Convert confusion-matrix totals to per-class Dice.

    Args:
        conf: (n_classes*3,) tensor (on any device)

    Returns:
        tensor (n_classes,) — per-class Dice or 0.0
    """
    n = conf.shape[0] // 3
    tp, pc, gc = conf[:n], conf[n:2*n], conf[2*n:]
    denom = pc + gc
    dice = torch.where(denom > 0, 2 * tp / denom,
                       torch.zeros_like(denom))
    return dice


# Keep old API for any caller that needs it
def dice_per_class(pred, target, n_classes=6):
    conf = confusion_matrix(pred, target, n_classes)
    return dice_from_confusion(conf)
