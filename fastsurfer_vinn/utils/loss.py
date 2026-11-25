import torch
import torch.nn.functional as F
import numpy as np


def _spatial_dims(logits):
    """Return (spatial_axes_tuple, has_depth) given logits shape."""
    ndim = logits.ndim
    if ndim == 5:  # 3D: (B, C, D, H, W)
        return (2, 3, 4), True
    else:           # 2D: (B, C, H, W)
        return (2, 3), False


def dice_loss(logits, target, ignore_index=-1, smooth=1e-6):
    n_cls = logits.shape[1]
    dtype = logits.dtype
    sp_dims, has_d = _spatial_dims(logits)

    if has_d:
        B, _, D, H, W = logits.shape
        target_onehot = torch.zeros(B, n_cls, D, H, W, dtype=dtype, device=logits.device)
    else:
        B, _, H, W = logits.shape
        target_onehot = torch.zeros(B, n_cls, H, W, dtype=dtype, device=logits.device)

    target_onehot.scatter_(1, target.unsqueeze(1).long(), 1.0)

    if ignore_index >= 0:
        mask = target != ignore_index
        target_onehot = target_onehot * mask.unsqueeze(1).to(dtype)

    prob = F.softmax(logits, dim=1)

    if ignore_index >= 0:
        prob = prob * mask.unsqueeze(1).to(dtype)
    intersection = (prob * target_onehot).sum(dim=sp_dims)   # (B, C)
    union = prob.sum(dim=sp_dims) + target_onehot.sum(dim=sp_dims)

    dice = (2. * intersection + smooth) / (union + smooth)

    if ignore_index >= 0:
        present = target_onehot.sum(dim=(0, *sp_dims)) > 0
        if present.any():
            dice = dice[:, present]

    return 1 - dice.mean(), intersection.sum(dim=0), union.sum(dim=0)


def ce_loss(logits, target, class_weight=None, voxel_weight=None):
    ce = F.cross_entropy(logits, target, weight=class_weight,
                          ignore_index=-1, reduction='none')
    if voxel_weight is not None:
        ce = ce * voxel_weight
    return ce.mean()


def dice_ce_loss(logits, target, ignore_index=0, dice_weight=0.5,
                 ce_weight=0.5, class_weight=None, smooth=1e-6, voxel_weight=None):
    d_loss, inter, union = dice_loss(logits, target, ignore_index, smooth)
    c_loss = ce_loss(logits, target, class_weight, voxel_weight=voxel_weight)
    return dice_weight * d_loss + ce_weight * c_loss, inter, union


def deep_supervision_loss(outputs, target, aux_weights=(0.5, 0.25), spatial_dims=3, **loss_kwargs):
    ignore_index = loss_kwargs.pop("ignore_index", 0)
    class_weight = loss_kwargs.pop("class_weight", None)
    dice_weight = loss_kwargs.pop("dice_weight", 0.5)
    ce_weight = loss_kwargs.pop("ce_weight", 0.5)
    smooth = loss_kwargs.pop("smooth", 1e-6)
    voxel_weight = loss_kwargs.pop("voxel_weight", None)

    def _head_loss(logits):
        return dice_ce_loss(logits, target, ignore_index,
                            dice_weight, ce_weight, class_weight, smooth,
                            voxel_weight=voxel_weight)

    # 3D deep sup: (B, 1+N, C, D, H, W)
    # 2D deep sup: (B, 1+N, C, H, W)
    # no deep sup: (B, C, ...) where ndim == spatial_dims + 2
    has_aux = outputs.dim() == spatial_dims + 3
    if has_aux:
        main_logits = outputs[:, 0]
        aux_logits = outputs[:, 1:]
        total_loss, inter, union = _head_loss(main_logits)
        for i, aux in enumerate(aux_logits.unbind(1)):
            w = aux_weights[i] if i < len(aux_weights) else aux_weights[-1]
            aux_loss, _, _ = _head_loss(aux)
            total_loss = total_loss + w * aux_loss
        return total_loss, main_logits, inter, union
    else:
        total_loss, inter, union = _head_loss(outputs)
        return total_loss, outputs, inter, union


def accumulate_dice_stats(logits, target, ignore_index=0):
    n_cls = logits.shape[1]
    dtype = logits.dtype
    sp_dims, has_d = _spatial_dims(logits)

    if has_d:
        B, _, D, H, W = logits.shape
        tgt = torch.zeros(B, n_cls, D, H, W, dtype=dtype, device=logits.device)
    else:
        B, _, H, W = logits.shape
        tgt = torch.zeros(B, n_cls, H, W, dtype=dtype, device=logits.device)

    if target.max() >= n_cls or target.min() < 0:
        import warnings
        max_n_cls = n_cls - 1
        warnings.warn(f"target out of range [0, {max_n_cls}]: "
                      f"min={target.min().item()}, max={target.max().item()}, will clamp")
        target = target.clamp(0, max_n_cls)
    tgt.scatter_(1, target.unsqueeze(1).long(), 1.0)
    prob = F.softmax(logits, dim=1)
    if ignore_index >= 0:
        mask = target != ignore_index
        prob = prob * mask.unsqueeze(1).to(dtype)
        tgt = tgt * mask.unsqueeze(1).to(dtype)
    intersection = (prob * tgt).sum(dim=(0, *sp_dims))
    union = (prob + tgt).sum(dim=(0, *sp_dims))
    return intersection, union


def dice_from_stats(inter, unn, ignore_index=0):
    """Per-class Dice + foreground macro from accumulated (inter, unn)."""
    dice_per_cls = (2. * inter / (unn + 1e-6)).cpu().numpy()
    if ignore_index >= 0:
        fore = dice_per_cls[ignore_index + 1:]
        present = unn[ignore_index + 1:].cpu().numpy() > 1
        total = float(fore[present].mean()) if present.any() else 0.0
    else:
        valid = dice_per_cls[np.isfinite(dice_per_cls)]
        total = float(valid.mean()) if len(valid) > 0 else 0.0
    return dice_per_cls, total


def compute_class_weights(labels, num_classes):
    """Per-class CE weight = 1/sqrt(freq + 1)."""
    device = labels[0].device
    counts = torch.zeros(num_classes, dtype=torch.float64, device=device)
    for lab in labels:
        flat = lab.flatten()
        if flat.device != device:
            flat = flat.to(device)
        counts += torch.bincount(flat, minlength=num_classes)[:num_classes].float()
    return compute_class_weights_from_counts(counts, num_classes)


def compute_class_weights_from_counts(counts, num_classes):
    """Per-class CE weight from accumulated counts."""
    w = 1.0 / torch.sqrt(counts + 1.0)
    w = (w / w.sum() * num_classes).float()
    return w


# ── FastSurfer-style voxel weight for boundary emphasis ────────────

def compute_voxel_weight(label, edge_w=5.0, max_w=5.0):
    """Per-voxel weight map: boundaries get higher weight.

    Args:
        label: (B, H, W) or (H, W) — class indices.
        edge_w: Multiplier for boundary voxels (default 5.0).
        max_w: Cap for weight values (default 5.0).

    Returns:
        weight: (B, H, W) or (H, W) — float32.
    """
    if label.ndim == 2:
        label = label.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, H, W = label.shape
    label_f = label.float().unsqueeze(1)  # (B, 1, H, W)

    # Sobel kernels for boundary detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
                           device=label.device).view(1, 1, 3, 3) / 8.0
    sobel_y = sobel_x.transpose(2, 3)

    grad_x = F.conv2d(label_f, sobel_x, padding=1)
    grad_y = F.conv2d(label_f, sobel_y, padding=1)
    edge = (grad_x.abs() + grad_y.abs()) > 0.01  # boundary mask

    weight = torch.ones_like(label_f)
    weight[edge] = edge_w
    weight = weight.clamp(1.0, max_w).squeeze(1)  # (B, H, W)

    if squeeze:
        weight = weight.squeeze(0)
    return weight


# ── FastSurfer-style CombinedLoss (Dice + CE + voxel_weight) ───────

class CombinedLoss(torch.nn.Module):
    """FastSurferVINN-style Combined Loss (Dice + CE with voxel weight).

    Args:
        weight_dice: Dice loss weight (default 1.0).
        weight_ce: CE loss weight (default 1.0).
    """
    def __init__(self, weight_dice=1.0, weight_ce=1.0):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, logits, target, voxel_weight=None, ignore_index=-1):
        """Compute CombinedLoss.

        Args:
            logits: (B, C, H, W) or (B, C, D, H, W).
            target: (B, H, W) or (B, D, H, W).
            voxel_weight: Optional (B, H, W) per-voxel weight map.
            ignore_index: Class index to ignore.

        Returns:
            (total_loss, dice_loss_val, ce_loss_val).
        """
        prob = F.softmax(logits, dim=1)
        sp_dims, _ = _spatial_dims(logits)

        # Dice loss
        n_cls = logits.shape[1]
        target_onehot = torch.zeros_like(prob)
        target_onehot.scatter_(1, target.unsqueeze(1).long(), 1.0)
        if ignore_index >= 0:
            mask = target != ignore_index
            target_onehot = target_onehot * mask.unsqueeze(1).to(prob.dtype)
            prob_m = prob * mask.unsqueeze(1).to(prob.dtype)
        else:
            prob_m = prob
        inter = (prob_m * target_onehot).sum(dim=sp_dims)
        union = prob_m.sum(dim=sp_dims) + target_onehot.sum(dim=sp_dims)
        dice = (2. * inter + 1e-6) / (union + 1e-6)
        if ignore_index >= 0:
            present = target_onehot.sum(dim=(0, *sp_dims)) > 0
            if present.any():
                dice = dice[:, present]
        dice_val = dice.mean()

        # CE loss (with optional per-pixel voxel weight, aligned with original)
        ce = F.cross_entropy(logits, target, ignore_index=-1, reduction='none')
        if voxel_weight is not None:
            ce = ce * voxel_weight
        ce_val = ce.mean()

        total = self.weight_dice * (1 - dice_val) + self.weight_ce * ce_val
        return total, 1 - dice_val, ce_val
