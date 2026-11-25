#!/usr/bin/env python3
"""FastSurferVINN2D inference — keep original geometry.

The CLI always normalizes nonzero voxels within each volume:

  **``--clip``**: Apply ClipIntensityPercentilesd(0.5,99.5) before z-score (for raw 0–1000+).

  **``--clean-bg``**: Apply robust background cleaning (Otsu threshold + morphology + hole fill)
  before any other preprocessing.

  Both always apply ``NormalizeIntensityd(nonzero=True)``.

VINN's internal ``Zoom2d`` handles resolution adaptation.
Scale factor = 1.0 / in_plane_spacing for an isotropic in-plane grid.

Usage:
    python -m fastsurfer_vinn.inference \
        --checkpoint checkpoints/best.pth \
        --input /data/T1W/0.1T/subject.nii.gz \
        --output seg.nii.gz
"""

import argparse
import math
from pathlib import Path
import shutil
import importlib.util

import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose, LoadImaged, Orientationd,
    ClipIntensityPercentilesd,
    NormalizeIntensityd, EnsureChannelFirstd, EnsureTyped,
)
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation

from fastsurfer_vinn.models.vinn import FastSurferVINN2D
from fastsurfer_vinn.utils.labels import _C2L_LUT

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "best.pth"

# ── NPU detection ──────────────────────────────────────────────────────────────

if shutil.which("npu-smi") and importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False


# ── Helpers ────────────────────────────────────────────────────────────────────


def _clean_background(data: np.ndarray) -> np.ndarray:
    """Zero out background via Otsu threshold + morphology + hole filling.

    Pipeline: Otsu threshold → binary closing → hole fill → apply mask.
    Works on 3D volumes (H, W, D).
    """
    from scipy import ndimage as ndi

    data = data.copy()
    flat = data[data > 0]
    if len(flat) < 100:
        return data

    # 1. Otsu threshold on non-zero voxels
    lo, hi = float(flat.min()), float(flat.max())
    if hi - lo < 1e-6:
        return data
    hist, bins = np.histogram(flat, bins=64, range=(lo, hi))
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    total = float(hist.sum())
    sum_total = float((hist * bin_centers).sum())

    weight_b = 0.0
    sum_b = 0.0
    best_thresh = bins[0]
    best_var = 0.0

    for i in range(len(hist)):
        w = float(hist[i])
        if w == 0:
            continue
        weight_b += w
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += w * bin_centers[i]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = bins[i]

    # 2. Binary mask
    binary = data > best_thresh

    # 3. Morphological closing — merge nearby foreground regions
    structure = ndi.generate_binary_structure(3, 2)  # 3D connectivity=2
    binary = ndi.binary_closing(binary, structure=structure, iterations=2)

    # 4. Fill holes
    binary = ndi.binary_fill_holes(binary)

    # 5. Apply mask
    data[~binary] = 0.0
    return data


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_ras_zooms(img: nib.Nifti1Image) -> list[float]:
    """Get voxel spacings for each axis after RAS orientation."""
    pixdim = img.header.get_zooms()
    orig_ornt = io_orientation(img.affine)
    ras_ornt = axcodes2ornt(("R", "A", "S"))
    tfm = ornt_transform(orig_ornt, ras_ornt)
    return [pixdim[int(axis)] for axis in np.argsort(tfm[:, 0])]


def _in_plane_voxel_size(img: nib.Nifti1Image, override: float | None) -> float:
    if override is not None:
        if not math.isfinite(override) or override <= 0:
            raise ValueError("In-plane voxel spacing must be finite and positive.")
        return override
    x, y, _ = _get_ras_zooms(img)
    if not all(math.isfinite(s) and s > 0 for s in (x, y)):
        raise ValueError("In-plane voxel spacing must be finite and positive.")
    if not math.isclose(x, y, rel_tol=0.002):
        raise ValueError(
            f"Axial in-plane spacings differ ({x:.4f} and {y:.4f} mm). "
            "Resample to equal in-plane spacing before inference, or supply "
            "--voxel-size to explicitly use one shared scale."
        )
    return x


def load_inference_model(checkpoint: str | Path = DEFAULT_CHECKPOINT) -> FastSurferVINN2D:
    """Load the default inference weights, preferring EMA where available."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = {key.removeprefix("module."): value for key, value in state["model"].items()}
    weights.update({key.removeprefix("module."): value for key, value in state.get("ema", {}).items()})
    model = FastSurferVINN2D()
    model.load_state_dict(weights)
    return model.eval()


# ── Main inference ──────────────────────────────────────────────────────────────


def predict_keepgeom(
    model: torch.nn.Module,
    input_path: str,
    output_path: str,
    voxel_size: float | None = None,
    batch_size: int = 32,
    device: str | None = None,
    clip: bool = False,
    clean_bg: bool = False,
):
    """Run VINN inference preserving original geometry.

    All branches apply z-score normalization (``NormalizeIntensityd(nonzero=True)``).

      - **``clip=True``**: also apply ``ClipIntensityPercentilesd(0.5,99.5)``
        before normalization (for raw 0–1000+ inputs).
      - **``clean_bg=True``**: apply robust background cleaning (Otsu threshold +
        morphology + hole fill) before any other step.

    Scale factor = 1.0 / in_plane_spacing.

    Args:
        model: FastSurferVINN2D in eval mode.
        input_path: Path to input NIfTI.
        output_path: Path to save output NIfTI.
        voxel_size: Override in-plane voxel spacing (mm). If None, read from header.
        batch_size: Number of slices to process simultaneously.
        device: "cuda", "npu", or "cpu".
        clip: Apply percentile clipping before z-score (for raw 0–1000+ data).
        clean_bg: Apply robust background cleaning before any preprocessing.
    """
    if device is None:
        device = _auto_device()
    print(f"Device: {device}")

    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    img = nib.load(input_path)
    if len(img.shape) != 3:
        raise ValueError(f"Expected a 3D input volume, got shape {img.shape}.")
    vs = _in_plane_voxel_size(img, voxel_size)
    orig_affine = img.affine.copy()
    orig_header = img.header.copy()

    # ── 1. Load ──
    loader = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
    ])
    item = loader({"image": input_path})
    arr = item["image"].float().numpy()  # (1, H, W, D)

    # ── 2. Optional background cleaning ──
    if clean_bg:
        arr[0] = _clean_background(arr[0])
        item["image"].copy_(torch.from_numpy(arr))

    # ── 3. Core preprocessing ──
    if clip:
        steps = [
            Orientationd(keys=["image"], axcodes="RAS"),
            ClipIntensityPercentilesd(keys=["image"], lower=0.5, upper=99.5),
            NormalizeIntensityd(keys=["image"], nonzero=True),
            EnsureTyped(keys=["image"]),
        ]
    else:
        steps = [
            Orientationd(keys=["image"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image"], nonzero=True),
            EnsureTyped(keys=["image"]),
        ]

    out = Compose(steps)(item)
    data_ras = out["image"].squeeze(0).numpy()  # (H, W, D)

    # ── 3. Scale factor — in-plane spacing ──
    sf = 1.0 / vs

    # ── 3. Slice-by-slice VINN forward ──
    D = data_ras.shape[2]
    pred_ras = np.zeros((*data_ras.shape[:2], D), dtype=np.int16)

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for z_start in range(0, D, batch_size):
            z_end = min(z_start + batch_size, D)
            slices = []
            for z in range(z_start, z_end):
                sl = data_ras[:, :, z]
                slices.append(sl[np.newaxis, np.newaxis])

            batch = torch.from_numpy(np.concatenate(slices, axis=0)).to(device)
            sf_t = torch.tensor([sf] * batch.shape[0], device=device)
            logits = model(batch, scale_factor=sf_t)
            labels = logits.argmax(dim=1).cpu().numpy()

            for i, z in enumerate(range(z_start, z_end)):
                pred_ras[:, :, z] = labels[i]

    # ── 4. Reorient back to original orientation ──
    orig_ornt = io_orientation(orig_affine)
    tgt_ornt = axcodes2ornt(("R", "A", "S"))
    inv_ornt = ornt_transform(tgt_ornt, orig_ornt)
    pred_orig = apply_orientation(pred_ras.astype(np.int16), inv_ornt).copy()

    # ── 5. Save ──
    out = _C2L_LUT[torch.from_numpy(pred_orig).long()].numpy().astype(np.int16)
    segmentation = nib.Nifti1Image(out, orig_affine, orig_header)
    segmentation.set_data_dtype(np.int16)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(segmentation, output_path)
    print(f"Saved: {output_path}  (shape={out.shape}, sf={sf:.2f}, spacing={vs:.4f}mm)")


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="FastSurferVINN2D Keepgeom inference")
    parser.add_argument("--checkpoint", type=Path,
                        default=DEFAULT_CHECKPOINT,
                        help="Path to checkpoint (default: bundled best.pth)")
    parser.add_argument("--input", required=True, help="Input NIfTI path")
    parser.add_argument("--output", required=True, help="Output NIfTI path")
    parser.add_argument("--device", default=None,
                        help="Device (cuda / npu / cpu). Auto-detect if not set.")
    parser.add_argument("--batch-size", type=int, default=32, help="Slice batch size")
    parser.add_argument("--voxel-size", type=float, default=None,
                        help="In-plane voxel spacing (mm). Auto-detect from header if not set.")
    parser.add_argument("--clip", action="store_true",
                        help="Apply ClipIntensityPercentilesd(0.5,99.5) before z-score (for raw 0–1000+ data)")
    parser.add_argument("--clean-bg", action="store_true",
                        help="Apply background cleaning (Otsu + morphology + hole fill) before preprocessing")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.voxel_size is not None and not (0 < args.voxel_size < float("inf")):
        parser.error("--voxel-size must be finite and positive")

    model = load_inference_model(args.checkpoint)
    print(f"Model loaded, params={sum(p.numel() for p in model.parameters()):,}")

    try:
        predict_keepgeom(model, args.input, args.output, args.voxel_size,
                         args.batch_size, args.device, args.clip, args.clean_bg)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
