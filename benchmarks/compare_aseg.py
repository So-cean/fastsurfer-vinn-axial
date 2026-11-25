"""Compare reduced aseg segmentations on the same MRI grid."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from fastsurfer_vinn.utils.labels import GM_ASEG_IDS, WM_ASEG_IDS


GROUPS = {"WM": WM_ASEG_IDS, "GM": GM_ASEG_IDS, "Ventricles": (4, 5, 14, 15, 43, 44)}
COLORS = ("#25c7e4", "#ffac58", "#b6a0ee")


def labels_on_grid(path, target):
    img = nib.as_closest_canonical(nib.load(path))
    if img.shape != target.shape or not np.allclose(img.affine, target.affine, atol=1e-5):
        img = resample_from_to(img, target, order=0)
    return np.asarray(img.dataobj).astype(np.int16)


def summary(masks, brain, voxel_ml):
    brain_voxels = np.count_nonzero(brain)
    return {
        name: {
            "foreground_ml": float(np.count_nonzero(values > 0) * voxel_ml),
            "mask_coverage_pct": 100 * np.count_nonzero((values > 0) & brain) / brain_voxels,
            "foreground_outside_mask_ml": float(np.count_nonzero((values > 0) & ~brain) * voxel_ml),
            "tissue_ml": {key: float(np.count_nonzero(np.isin(values, group)) * voxel_ml)
                          for key, group in GROUPS.items()},
        }
        for name, values in masks.items()
    }


def dice_agreement(left, right, group):
    a, b = np.isin(left, group), np.isin(right, group)
    size = np.count_nonzero(a) + np.count_nonzero(b)
    return 2 * np.count_nonzero(a & b) / size if size else None


def save_figure(path, image, brain, segmentations, title):
    spacing = nib.affines.voxel_sizes(image.affine)
    data = image.get_fdata(dtype=np.float32)
    lo, hi = np.percentile(data[brain], (1, 99))
    bounds = [np.flatnonzero(brain.any(axis=tuple(i for i in range(3) if i != axis)))
              for axis in range(3)]
    crop = [slice(max(0, indices[0] - 9), min(n, indices[-1] + 10))
            for indices, n in zip(bounds, image.shape)]
    centers = [int((indices[0] + indices[-1]) // 2) for indices in bounds]
    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(3, 1 + len(segmentations), figsize=(3.2 * (1 + len(segmentations)), 9.4),
                             facecolor="white", squeeze=False)
    for row, (axis, plane) in enumerate(((2, "Axial"), (1, "Coronal"), (0, "Sagittal"))):
        index = crop.copy()
        index[axis] = centers[axis]
        index = tuple(index)
        x, y = [i for i in range(3) if i != axis]
        aspect = spacing[y] / spacing[x]
        for ax in axes[row]:
            ax.imshow(data[index].T, cmap="gray", origin="lower", vmin=lo, vmax=hi, aspect=aspect)
            ax.axis("off")
        axes[row, 0].text(0.03, 0.04, plane, color="white", transform=axes[row, 0].transAxes)
        for col, values in enumerate(segmentations.values(), start=1):
            overlay = np.zeros(values[index].shape, dtype=np.uint8)
            for code, group in enumerate(GROUPS.values(), start=1):
                overlay[np.isin(values[index], group)] = code
            axes[row, col].imshow(np.ma.masked_equal(overlay.T, 0), cmap=cmap, alpha=0.5,
                                  origin="lower", vmin=1, vmax=len(GROUPS), aspect=aspect,
                                  interpolation="nearest")
    for col, text in enumerate(("Input MRI", *segmentations)):
        axes[0, col].set_title(text)
    fig.suptitle(title, fontsize=15)
    fig.legend(handles=[Patch(color=color, label=name) for name, color in zip(GROUPS, COLORS)],
               loc="lower center", ncols=len(GROUPS), frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--brain-mask", type=Path, required=True)
    parser.add_argument("--vinn", type=Path, required=True)
    parser.add_argument("--fastsurfer", type=Path, required=True, help="FastSurfer reduced aseg on whole-head input")
    parser.add_argument("--fastsurfer-brain", type=Path, help="FastSurfer reduced aseg on the shared brain input")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Axial VINN and FastSurfer")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image = nib.as_closest_canonical(nib.load(args.image))
    brain = labels_on_grid(args.brain_mask, image) > 0
    if not brain.any():
        parser.error("Brain mask is empty")
    segmentations = {
        "Axial VINN": labels_on_grid(args.vinn, image),
        "FastSurfer (whole head)": labels_on_grid(args.fastsurfer, image),
    }
    if args.fastsurfer_brain:
        segmentations["FastSurfer (brain only)"] = labels_on_grid(args.fastsurfer_brain, image)

    voxel_ml = abs(np.linalg.det(image.affine[:3, :3])) / 1000
    stats = {
        "brain_mask_ml": float(np.count_nonzero(brain) * voxel_ml),
        "segmentations": summary(segmentations, brain, voxel_ml),
        "dice_agreement_with_axial_vinn": {
            name: {key: dice_agreement(segmentations["Axial VINN"], values, group)
                   for key, group in GROUPS.items()}
            for name, values in segmentations.items() if name != "Axial VINN"
        },
    }
    (args.output_dir / "aseg-comparison.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    save_figure(args.output_dir / "aseg-comparison.png", image, brain, segmentations, args.title)


if __name__ == "__main__":
    main()
