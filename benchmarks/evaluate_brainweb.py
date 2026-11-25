"""Reproduce the BrainWeb-20 whole-volume tissue Dice benchmark."""

import argparse
import csv
import gzip
from pathlib import Path
import shutil
from statistics import mean, stdev
import subprocess
import sys
from urllib.parse import urlencode
from urllib.request import urlopen

import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from nibabel.processing import resample_from_to
import numpy as np

from fastsurfer_vinn.inference import DEFAULT_CHECKPOINT, load_inference_model, predict_keepgeom
from fastsurfer_vinn.utils.labels import GM_ASEG_IDS, WM_ASEG_IDS


PHANTOMS = (4, 5, 6, 18, 20, 38, *range(41, 55))
BRAINWEB = "https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1"


def download(phantom, kind, cache_dir):
    alias = f"subject{phantom:02d}_{kind}"
    target = cache_dir / f"{alias}.mnc"
    if target.is_file():
        return target
    query = urlencode({"do_download_alias": alias, "format_value": "minc",
                       "zip_value": "gnuzip", "download_for_real": "[Start download!]"})
    partial = target.with_suffix(".partial")
    try:
        with urlopen(f"{BRAINWEB}?{query}", timeout=60) as response:
            with gzip.GzipFile(fileobj=response) as archive, partial.open("wb") as output:
                shutil.copyfileobj(archive, output)
        with partial.open("rb") as stream:
            if not stream.read(4).startswith((b"CDF", b"\x89HDF")):
                raise ValueError(f"BrainWeb returned a non-MINC response for {alias}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def tissue_dice(predicted, reference, aseg_ids, tissue_id):
    pred_mask = np.isin(predicted, aseg_ids)
    ref_mask = reference == tissue_id
    pred_count = int(np.count_nonzero(pred_mask))
    ref_count = int(np.count_nonzero(ref_mask))
    if not ref_count:
        raise ValueError(f"BrainWeb reference is missing tissue {tissue_id}")
    intersection = int(np.count_nonzero(pred_mask & ref_mask))
    return 2 * intersection / (pred_count + ref_count), pred_count, ref_count, intersection


def score_prediction(phantom, prediction, reference, allow_regrid=False):
    if prediction.shape != reference.shape or not np.allclose(prediction.affine, reference.affine, atol=1e-5):
        if not allow_regrid:
            raise ValueError(f"Prediction and reference grids differ for phantom {phantom:02d}")
        prediction = resample_from_to(prediction, reference, order=0)

    labels, truth = np.asarray(prediction.dataobj), np.asarray(reference.dataobj)
    wm = tissue_dice(labels, truth, WM_ASEG_IDS, 3)
    gm = tissue_dice(labels, truth, GM_ASEG_IDS, 2)
    print(f"Phantom {phantom:02d}: WM {wm[0]:.6f}, GM {gm[0]:.6f}", flush=True)
    return (f"{phantom:02d}", *wm, *gm)


def evaluate(phantom, model, args, synthstrip):
    folder = args.output_dir / f"subject-{phantom:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder
    if args.cache_dir is not None:
        cache = args.cache_dir / folder.name
        cache.mkdir(parents=True, exist_ok=True)
    t1_minc = nib.as_closest_canonical(nib.load(download(phantom, "t1w", cache)))
    reference_minc = nib.as_closest_canonical(nib.load(download(phantom, "crisp", cache)))
    values = t1_minc.get_fdata(dtype=np.float32)
    if values.shape != (256, 256, 181) or not np.isfinite(values).all():
        raise ValueError(f"Unexpected BrainWeb T1w data for phantom {phantom:02d}: {values.shape}")
    t1 = nib.Nifti1Image(values, t1_minc.affine)
    tissue = np.rint(reference_minc.get_fdata(dtype=np.float32)).astype(np.uint8)
    reference = resample_from_to(nib.Nifti1Image(tissue, reference_minc.affine), t1, order=0)

    if args.predictions_dir is not None:
        prediction = nib.load(args.predictions_dir / f"subject-{phantom:02d}.nii.gz")
        return score_prediction(phantom, prediction, reference, allow_regrid=True)

    lia = ornt_transform(io_orientation(t1.affine), axcodes2ornt(("L", "I", "A")))
    strip_input = folder / "t1w-lia.nii.gz"
    mask_path = folder / "brain-mask-lia.nii.gz"
    nib.save(t1.as_reoriented(lia), strip_input)
    subprocess.run([*synthstrip, "-i", str(strip_input), "-m", str(mask_path)], check=True)
    mask = nib.as_closest_canonical(nib.load(mask_path))
    if mask.shape != t1.shape or not np.allclose(mask.affine, t1.affine, atol=1e-5):
        raise ValueError(f"SynthStrip mask grid differs from the T1w grid for phantom {phantom:02d}")
    brain_path = folder / "t1w-brain.nii.gz"
    brain = np.where(np.asarray(mask.dataobj) != 0, values, 0)
    nib.save(nib.Nifti1Image(brain, t1.affine), brain_path)

    pred_path = folder / "vinn.nii.gz"
    predict_keepgeom(model, str(brain_path), str(pred_path), batch_size=args.batch_size,
                     device=args.device, clip=True)
    return score_prediction(phantom, nib.load(pred_path), reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", nargs="+", type=int, choices=PHANTOMS, default=PHANTOMS)
    parser.add_argument("--output-dir", type=Path, help="Directory for intermediate files and dice-per-case.csv")
    parser.add_argument("--cache-dir", type=Path, help="Directory with downloaded MINC files in subject-NN folders")
    parser.add_argument("--predictions-dir", type=Path,
                        help="Score existing aseg NIfTIs named subject-NN.nii.gz")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", help="VINN device, auto-detected by default")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--synthstrip", default="mri_synthstrip", help="SynthStrip command or standalone Python script")
    parser.add_argument("--synthstrip-model", type=Path, help="Optional path to the SynthStrip v1 weights")
    parser.add_argument("--strip-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir is None:
        subdir = "brainweb-scores" if args.predictions_dir is not None else "brainweb-eval"
        args.output_dir = Path("outputs") / subdir
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    synthstrip = []
    model = None
    if args.predictions_dir is not None:
        for phantom in sorted(set(args.subject)):
            prediction = args.predictions_dir / f"subject-{phantom:02d}.nii.gz"
            if not prediction.is_file():
                parser.error(f"Missing prediction: {prediction}")
    else:
        if shutil.which(args.synthstrip) is None and not Path(args.synthstrip).is_file():
            parser.error("Set --synthstrip to an installed mri_synthstrip command or its Python script.")
        synthstrip = [args.synthstrip]
        if Path(args.synthstrip).suffix == ".py":
            synthstrip.insert(0, sys.executable)
        if args.synthstrip_model:
            synthstrip += ["--model", str(args.synthstrip_model)]
        if args.strip_device == "cuda":
            synthstrip.append("-g")
        model = load_inference_model(args.checkpoint)
    rows = [evaluate(phantom, model, args, synthstrip) for phantom in sorted(set(args.subject))]
    columns = ("dice", "prediction_voxels", "reference_voxels", "intersection")
    output = args.output_dir / "dice-per-case.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["phantom", *(f"{tissue}_{name}" for tissue in ("wm", "gm") for name in columns)])
        writer.writerows(rows)

    for tissue, column in (("WM", 1), ("GM", 5)):
        scores = [row[column] for row in rows]
        spread = f" +/- {stdev(scores):.4f}" if len(scores) > 1 else ""
        print(f"{tissue}: {mean(scores):.4f}{spread} ({len(scores)} phantoms)")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
