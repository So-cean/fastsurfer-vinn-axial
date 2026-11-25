"""Data discovery: find image + seg pairs on disk (synthseg2d-based).

Splits are pre-defined on disk::

    {data_root}/{name}/fastsurfer_seg/*_seg.nii.gz
    {data_root}/{name}/{modality}_split/{train,val}/*.nii.gz
"""

from pathlib import Path


def discover_train_items(
    data_root: str | Path,
    datasets_cfg: list[dict],
    split: str = "train",
) -> list[dict]:
    """Discover seg + image pairs for training, aligned to synthseg2d pattern."""
    items = []
    data_root = Path(data_root)

    for ds in datasets_cfg:
        name = ds["name"]
        modalities = ds.get("modalities", ["T1w"])
        seg_dir_name = ds.get("seg_dir", "fastsurfer_seg")
        seg_root = data_root / name / seg_dir_name
        if not seg_root.is_dir():
            print(f"  SKIP: {name} ({seg_root} not found)")
            continue

        counts = {m: 0 for m in modalities}

        for seg_path in sorted(seg_root.glob("*_seg.nii.gz")):
            seg_str = str(seg_path)

            for mod in modalities:
                if mod.upper() == "T1W":
                    image_name = seg_path.name.replace("_seg.nii.gz", ".nii.gz")
                    img_path = data_root / name / "T1w_split" / split / image_name

                elif mod.upper() == "T2W":
                    image_name = seg_path.name.replace("_T1w_seg.nii.gz", "_T2w.nii.gz")
                    img_path = data_root / name / "T2w_split" / split / image_name

                elif mod.upper() == "FLAIR":
                    flair_dir = data_root / name / f"FLAIR_split/{split}"
                    if not flair_dir.is_dir():
                        continue
                    bn = seg_path.name.replace("_seg.nii.gz", "")
                    prefix = bn.replace("_registered", "").rsplit("-", 1)[0]
                    img_path = None
                    for f in flair_dir.iterdir():
                        if f.name.endswith(".nii.gz") and f.name.startswith(prefix):
                            img_path = f
                            break

                else:
                    continue

                if img_path and img_path.is_file():
                    items.append({"image": str(img_path), "label": str(seg_path)})
                    counts[mod] += 1

        parts = [f"{m}={counts[m]}" for m in modalities]
        total_seg = max(counts.values()) if counts else 0
        print(f"  {name}: {', '.join(parts)}, total_seg={total_seg}")

    print(f"Total train items: {len(items)}")
    return items


def discover_val_items(
    data_root: str | Path,
    datasets_cfg: list[dict],
    split: str = "val",
) -> list[dict]:
    """Discover image + seg pairs for validation (synthseg2d pattern)."""
    items = []
    data_root = Path(data_root)

    for ds in datasets_cfg:
        name = ds["name"]
        modalities = ds.get("modalities", ["T1w"])
        seg_dir_name = ds.get("seg_dir", "fastsurfer_seg")
        seg_root = data_root / name / seg_dir_name
        if not seg_root.is_dir():
            continue

        all_seg = {}
        for f in seg_root.glob("*_seg.nii.gz"):
            stem = f.name.replace("_seg.nii.gz", "")
            all_seg[stem] = str(f)

        for mod in modalities:
            val_dir = data_root / name / f"{mod}_split" / split
            if not val_dir.is_dir():
                continue

            for fname in sorted(val_dir.glob("*.nii.gz")):
                if mod.upper() == "T1W":
                    seg_stem = fname.name.replace(".nii.gz", "")
                elif mod.upper() == "T2W":
                    seg_stem = fname.name.replace("_T2w.nii.gz", "_T1w")
                elif mod.upper() == "FLAIR":
                    prefix = fname.name.replace("_registered.nii.gz", "").rsplit("-", 1)[0]
                    seg_stem = next((s for s in all_seg if s.startswith(prefix)), None)
                    if seg_stem is None:
                        continue
                else:
                    continue

                seg_path = all_seg.get(seg_stem)
                if seg_path is not None:
                    items.append({"image": str(fname), "label": seg_path})

    print(f"Total val items: {len(items)}")
    return items
