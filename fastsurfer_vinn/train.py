#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train FastSurferVINN2D on MRI and FastSurfer labels.

Single GPU:
    python -m fastsurfer_vinn.train --config fastsurfer_vinn/configs/vinn.yaml

Multi-GPU (torchrun):
    torchrun --nproc_per_node=8 -m fastsurfer_vinn.train --config fastsurfer_vinn/configs/vinn.yaml --dist
"""

import fastsurfer_vinn.utils.npu_setup  # must precede torch import (NPU config)

import argparse
import os
import sys

import torch
import yaml


def _resolve_max(val):
    """If val is a list/tuple, return max; else return val."""
    if isinstance(val, (list, tuple)):
        return max(val)
    return val


def _cache_loader(loader, name, use_dist):
    """Trigger PersistentDataset caching — direct iteration, no DataLoader workers."""
    ds = loader.dataset.cached
    total = len(ds)

    rank = 0
    world = 1
    if use_dist:
        import torch.distributed as dist
        rank = dist.get_rank()
        world = dist.get_world_size()
    else:
        rank, world = 0, 1

    for i in range(rank, total, world):
        _ = ds[i]
        if rank == 0 and i % 500 == 0:
            print(f"  [{name}] {i}/{total}", flush=True)

    if use_dist:
        dist.barrier()
    if rank == 0:
        print(f"  [{name}] cache done ({total} items)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--build-cache", action="store_true",
                        help="Build persistent cache before training (multi-rank safe)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    use_dist = args.dist
    local_rank = 0
    if use_dist:
        from fastsurfer_vinn.utils.ddp import init_dist
        local_rank = init_dist()

    from fastsurfer_vinn.data.discovery import discover_train_items, discover_val_items
    from fastsurfer_vinn.data.dataset import build_dataloaders

    data_root = cfg.get("data_root", "")
    cache_dir = os.path.join(data_root, cfg.get("data", {}).get("cache_dir", ".monai_cache/fastsurfer3d"))

    train_cfg = cfg.get("train_datasets", cfg.get("datasets", []))
    if not train_cfg:
        print("No train_datasets in config.")
        sys.exit(1)
    train_items = discover_train_items(data_root, train_cfg, split="train")

    val_cfg = cfg.get("val_datasets", [])
    val_items = discover_val_items(data_root, val_cfg, split="val") if val_cfg else []

    aug_cfg = cfg.get("data", {}).get("augment", {})
    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "")
    spatial_dims = model_cfg.get("spatial_dims", 2 if model_type == "fastsurfervinn" else 3)
    data_cfg = cfg.get("data", {})

    if spatial_dims == 2:
        from fastsurfer_vinn.data.dataset2d import build_dataloaders_2d
        train_loader, val_loader = build_dataloaders_2d(
            train_items=train_items,
            val_items=val_items if val_items else None,
            cache_dir=cache_dir,
            batch_size=cfg.get("train", {}).get("batch_size", 64),
            num_workers=cfg.get("train", {}).get("num_workers", 8),
            num_samples=data_cfg.get("num_samples", 8),
            spatial_size=tuple(data_cfg.get("spatial_size", (192, 192, 192))),
            use_dist=use_dist,
            scale_factor=aug_cfg.get("scale_factor", 0.20),
            shift_offset=aug_cfg.get("shift_offset", 0.10),
            noise_std=_resolve_max(aug_cfg.get("noise_std", 0.05)),
            smooth_sigma=tuple(aug_cfg.get("smooth_sigma", (0.3, 0.8))),
            gibbs_alpha=tuple(aug_cfg.get("gibbs_alpha", (0.0, 0.5))),
            bias_coeff=aug_cfg.get("bias_coeff", 0.30),
            gmm_prob_real=aug_cfg.get("gmm_prob_real", 0.8),
            gmm_mu_jitter=aug_cfg.get("gmm_mu_jitter", 0.05),
            gmm_sigma_jitter=aug_cfg.get("gmm_sigma_jitter", 0.1),
        )
        is_2d = True
    else:
        from fastsurfer_vinn.data.dataset import build_dataloaders
        train_loader, val_loader = build_dataloaders(
            train_items=train_items,
            val_items=val_items if val_items else None,
            cache_dir=cache_dir,
            batch_size=cfg.get("train", {}).get("batch_size", 8),
            num_workers=cfg.get("train", {}).get("num_workers", 8),
            patch_size=tuple(data_cfg.get("patch_size", (128, 128, 128))),
            spatial_size=tuple(data_cfg.get("spatial_size", (192, 192, 192))),
            num_samples=data_cfg.get("num_samples", 1),
            gmm_prob_real=aug_cfg.get("gmm_prob_real", 0.8),
            gmm_mu_jitter=aug_cfg.get("gmm_mu_jitter", 0.05),
            gmm_sigma_jitter=aug_cfg.get("gmm_sigma_jitter", 0.1),
            scale_factor=aug_cfg.get("scale_factor", 0.15),
            shift_offset=aug_cfg.get("shift_offset", 0.1),
            noise_std=_resolve_max(aug_cfg.get("noise_std", 0.03)),
            smooth_sigma=tuple(aug_cfg.get("smooth_sigma", (0.3, 0.7))),
            bias_coeff=aug_cfg.get("bias_coeff", 0.25),
            gibbs_alpha=tuple(aug_cfg.get("gibbs_alpha", (0.0, 0.5))),
            use_dist=use_dist,
        )
        is_2d = False

    # ── Build persistent cache (multi-rank safe) ──
    if args.build_cache:
        is_main = not use_dist
        if use_dist:
            import torch.distributed as dist
            is_main = dist.get_rank() == 0
        if is_main:
            print("Building persistent cache ...", flush=True)
        _cache_loader(train_loader, "train", use_dist)
        if val_loader:
            _cache_loader(val_loader, "val", use_dist)

    if model_type == "fastsurfervinn":
        from fastsurfer_vinn.models.vinn import FastSurferVINN2D
        model = FastSurferVINN2D(
            in_ch=model_cfg.get("input_nc", 1),
            out_ch=model_cfg.get("output_nc", 35),
            num_filters=model_cfg.get("num_filters", 64),
            num_filters_interpol=model_cfg.get("num_filters_interpol", 64),
        )

    elif spatial_dims == 2:
        from fastsurfer_vinn.models.unet2d import UNet2D
        model = UNet2D(
            in_ch=model_cfg.get("input_nc", 1),
            out_ch=model_cfg.get("output_nc", 35),
            channels=tuple(model_cfg.get("channels", (32, 64, 128, 256))),
            strides=tuple(tuple(s) for s in model_cfg.get("strides", [[1,1],[2,2],[2,2],[2,2]])),
            deep_supervision=model_cfg.get("deep_supervision", True),
            deep_supr_num=model_cfg.get("deep_supr_num", 2),
            dropout=model_cfg.get("dropout", 0.1),
        )
    else:
        from fastsurfer_vinn.models import UNet3D
        model = UNet3D(
            in_ch=model_cfg.get("input_nc", 4),
            out_ch=model_cfg.get("output_nc", 35),
            channels=tuple(model_cfg.get("channels", (32, 64, 128, 256))),
            strides=tuple(tuple(s) for s in model_cfg.get("strides", [[1,1,1],[2,2,2],[2,2,2],[2,2,2]])),
            deep_supervision=model_cfg.get("deep_supervision", True),
            deep_supr_num=model_cfg.get("deep_supr_num", 2),
            dropout=model_cfg.get("dropout", 0.1),
        )

    train_cfg = cfg.get("train", {})
    train_cfg["spatial_dims"] = spatial_dims

    from fastsurfer_vinn.training import Trainer
    trainer = Trainer(
        cfg=train_cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_dist=use_dist,
        local_rank=local_rank,
    )

    try:
        trainer.train(checkpoint=args.checkpoint, resume=args.resume)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
