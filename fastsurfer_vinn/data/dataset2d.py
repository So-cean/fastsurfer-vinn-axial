# -*- coding: utf-8 -*-
"""2D slice training dataset — reuses 3D PersistentDataset cache.

Extracts random axial slices from cached 192³ volumes,
applies 2D augmentations independently per slice.
"""

import random
from pathlib import Path

import torch
from monai.data import PersistentDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai.transforms import (
    Compose,
    RandScaleIntensityd, RandShiftIntensityd,
    RandGaussianNoised, RandGaussianSmoothd, RandGibbsNoised, RandBiasFieldd,
    RandAffined,
    ThresholdIntensityd,
    EnsureTyped,
)

from fastsurfer_vinn.data.dataset import volume_transforms, GMMSample3Dd


# ── 2D augmentation pipeline (matches 3D config, no Z-dim) ───────────

def slice_aug_2d(
    scale_factor: float = 0.20,
    shift_offset: float = 0.10,
    noise_std: float = 0.05,
    smooth_sigma: tuple = (0.3, 0.8),
    gibbs_alpha: tuple = (0.0, 0.5),
    bias_coeff: float = 0.30,
    gmm_prob_real: float = 0.8,
    gmm_mu_jitter: float = 0.05,
    gmm_sigma_jitter: float = 0.1,
):
    """2D intensity + geometric augmentation, applied per slice."""
    return Compose([
        GMMSample3Dd(
            keys=["label"], output_key="image", stats_key="gmm_stats",
            prob_real=gmm_prob_real, mu_jitter=gmm_mu_jitter,
            sigma_jitter=gmm_sigma_jitter,
        ),
        RandScaleIntensityd(keys=["image"], factors=scale_factor, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=shift_offset, prob=0.5),
        RandGaussianNoised(keys=["image"], std=noise_std, prob=0.3),
        RandGaussianSmoothd(
            keys=["image"], prob=0.2,
            sigma_x=smooth_sigma, sigma_y=smooth_sigma,
        ),
        RandGibbsNoised(keys=["image"], alpha=gibbs_alpha, prob=0.2),
        RandBiasFieldd(keys=["image"], coeff_range=(0.0, bias_coeff), prob=0.3),
        ThresholdIntensityd(keys=["image"], threshold=-5.0, above=True,  cval=-5.0),
        ThresholdIntensityd(keys=["image"], threshold=5.0,  above=False, cval=5.0),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=0.087,
            scale_range=0.05,
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"],
                    dtype=[torch.float32, torch.long], track_meta=False),
    ])


# ── 2D training dataset ─────────────────────────────────────────────

class SliceTrainingDataset2D(torch.utils.data.Dataset):
    """2D slice training: axial slices from 3D cached volumes.

    Each volume yields ``num_samples`` random axial slices from the
    brain ROI (Z range 30-150).  Slices are augmented independently
    with 2D transforms.  ``__len__`` returns number of *volumes*,
    not slices (batch_size configures slices per batch).
    """

    Z_MIN, Z_MAX = 30, 150

    def __init__(
        self,
        items: list[dict],
        cache_dir: str | Path,
        spatial_size: tuple = (192, 192, 192),
        num_samples: int = 8,
        augment: bool = True,
        scale_factor: float = 0.20,
        shift_offset: float = 0.10,
        noise_std: float = 0.05,
        smooth_sigma: tuple = (0.3, 0.8),
        gibbs_alpha: tuple = (0.0, 0.5),
        bias_coeff: float = 0.30,
        gmm_prob_real: float = 0.8,
        gmm_mu_jitter: float = 0.05,
        gmm_sigma_jitter: float = 0.1,
    ):
        self.num_samples = num_samples
        self.num_z = self.Z_MAX - self.Z_MIN + 1
        self.cached = PersistentDataset(
            data=items,
            transform=volume_transforms(spatial_size),
            cache_dir=cache_dir,
        )
        self.aug = slice_aug_2d(
            scale_factor, shift_offset, noise_std,
            smooth_sigma, gibbs_alpha, bias_coeff,
            gmm_prob_real, gmm_mu_jitter, gmm_sigma_jitter,
        ) if augment else None

    def __len__(self):
        return len(self.cached)

    def __getitem__(self, idx):
        vol = self.cached[idx]
        imgs, labs = [], []
        for _ in range(self.num_samples):
            z = random.randint(self.Z_MIN, self.Z_MAX)
            item = {
                "image": vol["image"][:, :, :, z].float(),
                "label": vol["label"][:, :, :, z].float(),
                "gmm_stats": vol["gmm_stats"],
            }
            if self.aug is not None:
                item = self.aug(item)
            imgs.append(item["image"])
            labs.append(item["label"])
        return torch.stack(imgs, dim=0), torch.stack(labs, dim=0).squeeze(1)


# ── Custom collate: flatten num_samples into batch dim ─────────────

def collate_2d(batch):
    """Collate (N, C, H, W), (N, H, W) lists into (B*N, C, H, W), (B*N, H, W)."""
    imgs = torch.cat([item[0] for item in batch], dim=0)
    labs = torch.cat([item[1] for item in batch], dim=0)
    return imgs, labs


# ── Dataloader factory ──────────────────────────────────────────────

def build_dataloaders_2d(
    train_items: list[dict],
    cache_dir: str | Path,
    val_items: list[dict] | None = None,
    batch_size: int = 64,
    num_workers: int = 8,
    num_samples: int = 8,
    spatial_size: tuple = (192, 192, 192),
    use_dist: bool = False,
    **aug_kwargs,
) -> tuple[DataLoader, DataLoader | None]:
    train_ds = SliceTrainingDataset2D(
        train_items, cache_dir, spatial_size, num_samples,
        augment=True, **aug_kwargs,
    )
    if use_dist:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True, timeout=120,
            persistent_workers=num_workers > 0,
            collate_fn=collate_2d,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, timeout=120,
            persistent_workers=num_workers > 0,
            collate_fn=collate_2d,
        )

    val_loader = None
    if val_items:
        val_ds = SliceTrainingDataset2D(
            val_items, cache_dir, spatial_size, num_samples,
            augment=False,
        )
        if use_dist:
            val_sampler = DistributedSampler(val_ds, shuffle=False)
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, sampler=val_sampler,
                num_workers=num_workers, pin_memory=True, timeout=120,
                persistent_workers=num_workers > 0,
                collate_fn=collate_2d,
            )
        else:
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True, timeout=120,
                persistent_workers=num_workers > 0,
                collate_fn=collate_2d,
            )

    return train_loader, val_loader
