# -*- coding: utf-8 -*-
from pathlib import Path

import torch
from monai.data import PersistentDataset, Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai.transforms import (
    Compose, LoadImaged, Orientationd, Spacingd, SpatialPadd,
    CenterSpatialCropd, NormalizeIntensityd,
    ClipIntensityPercentilesd,
    EnsureChannelFirstd,
    RandScaleIntensityd, RandShiftIntensityd,
    RandGaussianNoised, RandGaussianSmoothd, RandGibbsNoised, RandBiasFieldd,
    RandCropByPosNegLabeld, RandAffined,
    EnsureTyped, MapTransform,
)

from fastsurfer_vinn.utils.labels import _LUT


# ── Label remap ──────────────────────────────────────────────────────────────

class RemapLabelsd(MapTransform):
    """Remap FreeSurfer labels → continuous [0..34] (pure torch)."""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            arr = d[key].clone()
            arr = arr.round().long()

            mask = arr == 77
            if mask.any():
                xmid = arr.shape[1] // 2
                left  = mask.clone(); left [:, xmid:] = False
                right = mask.clone(); right[:, :xmid] = False
                arr[left]  = 2
                arr[right] = 41

            d[key] = _LUT[arr]
        return d


# ── GMM stats computation (cache-phase transform) ────────────────────────────

class ComputeGMMStatsd(MapTransform):
    def __init__(self, keys, n_classes: int = 35):
        super().__init__(keys)
        self.n_classes = n_classes

    def __call__(self, data):
        d = dict(data)
        lbl_key, img_key = self.keys
        label = d[lbl_key].squeeze(0)
        real  = d[img_key].float().squeeze(0)

        stats = {}
        for l in range(1, self.n_classes):
            mask = label == l
            nv = mask.sum().item()
            if nv > 10:
                stats[l] = (real[mask].mean().item(), real[mask].std().item())
        d["gmm_stats"] = stats
        return d


# ── GMM synthesis (runtime transform) ────────────────────────────────────────

class GMMSample3Dd(MapTransform):
    def __init__(self, keys, output_key: str = "image", stats_key: str = "gmm_stats",
                 prob_real: float = 0.8, mu_jitter: float = 0.05, sigma_jitter: float = 0.1):
        super().__init__(keys)
        self.output_key = output_key
        self.stats_key = stats_key
        self.prob_real = prob_real
        self.mu_jitter = mu_jitter
        self.sigma_jitter = sigma_jitter

    def __call__(self, data):
        d = dict(data)
        if torch.rand(1).item() < self.prob_real:
            return d

        label = d[self.keys[0]]
        stats = d[self.stats_key]

        n = int(label.max().item()) + 1
        mus = torch.zeros(n, device=label.device)
        sigmas = torch.zeros(n, device=label.device)

        present = list(stats.values())
        if present:
            mu_global  = sum(m for m, _ in present) / len(present)
            std_global = sum(s for _, s in present) / len(present)
        else:
            mu_global, std_global = 0.0, 0.5

        for l in range(1, n):
            if l in stats:
                mu_l, std_l = stats[l]
                mus[l] = mu_l + std_l * torch.randn(1, device=label.device).item() * self.mu_jitter
                sigmas[l] = std_l * (
                    1.0 - self.sigma_jitter + 2 * self.sigma_jitter
                    * torch.rand(1, device=label.device).item()
                )
            else:
                mus[l] = mu_global + std_global * 0.1 * torch.randn(1, device=label.device).item()
                sigmas[l] = std_global * 0.5

        img = mus[label.long()] + sigmas[label.long()] * torch.randn_like(label.float())
        img.clamp_(-5.0, 5.0)
        d[self.output_key] = img
        return d


# ── Cache-phase transforms ───────────────────────────────────────────────────

def volume_transforms(spatial_size: tuple = (192, 192, 192)):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        EnsureTyped(keys=["image", "label"],dtype=[torch.float32, torch.long], track_meta=False),
        Spacingd(keys=["image", "label"], pixdim=1.0,
                 mode=("bilinear", "nearest")),
        RemapLabelsd(keys=["label"]),
        ClipIntensityPercentilesd(keys=["image"], lower=0.5, upper=99.5),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        SpatialPadd(keys=["image", "label"], spatial_size=spatial_size,
                    method="symmetric"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=spatial_size),
        ComputeGMMStatsd(keys=["label", "image"]),
        EnsureTyped(keys=["image", "label"],
                    dtype=[torch.float16, torch.uint8], track_meta=False),
    ])


# ── Per-patch augmentations (GMM → intensity → geometric) ───────────

def patch_aug(
    gmm_prob_real: float = 0.8,
    mu_jitter: float = 0.05,
    sigma_jitter: float = 0.1,
    scale_factor: float = 0.15,
    shift_offset: float = 0.1,
    noise_std: tuple = (0.03, 0.03),
    smooth_sigma: tuple = (0.3, 0.7),
    bias_coeff: float = 0.25,
    gibbs_alpha: tuple = (0.0, 0.5),
):
    return Compose([
        GMMSample3Dd(
            keys=["label"], output_key="image", stats_key="gmm_stats",
            prob_real=gmm_prob_real, mu_jitter=mu_jitter, sigma_jitter=sigma_jitter,
        ),
        RandScaleIntensityd(keys=["image"], factors=scale_factor, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=shift_offset, prob=0.5),
        RandGaussianNoised(keys=["image"], std=noise_std, prob=0.3),
        RandGaussianSmoothd(
            keys=["image"], prob=0.2,
            sigma_x=smooth_sigma, sigma_y=smooth_sigma, sigma_z=smooth_sigma,
        ),
        RandGibbsNoised(keys=["image"], alpha=gibbs_alpha, prob=0.2),
        RandBiasFieldd(keys=["image"], coeff_range=(0.0, bias_coeff), prob=0.3),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(0.087, 0.087, 0.087),
            scale_range=(0.05, 0.05, 0.05),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        EnsureTyped(keys=["image", "label"],
                    dtype=[torch.float32, torch.long], track_meta=False),
    ])


# ── Training dataset ─────────────────────────────────────────────────────────

class PatchTrainingDataset(Dataset):
    """Yield random patches from cached volumes.

    Pipeline (single Compose):
      crop → float() → GMM/real → intensity augs → geometric augs.
    """

    def __init__(
        self,
        items: list[dict],
        cache_dir: str | Path,
        spatial_size: tuple = (192, 192, 192),
        patch_size: tuple = (128, 128, 128),
        num_samples: int = 1,
        pos_weight: float = 5.0,
        augment: bool = True,
        gmm_prob_real: float = 0.8,
        gmm_mu_jitter: float = 0.05,
        gmm_sigma_jitter: float = 0.1,
        scale_factor: float = 0.15,
        shift_offset: float = 0.1,
        noise_std: tuple = (0.03, 0.03),
        smooth_sigma: tuple = (0.3, 0.7),
        bias_coeff: float = 0.25,
        gibbs_alpha: tuple = (0.0, 0.5),
    ):
        self.num_samples = num_samples
        self.cached = PersistentDataset(
            data=items,
            transform=volume_transforms(spatial_size),
            cache_dir=cache_dir,
        )
        self.cropper = RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=pos_weight,
            neg=1.0,
            num_samples=num_samples,
        )
        self.aug = patch_aug(
            gmm_prob_real, gmm_mu_jitter, gmm_sigma_jitter,
            scale_factor=scale_factor, shift_offset=shift_offset,
            noise_std=noise_std, smooth_sigma=smooth_sigma,
            bias_coeff=bias_coeff, gibbs_alpha=gibbs_alpha,
        ) if augment else None
        self.cropper = RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=pos_weight,
            neg=1.0,
            num_samples=num_samples,
        )
        self.aug = patch_aug(
            gmm_prob_real, gmm_mu_jitter, gmm_sigma_jitter,
            scale_factor=scale_factor, shift_offset=shift_offset,
            noise_std=noise_std, smooth_sigma=smooth_sigma,
            bias_coeff=bias_coeff, gibbs_alpha=gibbs_alpha,
        )

    def __len__(self):
        return len(self.cached) * self.num_samples

    def __getitem__(self, idx):
        vol_idx = idx // self.num_samples
        crop_idx = idx % self.num_samples

        vol = self.cached[vol_idx]
        patch = dict(self.cropper(vol)[crop_idx])
        patch["image"] = patch["image"].float()
        patch["label"] = patch["label"].float()

        if self.aug is not None:
            patch = self.aug(patch)
        img, lab = patch["image"], patch["label"].squeeze(0)
        return img, lab


def build_dataloaders(
    train_items: list[dict],
    cache_dir: str | Path,
    val_items: list[dict] | None = None,
    batch_size: int = 8,
    num_workers: int = 8,
    patch_size: tuple = (128, 128, 128),
    spatial_size: tuple = (192, 192, 192),
    num_samples: int = 1,
    gmm_prob_real: float = 0.8,
    gmm_mu_jitter: float = 0.05,
    gmm_sigma_jitter: float = 0.1,
    scale_factor: float = 0.15,
    shift_offset: float = 0.1,
    noise_std: tuple = (0.03, 0.03),
    smooth_sigma: tuple = (0.3, 0.7),
    bias_coeff: float = 0.25,
    gibbs_alpha: tuple = (0.0, 0.5),
    use_dist: bool = False,
) -> tuple[DataLoader, DataLoader | None]:
    train_ds = PatchTrainingDataset(
        train_items, cache_dir, spatial_size, patch_size, num_samples,
        gmm_prob_real=gmm_prob_real,
        gmm_mu_jitter=gmm_mu_jitter,
        gmm_sigma_jitter=gmm_sigma_jitter,
        scale_factor=scale_factor, shift_offset=shift_offset,
        noise_std=noise_std, smooth_sigma=smooth_sigma,
        bias_coeff=bias_coeff, gibbs_alpha=gibbs_alpha,
    )

    if use_dist:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True, timeout=120,
            persistent_workers=num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, timeout=120,
            persistent_workers=num_workers > 0,
        )

    val_loader = None
    if val_items:
        val_ds = PatchTrainingDataset(
            val_items, cache_dir, spatial_size, patch_size,
            num_samples=num_samples,
            augment=False,
            gmm_prob_real=1.0,
        )
        if use_dist:
            val_sampler = DistributedSampler(val_ds, shuffle=False)
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, sampler=val_sampler,
                num_workers=num_workers, pin_memory=True,
                persistent_workers=num_workers > 0, timeout=120,
            )
        else:
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
                persistent_workers=num_workers > 0, timeout=120,
            )

    return train_loader, val_loader
