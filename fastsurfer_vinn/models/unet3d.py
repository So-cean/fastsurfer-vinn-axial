# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DynUNet
from fastsurfer_vinn.utils.labels import NUM_CLASSES


class UNet3D(nn.Module):
    """3D DynUNet for multi-class brain segmentation.

    Supports deep supervision (``deep_supervision=True``). In training mode
    ``forward`` returns a ``list[tensor]``:
        ``[main_logits, aux_1, aux_2, ...]``
    each with shape ``(B, C, D, H, W)`` at progressively coarser resolutions.
    The caller (trainer) is responsible for upsampling and weighting aux losses.

    Parameters
    ----------
    in_ch : int
        Input channels (default 1: single MRI channel).
    out_ch : int
        Output classes (default ``NUM_CLASSES`` = 35).
        Output classes (default ``NUM_CLASSES`` = 35).
    channels : tuple
        Feature channels per encoder level. Default (32, 64, 128, 256).
    strides : tuple of tuples
        Strides per encoder level. First stride ``(1, 1, 1)`` preserves resolution;
        subsequent strides ``(2, 2, 2)`` downsample. Default as below.
    deep_supervision : bool
        Enable deep supervision aux heads (default True).
    deep_supr_num : int
        Number of aux heads (default 2).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = NUM_CLASSES,
        channels: tuple = (32, 64, 128, 256),
        strides: tuple = ((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        deep_supervision: bool = True,
        deep_supr_num: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        n_enc = len(channels)
        self._deep_supervision = deep_supervision

        self.unet = DynUNet(
            spatial_dims=3,
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=[3] * n_enc,
            strides=list(strides),
            upsample_kernel_size=list(strides[1:]),
            filters=list(channels),
            dropout=dropout,
            deep_supervision=deep_supervision,
            deep_supr_num=deep_supr_num,
        )

    def forward(self, x):
        out = self.unet(x)
        if self.training and self._deep_supervision:
            return out  # [main, aux1, aux2, ...]
        return out[0] if isinstance(out, list) else out

    @property
    def deep_supervision(self) -> bool:
        return self._deep_supervision
