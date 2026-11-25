# -*- coding: utf-8 -*-
import torch.nn as nn
from monai.networks.nets import DynUNet
from fastsurfer_vinn.utils.labels import NUM_CLASSES


class UNet2D(nn.Module):
    """2D DynUNet for multi-class brain segmentation.

    Same architecture as UNet3D but with spatial_dims=2.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = NUM_CLASSES,
        channels: tuple = (32, 64, 128, 256),
        strides: tuple = ((1, 1), (2, 2), (2, 2), (2, 2)),
        deep_supervision: bool = True,
        deep_supr_num: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        n_enc = len(channels)
        self._deep_supervision = deep_supervision

        self.unet = DynUNet(
            spatial_dims=2,
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
            return out
        return out[0] if isinstance(out, list) else out

    @property
    def deep_supervision(self) -> bool:
        return self._deep_supervision
