# -*- coding: utf-8 -*-
"""FastSurferVINN 2D — simplified single-channel version for brain segmentation.

Port from Deep-MI/FastSurfer (Apache-2.0) with modifications:
  - in_ch=1 (single slice, not 7)
  - out_ch=35 (aseg, not DKT 95)
  - num_filters=64
  - single resolution (192), no multi-scale
  - pure torch/nn, no yacs dependency
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

# ── NPU detection (zero-overhead: resolved at import time) ──
try:
    import torch_npu
    _NPU = True
except ImportError:
    _NPU = False


# ===================================================================
# Interpolation Layer (Zoom2d) — resolution-independent normalization
# ===================================================================

class _ZoomNd(nn.Module):
    def __init__(self, interpolation_mode="nearest"):
        super().__init__()
        self._mode = interpolation_mode
        self._N = -1
        self._target_shape = tuple()

    @property
    def target_shape(self):
        return self._target_shape

    @target_shape.setter
    def target_shape(self, target_shape):
        if target_shape is not None:
            tup = tuple(target_shape)
        else:
            tup = tuple()
        if len(tup) not in (0, self._N):
            raise TypeError(f"expected {self._N}-sequence, got {target_shape}")
        self._target_shape = tup

    def forward(self, input_tensor, scale_factors, rescale=False):
        if self._N == -1:
            raise RuntimeError("Direct instantiation not supported")
        if input_tensor.dim() != 2 + self._N:
            raise ValueError(f"expected {self._N+2}D, got {input_tensor.dim()}D")
        if not self._target_shape:
            raise AttributeError("target_shape not set")
        scales_chunks = list(zip(*self._fix_scale_factors(scale_factors, input_tensor.shape[0])))
        scales, chunks = map(list, scales_chunks)
        interp, scales_out = [], []
        for tensor, scale_f, num in zip(torch.split(input_tensor, chunks, dim=0), scales, chunks):
            if rescale:
                scale_f = [1 / sf for sf in scale_f] if isinstance(scale_f, list) else torch.div(1, scale_f)
            image, sf = self._interpolate(tensor, scale_f)
            interp.append(image)
            scales_out.extend([sf] * num)
        return torch.cat(interp, dim=0), scales_out

    def _fix_scale_factors(self, scale_factors, batch_size):
        if isinstance(scale_factors, (torch.Tensor, np.ndarray)):
            batch_size_sf = scale_factors.shape[0]
        elif isinstance(scale_factors, Sequence):
            scale_factors = list(scale_factors)
            batch_size_sf = len(scale_factors)
        else:
            batch_size_sf = None
        if isinstance(scale_factors, (torch.Tensor, np.ndarray)):
            if scale_factors.ndim == 0:
                scale_factors = float(scale_factors)
            elif len(scale_factors) == 1:
                scale_factors = [float(scale_factors[0])] * batch_size
            else:
                scale_factors = [float(sf) for sf in scale_factors]
        if isinstance(scale_factors, Sequence):
            num = 0
            last_sf = None
            for i, sf in enumerate(scale_factors):
                if isinstance(sf, (int, float)):
                    sf = [sf] * self._N
                else:
                    if isinstance(sf, (np.ndarray, torch.Tensor)):
                        if isinstance(sf, torch.Tensor) and sf.dim() == 0:
                            sf = [sf] * self._N
                        else:
                            sf_dim = sf.shape[0]
                            sf = [sf] * self._N if sf_dim == 1 else sf
                    elif isinstance(sf, Sequence):
                        sf = list(sf)
                        sf = sf * self._N if len(sf) == 1 else sf
                    else:
                        raise ValueError(f"invalid scale factor: {type(sf)}")
                    if len(sf) != self._N:
                        raise ValueError(f"expected {self._N} factors, got {len(sf)}")
                if last_sf is not None and any(a != b for a, b in zip(last_sf, sf)):
                    yield last_sf, num
                    num = 0
                num += 1
                last_sf = sf
            if last_sf is not None:
                yield last_sf, num
        elif isinstance(scale_factors, (int, float)):
            yield [scale_factors] * self._N, batch_size
        else:
            raise ValueError(f"invalid scale_factors: {type(scale_factors)}")

    def _interpolate(self, *args):
        raise NotImplementedError

    def _calculate_crop_pad(self, in_shape, scale_factor, dim, alignment):
        this_in = in_shape[dim + 2]
        source = self._target_shape[dim] * scale_factor[dim]
        start, end = 0, this_in
        if alignment == "mid" and (this_in - int(source)) % 2 != 0:
            out = int(source / 2) * 2
        else:
            out = int(source)
        rescale = out / self._target_shape[dim]
        if out > this_in:
            padding = (1.0, 0.0) if alignment == "from_end" else (0.5, 0.5) if alignment == "mid" else (0.0, 1.0)
            interp_target = int(this_in / scale_factor[dim])
            if alignment == "mid" and (interp_target - self._target_shape[dim]) % 2 != 0:
                interp_target -= 1
            rescale = this_in / interp_target
            full_pad = self._target_shape[dim] - interp_target
            padding = tuple(int(p * full_pad) for p in padding)
        else:
            if alignment == "mid":
                start = int((end - out) / 2)
            if alignment == "from_end":
                start = end - out
            else:
                end = start + out
            padding, interp_target = (0, 0), self._target_shape[dim]
        scale_factor[dim] = rescale
        return slice(start, end), scale_factor, padding, interp_target


class Zoom2d(_ZoomNd):
    def __init__(self, target_shape=None, interpolation_mode="bilinear", crop_position="top_left"):
        super().__init__(interpolation_mode)
        self._N = 2
        self._target_shape = ()
        self.crop_position = crop_position
        self.target_shape = target_shape

    def _interpolate(self, data, scale_factor):
        if isinstance(scale_factor, np.ndarray):
            scale_factor = scale_factor.tolist()
        elif isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.tolist() if scale_factor.shape == (2,) else list(scale_factor)
        elif isinstance(scale_factor, Sequence):
            scale_factor = list(scale_factor)
        else:
            raise ValueError(f"invalid scale_factor: {type(scale_factor)}")

        vert_align = "mid" if self.crop_position == "center" else "from_start"
        horz_align = "mid" if self.crop_position == "center" else "from_start"

        if self.crop_position.startswith("bottom"):
            vert_align = "from_end"
        if self.crop_position.endswith("right"):
            horz_align = "from_end"

        tb_slice, scale_factor, pad_tb, shape_tb = self._calculate_crop_pad(data.shape, scale_factor, 0, vert_align)
        lr_slice, scale_factor, pad_lr, shape_lr = self._calculate_crop_pad(data.shape, scale_factor, 1, horz_align)

        data = data[:, :, tb_slice, lr_slice]

        if data.shape[2:] != (shape_tb, shape_lr):
            data = F.interpolate(data, size=(shape_tb, shape_lr), mode=self._mode, align_corners=False)

        if pad_tb != (0, 0) or pad_lr != (0, 0):
            data = F.pad(data, pad=pad_lr + pad_tb)

        return data, scale_factor


# ===================================================================
# NPU-compatible MaxPool / MaxUnpool (zero-overhead branch at import)
# ===================================================================


def _max_pool_2d_with_indices(x, kH, kW, sH, sW):
    """MaxPool2d with indices — built from F.unfold + argmax.

    Returns standard (B, C, H_out, W_out) int64 indices, identical to CPU.
    """
    B, C, H, W = x.shape
    patches = F.unfold(x, kernel_size=(kH, kW), stride=(sH, sW))
    patches = patches.view(B, C, kH * kW, -1)
    _out, local_idx = patches.max(dim=2)

    H_out = (H - kH) // sH + 1
    W_out = (W - kW) // sW + 1

    h_grid = torch.arange(H_out, device=x.device).view(1, 1, -1, 1) * sH
    w_grid = torch.arange(W_out, device=x.device).view(1, 1, 1, -1) * sW

    local_h = (local_idx // kW).view(B, C, H_out, W_out)
    local_w = (local_idx % kW).view(B, C, H_out, W_out)

    global_idx = ((h_grid + local_h) * W + (w_grid + local_w)).to(torch.int64)
    return _out.view(B, C, H_out, W_out), global_idx


def _max_unpool_2d(x, indices, output_size):
    """MaxUnpool2d — built from scatter_.  Standard int64 indices."""
    B, C = x.shape[:2]
    H_out, W_out = output_size
    N = H_out * W_out

    b_off = torch.arange(B, device=x.device).view(B, 1, 1, 1) * C * N
    c_off = torch.arange(C, device=x.device).view(1, C, 1, 1) * N
    flat_idx = (indices.int() + b_off + c_off).reshape(-1)

    out = torch.zeros(B * C * N, device=x.device, dtype=x.dtype)
    out.scatter_(0, flat_idx, x.reshape(-1))
    return out.view(B, C, H_out, W_out)


class _NativeMaxPool2d(nn.Module):
    def __init__(self, k, s):
        super().__init__()
        self.pool = nn.MaxPool2d(k, s, return_indices=True)

    def forward(self, x):
        return self.pool(x)


class _NPUMaxPool2d(nn.Module):
    def __init__(self, k, s):
        super().__init__()
        self._k, self._s = k, s

    def forward(self, x):
        return _max_pool_2d_with_indices(x, self._k, self._k, self._s, self._s)


class _NativeMaxUnpool2d(nn.Module):
    def __init__(self, k, s):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(k, s)

    def forward(self, x, idx, out_size=None):
        if out_size is not None:
            return self.unpool(x, idx, output_size=out_size)
        return self.unpool(x, idx)


class _NPUMaxUnpool2d(nn.Module):
    def __init__(self, k, s):
        super().__init__()
        self._k = k

    def forward(self, x, idx, out_size):
        return _max_unpool_2d(x, idx, out_size)


# Resolve at import time — zero branch overhead in forward
if _NPU:
    _MaxPool = _NPUMaxPool2d
    _MaxUnpool = _NPUMaxUnpool2d
else:
    _MaxPool = _NativeMaxPool2d
    _MaxUnpool = _NativeMaxUnpool2d


# ===================================================================
# Building Blocks
# ===================================================================

class InputDenseBlock(nn.Module):
    def __init__(self, in_ch, num_filters_interpol, kernel_h=3, kernel_w=3, stride_conv=1):
        super().__init__()
        ph, pw = (kernel_h - 1) // 2, (kernel_w - 1) // 2
        self.conv0 = nn.Conv2d(in_ch, num_filters_interpol, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.conv1 = nn.Conv2d(num_filters_interpol, num_filters_interpol, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.conv2 = nn.Conv2d(num_filters_interpol, num_filters_interpol, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.conv3 = nn.Conv2d(num_filters_interpol, num_filters_interpol, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.bn0 = nn.BatchNorm2d(in_ch)
        self.gn1 = nn.BatchNorm2d(num_filters_interpol)
        self.gn2 = nn.BatchNorm2d(num_filters_interpol)
        self.gn3 = nn.BatchNorm2d(num_filters_interpol)
        self.gn4 = nn.BatchNorm2d(num_filters_interpol)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x0 = self.bn0(x)
        x0 = self.conv0(x0)
        x1_gn = self.gn1(x0)
        x1 = self.prelu(x1_gn)
        x1 = self.conv1(x1)
        x2_gn = self.gn2(x1)
        x2_max = torch.maximum(x2_gn, x1_gn)
        x2 = self.prelu(x2_max)
        x2 = self.conv2(x2)
        x3_gn = self.gn3(x2)
        x3_max = torch.maximum(x3_gn, x2_max)
        x3 = self.prelu(x3_max)
        x3 = self.conv3(x3)
        out = self.gn4(x3)
        return out


class CompetitiveDenseBlock(nn.Module):
    def __init__(self, in_ch, num_filters, kernel_h=3, kernel_w=3, stride_conv=1, outblock=False, out_ch=None):
        super().__init__()
        ph, pw = (kernel_h - 1) // 2, (kernel_w - 1) // 2
        out_ch = out_ch or num_filters
        self.conv0 = nn.Conv2d(in_ch, num_filters, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.conv1 = nn.Conv2d(num_filters, num_filters, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.conv2 = nn.Conv2d(num_filters, num_filters, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.conv3 = nn.Conv2d(num_filters, out_ch, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.bn4 = nn.BatchNorm2d(out_ch) if not outblock else nn.Identity()
        self.prelu = nn.PReLU()

    def forward(self, x):
        x0 = self.prelu(x)
        x0 = self.conv0(x0)
        x1 = self.bn1(x0)
        x1_max = torch.maximum(x, x1)
        x1 = self.prelu(x1_max)
        x1 = self.conv1(x1)
        x2 = self.bn2(x1)
        x2_max = torch.maximum(x2, x1_max)
        x2 = self.prelu(x2_max)
        x2 = self.conv2(x2)
        x3 = self.bn3(x2)
        x3_max = torch.maximum(x3, x2_max)
        x3 = self.prelu(x3_max)
        x3 = self.conv3(x3)
        out = self.bn4(x3)
        return out


class CompetitiveDenseBlockInput(nn.Module):
    def __init__(self, in_ch, num_filters, kernel_h=3, kernel_w=3, stride_conv=1):
        super().__init__()
        ph, pw = (kernel_h - 1) // 2, (kernel_w - 1) // 2
        self.conv0 = nn.Conv2d(in_ch, num_filters, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.conv1 = nn.Conv2d(num_filters, num_filters, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.conv2 = nn.Conv2d(num_filters, num_filters, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.conv3 = nn.Conv2d(num_filters, num_filters, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.bn0 = nn.BatchNorm2d(in_ch)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.bn4 = nn.BatchNorm2d(num_filters)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x0 = self.bn0(x)
        x0 = self.conv0(x0)
        x1_bn = self.bn1(x0)
        x1 = self.prelu(x1_bn)
        x1 = self.conv1(x1)
        x2_bn = self.bn2(x1)
        x2_max = torch.maximum(x2_bn, x1_bn)
        x2 = self.prelu(x2_max)
        x2 = self.conv2(x2)
        x3_bn = self.bn3(x2)
        x3_max = torch.maximum(x3_bn, x2_max)
        x3 = self.prelu(x3_max)
        x3 = self.conv3(x3)
        out = self.bn4(x3)
        return out


class CompetitiveEncoderBlock(nn.Module):
    def __init__(self, in_ch, num_filters, kernel_h=3, kernel_w=3, stride_conv=1, pool=2, stride_pool=2):
        super().__init__()
        self.block = CompetitiveDenseBlock(in_ch, num_filters, kernel_h, kernel_w, stride_conv)
        self.maxpool = _MaxPool(pool, stride_pool)

    def forward(self, x):
        out = self.block(x)
        pooled, indices = self.maxpool(out)
        return pooled, out, indices


class CompetitiveEncoderBlockInput(nn.Module):
    def __init__(self, in_ch, num_filters, kernel_h=3, kernel_w=3, stride_conv=1, pool=2, stride_pool=2):
        super().__init__()
        self.block = CompetitiveDenseBlockInput(in_ch, num_filters, kernel_h, kernel_w, stride_conv)
        self.maxpool = _MaxPool(pool, stride_pool)

    def forward(self, x):
        out = self.block(x)
        pooled, indices = self.maxpool(out)
        return pooled, out, indices


class CompetitiveDecoderBlock(nn.Module):
    def __init__(self, in_ch, num_filters, kernel_h=3, kernel_w=3, stride_conv=1, pool=2, stride_pool=2, out_ch=None):
        super().__init__()
        self.unpool = _MaxUnpool(pool, stride_pool)
        self.block = CompetitiveDenseBlock(in_ch, num_filters, kernel_h, kernel_w, stride_conv, out_ch=out_ch)

    def forward(self, x, skip, indices):
        unpooled = self.unpool(x, indices, skip.shape[2:])
        concat = torch.maximum(unpooled, skip)
        return self.block(concat)


class OutputDenseBlock(nn.Module):
    def __init__(self, in_ch, num_filters, kernel_h=3, kernel_w=3, stride_conv=1):
        super().__init__()
        ph, pw = (kernel_h - 1) // 2, (kernel_w - 1) // 2
        self.conv0 = nn.Conv2d(in_ch, num_filters, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.conv1 = nn.Conv2d(num_filters, num_filters, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.conv2 = nn.Conv2d(num_filters, num_filters, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.conv3 = nn.Conv2d(num_filters, num_filters, (kernel_h, kernel_w), stride_conv, (ph, pw))
        self.gn1 = nn.BatchNorm2d(num_filters)
        self.gn2 = nn.BatchNorm2d(num_filters)
        self.gn3 = nn.BatchNorm2d(num_filters)
        self.gn4 = nn.BatchNorm2d(num_filters)
        self.prelu = nn.PReLU()

    def forward(self, x, skip):
        cat = torch.cat((x, skip), dim=1)
        x0 = self.prelu(cat)
        x0 = self.conv0(x0)
        x1_gn = self.gn1(x0)
        x1 = self.prelu(x1_gn)
        x1 = self.conv1(x1)
        x2_gn = self.gn2(x1)
        x2_max = torch.maximum(x1_gn, x2_gn)
        x2 = self.prelu(x2_max)
        x2 = self.conv2(x2)
        x3_gn = self.gn3(x2)
        x3_max = torch.maximum(x3_gn, x2_max)
        x3 = self.prelu(x3_max)
        x3 = self.conv3(x3)
        out = self.gn4(x3)
        return out


class ClassifierBlock(nn.Module):
    def __init__(self, in_ch, num_classes, kernel_c=1, stride_conv=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_c, stride_conv)

    def forward(self, x):
        return self.conv(x)


# ===================================================================
# FastSurferVINN2D (main model)
# ===================================================================

class FastSurferVINN2D(nn.Module):
    """2D VINN for brain segmentation. Resolution-independent via Zoom2d.

    Args:
        in_ch: Input channels (default 1 — single axial slice).
        out_ch: Output segmentation classes (default 35).
        num_filters: Base filter count (default 64).
        num_filters_interpol: Interpolation block filters (default 64).
        base_size: Spatial size at base resolution (default 192).
    """

    def __init__(self, in_ch=1, out_ch=35, num_filters=64,
                 num_filters_interpol=64, base_size=192):
        super().__init__()
        self.base_size = base_size

        # Input + Flex to base resolution
        self.inp_block = InputDenseBlock(in_ch, num_filters_interpol)
        self.interpol1 = Zoom2d((base_size, base_size), interpolation_mode="bilinear", crop_position="center")

        # Encoder
        self.encode1 = CompetitiveEncoderBlockInput(num_filters_interpol, num_filters)
        self.encode2 = CompetitiveEncoderBlock(num_filters, num_filters)
        self.encode3 = CompetitiveEncoderBlock(num_filters, num_filters)
        self.encode4 = CompetitiveEncoderBlock(num_filters, num_filters)

        self.bottleneck = CompetitiveDenseBlock(num_filters, num_filters)

        # Decoder
        self.decode4 = CompetitiveDecoderBlock(num_filters, num_filters)
        self.decode3 = CompetitiveDecoderBlock(num_filters, num_filters)
        self.decode2 = CompetitiveDecoderBlock(num_filters, num_filters)
        self.decode1 = CompetitiveDecoderBlock(num_filters, num_filters)

        # Output + Flex to original resolution
        self.outp_block = OutputDenseBlock(num_filters + num_filters_interpol, num_filters)
        self.classifier = ClassifierBlock(num_filters, out_ch)

        self.interpol2 = Zoom2d(interpolation_mode="bilinear", crop_position="center")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, scale_factor=None):
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W).
            scale_factor: Per-batch scale factor = base_resolution / voxel_size.
                          e.g. 1.0mm -> 1.0, 0.5mm -> 2.0, 2.0mm -> 0.5.
                          None = assume 1.0 (training data at 1mm).
                          shape: (B,).
        """
        B = x.shape[0]
        if scale_factor is None:
            scale_factor = torch.ones(B, dtype=x.dtype, device=x.device)
        if isinstance(scale_factor, (int, float)):
            scale_factor = torch.full((B,), scale_factor, dtype=x.dtype, device=x.device)

        skip_encoder_0 = self.inp_block(x)
        encoder_output0, _ = self.interpol1(skip_encoder_0, scale_factor)

        encoder_output1, skip_encoder_1, indices_1 = self.encode1(encoder_output0)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2(encoder_output1)
        encoder_output3, skip_encoder_3, indices_3 = self.encode3(encoder_output2)
        encoder_output4, skip_encoder_4, indices_4 = self.encode4(encoder_output3)

        bottleneck = self.bottleneck(encoder_output4)

        decoder_output4 = self.decode4(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1(decoder_output2, skip_encoder_1, indices_1)

        self.interpol2.target_shape = skip_encoder_0.shape[2:]
        decoder_output0, _ = self.interpol2(decoder_output1, scale_factor, rescale=True)

        out_block = self.outp_block(decoder_output0, skip_encoder_0)
        logits = self.classifier(out_block)

        return logits
