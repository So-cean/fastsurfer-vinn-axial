# -*- coding: utf-8 -*-
"""FastSurfer aseg label ↔ continuous [0..34] mapping.

Bijective mapping table (single source of truth) + pre-computed forward/reverse
lookup tables as ``torch.long`` tensors.  Cast to ``uint8`` at cache save.
"""

import torch

# ── Bijective mapping: FreeSurfer aseg → continuous ─────────────────────────
MAPPING = {
    0: 0,     # background
    2: 1,     # L-WM
    3: 2,     # L-Cortex
    4: 3,     # L-LatVent
    5: 4,     # L-InfLatVent
    7: 5,     # L-Cbl-WM
    8: 6,     # L-Cbl-Cortex
    10: 7,    # L-Thalamus
    11: 8,    # L-Caudate
    12: 9,    # L-Putamen
    13: 10,   # L-Pallidum
    14: 11,   # 3rd-Ventricle
    15: 12,   # 4th-Ventricle
    16: 13,   # BrainStem
    17: 14,   # L-Hippocampus
    18: 15,   # L-Amygdala
    24: 16,   # CSF
    26: 17,   # L-Accumbens
    28: 18,   # L-VentralDC
    31: 19,   # L-ChoroidPlexus
    41: 20,   # R-WM
    42: 21,   # R-Cortex
    43: 22,   # R-LatVent
    44: 23,   # R-InfLatVent
    46: 24,   # R-Cbl-WM
    47: 25,   # R-Cbl-Cortex
    49: 26,   # R-Thalamus
    50: 27,   # R-Caudate
    51: 28,   # R-Putamen
    52: 29,   # R-Pallidum
    53: 30,   # R-Hippocampus
    54: 31,   # R-Amygdala
    58: 32,   # R-Accumbens
    60: 33,   # R-VentralDC
    63: 34,   # R-ChoroidPlexus
}

NUM_CLASSES = 35

# ── Human-readable class names (continuous → name) ──────────────────────────
CLASS_NAMES = [
    "BG", "L_WM", "L_Cortex", "L_LatVent", "L_InfLatV",
    "L_Cbl_WM", "L_Cbl_Ctx", "L_Thal", "L_Caud", "L_Put",
    "L_Pall", "3rd_Vent", "4th_Vent", "BrainStem", "L_Hipp",
    "L_Amyg", "CSF", "L_Accumb", "L_VentralDC", "L_ChorPlex",
    "R_WM", "R_Cortex", "R_LatVent", "R_InfLatV", "R_Cbl_WM",
    "R_Cbl_Ctx", "R_Thal", "R_Caud", "R_Put", "R_Pall",
    "R_Hipp", "R_Amyg", "R_Accumb", "R_VentralDC", "R_ChorPlex",
]

# ── Forward LUT: FS label (0-255) → continuous [0..34] ──────────────────────
# Unknown labels → 0.  Vessels → CSF.  77 is handled by RemapLabelsd before LUT.
_FWD = torch.zeros(256, dtype=torch.long)
for fs, ct in MAPPING.items():
    _FWD[fs] = ct
_FWD[30] = 16   # L-Vessel → CSF
_FWD[62] = 16   # R-Vessel → CSF

# ── Reverse LUT: continuous [0..34] → FS label ──────────────────────────────
_REV = torch.zeros(NUM_CLASSES, dtype=torch.long)
for fs, ct in MAPPING.items():
    _REV[ct] = fs

# ── Public API ──────────────────────────────────────────────────────────────
L2C      = MAPPING
C2L      = {v: k for k, v in MAPPING.items()}
WM_ASEG_IDS = (2, 7, 41, 46)
GM_ASEG_IDS = (3, 8, 10, 11, 12, 13, 17, 18, 26, 42, 47, 49, 50, 51, 52, 53, 54, 58)
_LUT     = _FWD          # (256,) torch.long
_C2L_LUT = _REV          # (35,)  torch.long
