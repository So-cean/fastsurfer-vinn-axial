"""NPU environment setup - module-level side effect on import.

Usage::

    import fastsurfer_vinn.utils.npu_setup  # before any ``import torch``

Configurable via environment variables:
    NPU_JIT_COMPILE=1    enable JIT operator fusion (default: 0)
    NPU_INTERNAL_FORMAT=1  allow NPU internal memory format (default: 0)
"""

import os
import shutil
import importlib.util


def _setup_npu():
    """Configure torch_npu - called once at module import."""
    if not (shutil.which("npu-smi") and importlib.util.find_spec("torch_npu")):
        return
    import torch_npu  # noqa: F401 - registers NPU backend
    from torch_npu.contrib import transfer_to_npu
    import torch as _torch

    jit = os.environ.get("NPU_JIT_COMPILE", "0") in ("1", "true", "True")
    internal = os.environ.get("NPU_INTERNAL_FORMAT", "0") in ("1", "true", "True")

    _torch.npu.set_compile_mode(jit_compile=jit)
    _torch.npu.config.allow_internal_format = internal


_setup_npu()
