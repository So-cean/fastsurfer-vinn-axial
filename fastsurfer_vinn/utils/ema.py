"""Exponential Moving Average for model parameters.

Stores a plain dict of parameter tensors (no buffers).  Uses
``named_parameters()`` for in-place update — no ``state_dict()``
allocation per batch.
"""

import torch
import torch.nn as nn


class ModelEMA:
    """Maintain an exponentially-decayed copy of model *parameters*.

    Usage::

        ema = ModelEMA(model, decay=0.999)
        for step in steps:
            loss.backward()
            optimizer.step()
            ema.update(model)
        ema.copy_to(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self._state = {name: param.data.detach().clone()
                       for name, param in model.named_parameters()}

    def update(self, model: nn.Module):
        d = self.decay
        with torch.no_grad():
            for name, param in model.named_parameters():
                self._state[name].mul_(d).add_(param.data, alpha=1.0 - d)

    def state_dict(self):
        return self._state

    def load_state_dict(self, state_dict):
        self._state = state_dict

    def copy_to(self, model: nn.Module):
        """Copy EMA parameters into *model* (skips buffers)."""
        model.load_state_dict(self._state, strict=False)
