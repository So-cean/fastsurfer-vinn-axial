# -*- coding: utf-8 -*-
"""Minimal DDP trainer base — device, checkpoint, logging.

No training-loop logic; subclasses implement ``train_epoch``/``validate``.
"""

import os
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from fastsurfer_vinn.utils.ddp import get_dist_info, cleanup_dist, reduce_tensors


class BaseTrainer:
    def __init__(self, local_rank: int, use_dist: bool):
        self.use_dist = use_dist
        self.rank, self.world_size = get_dist_info() if use_dist else (0, 1)
        self.is_main = (self.rank == 0)

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{local_rank}" if use_dist else "cuda")
            self.device_type = "cuda"
        elif hasattr(torch, "npu") and torch.npu.is_available():
            self.device = torch.device(f"npu:{local_rank}" if use_dist else "npu")
            self.device_type = "npu"
        else:
            self.device = torch.device("cpu")
            self.device_type = "cpu"

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None

    # ── DDP communication ──────────────────────────────────────────

    def _reduce(self, *tensors):
        return reduce_tensors(self.use_dist, *tensors)

    def close(self):
        if self.writer:
            self.writer.close()
        if self.use_dist:
            cleanup_dist()

    # ── checkpoint I/O ─────────────────────────────────────────────

    def _raw_model(self):
        return self.model.module if self.use_dist else self.model

    def save_checkpoint(self, path, epoch, **extra):
        if not self.is_main:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "epoch": epoch,
            "model": self._raw_model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        if hasattr(self, "ema") and self.ema is not None:
            state["ema"] = self.ema.state_dict()
        state.update(extra)
        torch.save(state, path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        model_sd = {key.removeprefix("module."): value for key, value in state["model"].items()}
        self._raw_model().load_state_dict(model_sd)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if hasattr(self, "ema") and self.ema is not None and "ema" in state:
            self.ema.load_state_dict(state["ema"])
        return state.get("epoch", 0), state.get("best_dice", 0.0)

    # ── optimizer / scheduler factories ────────────────────────────

    def build_optimizer(self, lr, **kwargs):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, **kwargs)

    def build_scheduler(self, warmup_epochs, total_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(warmup_epochs, 1)
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    # ── DDP wrapper ────────────────────────────────────────────────

    def wrap_ddp(self, **kwargs):
        if self.use_dist:
            if self.device.type in ("cuda", "npu"):
                device_ids = [self.device.index]
            else:
                device_ids = None
            self.model = DDP(self.model, device_ids=device_ids, **kwargs)

    # ── TensorBoard + CSV logging ──────────────────────────────────

    def _init_writer(self, output_dir):
        if self.is_main:
            os.makedirs(output_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=output_dir)

    def log_epoch(self, epoch, train_metrics, val_metrics=None,
                  class_names=None, lr=None, best=0.0):
        """Log one epoch's metrics to TensorBoard + CSV.

        ``train_metrics``:  ``{"loss": float, "dice": float, "dice_per_cls": ndarray}``
        ``val_metrics``:    ``{"dice": float, "dice_per_cls": ndarray}`` or None
        """
        from csv import DictWriter

        if self.writer:
            self.writer.add_scalar("train/loss", train_metrics.get("loss", 0), epoch)
            self.writer.add_scalar("train/dice", train_metrics.get("dice", 0), epoch)
            if val_metrics:
                self.writer.add_scalar("val/dice", val_metrics.get("dice", 0), epoch)

        row = {"epoch": epoch,
               "loss": train_metrics.get("loss", 0),
               "dice": train_metrics.get("dice", 0)}
        per_cls = train_metrics.get("dice_per_cls", [])
        names = class_names or [f"cls_{i}" for i in range(len(per_cls))]
        for i in range(min(len(per_cls), len(names))):
            row[f"{i}_{names[i]}"] = float(per_cls[i])
        if val_metrics:
            row["val_dice"] = val_metrics.get("dice", 0)
            val_per_cls = val_metrics.get("dice_per_cls", [])
            for i in range(min(len(val_per_cls), len(names))):
                row[f"val_{i}_{names[i]}"] = float(val_per_cls[i])
        row["lr"] = lr
        row["best"] = best

        log_dir = self.writer.log_dir if self.writer else "."
        csv_path = os.path.join(log_dir, "metrics.csv")
        exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            w = DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)
