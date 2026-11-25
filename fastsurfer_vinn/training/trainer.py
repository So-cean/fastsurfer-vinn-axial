# -*- coding: utf-8 -*-
import os
import time
import warnings

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from .base import BaseTrainer
from .visualizer import Visualizer
from fastsurfer_vinn.utils.ema import ModelEMA
from fastsurfer_vinn.utils.loss import dice_ce_loss, deep_supervision_loss, accumulate_dice_stats, dice_from_stats, compute_class_weights, compute_class_weights_from_counts, compute_voxel_weight
from fastsurfer_vinn.utils.labels import NUM_CLASSES, CLASS_NAMES

warnings.filterwarnings("ignore", message="data_array is not of type MetaTensor")


class Trainer(BaseTrainer):
    """3D patch-based segmentation trainer with EMA."""

    def __init__(
        self,
        cfg: dict,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        use_dist: bool = False,
        local_rank: int = 0,
    ):
        super().__init__(local_rank, use_dist)

        self.cfg = cfg
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_amp = cfg.get("mixed_precision") in ("bf16", "amp")
        self.amp_dtype = torch.bfloat16 if cfg.get("mixed_precision") == "bf16" else torch.float16
        self.ignore_index = cfg.get("ignore_index", 0)
        self.dice_weight = cfg.get("dice_weight", 0.5)
        self.ce_weight = cfg.get("ce_weight", 0.5)
        self._spatial_dims = cfg.get("spatial_dims", 3)

        # ── Class-balanced CE weights (1/sqrt freq, no background override) ──
        max_vols = cfg.get("class_weight_volumes", 100)
        if self.is_main:
            dataset = train_loader.dataset
            n = min(max_vols, len(dataset.cached))
            print(f"  Computing class weights from {n} volumes ...", flush=True)
            device = self.device
            counts = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
            for i in range(n):
                label = dataset.cached[i]["label"].squeeze(0).long().flatten().to(device)
                counts += torch.bincount(label, minlength=NUM_CLASSES)[:NUM_CLASSES].float()
                if (i + 1) % 1000 == 0:
                    print(f"    class weight progress: {i+1}/{n}", flush=True)
            self.class_weight = compute_class_weights_from_counts(counts, NUM_CLASSES).to(device)
            print(f"  Class weights computed.", flush=True)
        else:
            self.class_weight = None

        if use_dist:
            cw_t = torch.zeros(NUM_CLASSES, device=self.device)
            if self.is_main:
                cw_t.copy_(self.class_weight)
            torch.distributed.broadcast(cw_t, src=0)
            self.class_weight = cw_t

        self.build_optimizer(
            lr=cfg.get("lr", 1e-4),
            weight_decay=cfg.get("weight_decay", 1e-5),
        )
        self.build_scheduler(
            warmup_epochs=cfg.get("warmup_epochs", 5),
            total_epochs=cfg.get("epochs", 100),
        )

        # ── EMA (must init before DDP wrap) ──
        self.ema = None
        if cfg.get("use_ema", False):
            ema_decay = cfg.get("ema_decay", 0.999)
            self.ema = ModelEMA(self.model, decay=ema_decay)
            if self.is_main:
                print(f"  EMA enabled (decay={ema_decay})")

        if use_dist:
            self.wrap_ddp(find_unused_parameters=False)

        self.scaler = GradScaler(self.device_type, enabled=(cfg.get("mixed_precision") == "amp"))

        output_dir = cfg.get("output_dir", "runs/unet3d")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self._init_writer(os.path.join(output_dir, "logs"))

        self.epoch = 0
        self.best_dice = 0.0

        # ── Visualizer ──
        self.visualizer = None
        if val_loader is not None and cfg.get("val_visualize", True):
            self.visualizer = Visualizer(
                val_loader, self.device, output_dir, NUM_CLASSES,
            )
            if self.is_main:
                print("  Visualizer enabled")

        # ── Loss kwargs shared across all calls ──
        self._loss_kwargs = dict(
            ignore_index=self.ignore_index,
            dice_weight=self.dice_weight,
            ce_weight=self.ce_weight,
            class_weight=self.class_weight,
            spatial_dims=self.cfg.get("spatial_dims", 3),
        )
        self._use_voxel_weight = cfg.get("voxel_weight", False)
        self._edge_w = cfg.get("edge_w", 5.0)
        if self.is_main:
            print(f"  Trainer init done, starting training...", flush=True)

    def train(self, checkpoint=None, resume=False):
        """Run the full training loop."""
        if checkpoint:
            start_epoch, self.best_dice = self.load_checkpoint(checkpoint)
            self.epoch = start_epoch
            if not resume:
                return
            # Reset LR scheduler on resume (if configured)
            if self.cfg.get("reset_lr", False):
                self.epoch = 0
                warmup = self.cfg.get("warmup_epochs", 10)
                total = self.cfg.get("epochs", 100)
                self.build_scheduler(warmup, total)
        elif resume:
            ckpt = os.path.join(self.output_dir, "checkpoints", "latest.pth")
            if os.path.isfile(ckpt):
                self.epoch, self.best_dice = self.load_checkpoint(ckpt)

        epochs = self.cfg.get("epochs", 100)
        val_every = self.cfg.get("val_every", 5)
        save_every = self.cfg.get("save_every", 10)
        self.log_freq = self.cfg.get("display_freq", 10)
        early_stop = self.cfg.get("early_stop_patience", 0)
        early_stop_min = self.cfg.get("early_stop_min_epochs", 0)
        no_improve = 0
        t_start = time.time()

        for self.epoch in range(self.epoch, epochs):
            if self.use_dist and hasattr(self.train_loader, 'sampler'):
                self.train_loader.sampler.set_epoch(self.epoch)

            t_ep = time.time()
            train_metrics = self.train_epoch()
            elapsed = time.time() - t_start
            ep_time = time.time() - t_ep
            remaining = (epochs - self.epoch - 1) * ep_time
            eta = time.strftime("%H:%M", time.gmtime(remaining))

            val_metrics = None
            if self.val_loader and self.epoch % val_every == 0:
                val_metrics = self.validate()

            self.log_epoch(
                epoch=self.epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                class_names=CLASS_NAMES,
                lr=self.optimizer.param_groups[0]['lr'],
                best=self.best_dice,
            )
            if self.is_main:
                print(f"Epoch {self.epoch:3d}/{epochs} | "
                      f"loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} "
                      f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                      f"{ep_time:.0f}s | ETA {eta}")

            if self.epoch % save_every == 0 and self.is_main:
                self.save_checkpoint(
                    os.path.join(self.output_dir, "checkpoints", f"epoch_{self.epoch}.pth"),
                    self.epoch,
                )
                self.save_checkpoint(
                    os.path.join(self.output_dir, "checkpoints", "latest.pth"),
                    self.epoch,
                )

            if val_metrics:
                if self.is_main:
                    print(f"  VAL dice={val_metrics['dice']:.4f}")
                if self.visualizer:
                    self.visualizer(self.model, self.epoch, writer=self.writer)
                if val_metrics["dice"] > self.best_dice:
                    self.best_dice = val_metrics["dice"]
                    no_improve = 0
                    if self.is_main:
                        if self.ema is not None:
                            raw = self._raw_model()
                            raw_sd = {k: v.clone() for k, v in raw.state_dict().items()}
                            self.ema.copy_to(raw)
                        self.save_checkpoint(
                            os.path.join(self.output_dir, "checkpoints", "best.pth"),
                            self.epoch, best_dice=self.best_dice,
                        )
                        if self.ema is not None:
                            self._raw_model().load_state_dict(raw_sd, strict=False)
                else:
                    no_improve += 1
                    if early_stop and no_improve >= early_stop and self.epoch >= early_stop_min:
                        print(f"Early stop at epoch {self.epoch}")
                        break

        if self.is_main:
            if self.ema is not None:
                raw = self._raw_model()
                raw.load_state_dict(self.ema.state_dict(), strict=False)
            self.save_checkpoint(
                os.path.join(self.output_dir, "checkpoints", "latest.pth"),
                self.epoch,
            )

    def train_epoch(self) -> dict:
        self.model.train()
        n = 0
        n_total = len(self.train_loader)
        t_ep = time.time()
        inter = torch.zeros(NUM_CLASSES, device=self.device)
        unn   = torch.zeros(NUM_CLASSES, device=self.device)
        loss_sum = torch.tensor(0.0, device=self.device)

        if self.is_main:
            print(f"  Starting epoch (n_total={n_total})...", flush=True)
        for i, (t1, seg) in enumerate(self.train_loader):
            t1 = t1.to(self.device, non_blocking=True)
            t1 = torch.nan_to_num(t1, nan=0.0)
            seg = seg.to(self.device, non_blocking=True).long()

            with autocast(device_type=self.device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                model_out = self.model(t1)
                loss_kwargs = dict(self._loss_kwargs)
                if self._use_voxel_weight and self._spatial_dims == 2:
                    loss_kwargs["voxel_weight"] = compute_voxel_weight(
                        seg, edge_w=self._edge_w,
                    )
                loss, _main_logits, i_b, u_b = deep_supervision_loss(
                    model_out, seg, **loss_kwargs,
                )

            inter += i_b.detach()
            unn   += u_b.detach()

            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.cfg.get("max_grad_norm", 1.0))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.cfg.get("max_grad_norm", 1.0))
                self.optimizer.step()

            if self.ema is not None:
                self.ema.update(self._raw_model())

            loss_sum += loss.detach()
            n += 1
            n_cur = n

            if self.is_main and (n_cur % self.log_freq == 0 or n_cur == 1):
                dice_run = dice_from_stats(inter, unn, self.ignore_index)[1]
                speed = n_cur / (time.time() - t_ep)
                running_loss = loss_sum.item() / n_cur
                print(f"    batch {n_cur}/{n_total} | "
                      f"loss={running_loss:.4f} dice={dice_run:.4f} | "
                      f"{speed:.1f} batch/s", flush=True)

        if self.use_dist:
            inter, unn = self._reduce(inter, unn)
            total_loss = self._reduce(loss_sum / n).item()
        else:
            total_loss = (loss_sum / n).item()

        dice_per_cls, total_dice = dice_from_stats(inter, unn, self.ignore_index)

        self.scheduler.step()
        return {"loss": total_loss, "dice": total_dice,
                "dice_per_cls": dice_per_cls}

    @torch.no_grad()
    def validate(self) -> dict:
        use_ema = False
        if self.ema is not None:
            raw = self._raw_model()
            raw_sd = {k: v.clone() for k, v in raw.state_dict().items()}
            self.ema.copy_to(raw)
            use_ema = True

        self.model.eval()
        inter = torch.zeros(NUM_CLASSES, device=self.device)
        unn   = torch.zeros(NUM_CLASSES, device=self.device)

        for t1, seg in self.val_loader:
            t1 = t1.to(self.device, non_blocking=True)
            t1 = torch.nan_to_num(t1, nan=0.0)
            seg = seg.to(self.device, non_blocking=True).long()

            with autocast(device_type=self.device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                logits = self.model(t1)
                i_b, u_b = accumulate_dice_stats(logits, seg, self.ignore_index)
                inter += i_b
                unn   += u_b

        if self.use_dist:
            inter, unn = self._reduce(inter, unn)
        dice_per_cls, total_dice = dice_from_stats(inter, unn, self.ignore_index)

        if use_ema:
            self._raw_model().load_state_dict(raw_sd)

        return {"dice": total_dice, "dice_per_cls": dice_per_cls}
