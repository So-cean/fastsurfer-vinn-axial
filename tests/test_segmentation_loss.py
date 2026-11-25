from contextlib import nullcontext
from pathlib import Path
import unittest

import torch

from fastsurfer_vinn.segmentation_loss import VINNSegLoss


CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "best.pth"


class SegmentationLossTest(unittest.TestCase):
    def test_rejects_misaligned_batch_sizes(self):
        criterion = VINNSegLoss(CHECKPOINT, scale_factor=1.0)
        with self.assertRaisesRegex(ValueError, "must have the same"):
            criterion(torch.zeros(2, 1, 16, 16), torch.zeros(1, 1, 16, 16))

    def test_frozen_loss_backpropagates_only_to_prediction(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(min(old_threads, 4))
        self.addCleanup(torch.set_num_threads, old_threads)
        configs = [("cpu", None, "soft", (96, 112), 1.0),
                   ("cpu", None, "hard", (96, 112), 1.0)]
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            configs.append(("cuda", torch.bfloat16, "soft", (192, 192), 2.0))

        for device, dtype, mode, shape, scale in configs:
            with self.subTest(device=device, dtype=dtype, mode=mode, shape=shape, scale=scale):
                criterion = VINNSegLoss(CHECKPOINT, device=torch.device(device), scale_factor=scale, mode=mode)
                parent = torch.nn.ModuleDict({"segmentation_loss": criterion}).train()
                self.assertTrue(parent.training)
                self.assertTrue(criterion.training)
                self.assertFalse(criterion.vinn.training)
                before = {name: buffer.clone() for name, buffer in criterion.vinn.named_buffers()}
                generator = torch.Generator(device=device).manual_seed(17)
                pred = torch.randn(1, 1, *shape, device=device, generator=generator, requires_grad=True)
                ref = torch.randn(1, 1, *shape, device=device, generator=generator, requires_grad=True)

                autocast = torch.autocast(device, dtype=dtype) if dtype else nullcontext()
                with autocast:
                    loss = criterion(pred, ref)
                loss.backward()

                self.assertTrue(torch.isfinite(loss).item())
                self.assertGreater(loss.item(), 0)
                self.assertIsNotNone(pred.grad)
                self.assertTrue(torch.isfinite(pred.grad).all().item())
                self.assertGreater(pred.grad.abs().sum().item(), 0)
                self.assertIsNone(ref.grad)
                for parameter in criterion.vinn.parameters():
                    self.assertFalse(parameter.requires_grad)
                    self.assertIsNone(parameter.grad)
                for name, buffer in criterion.vinn.named_buffers():
                    self.assertTrue(torch.equal(buffer, before[name]), name)

                if mode == "soft":
                    identical = ref.detach().clone().requires_grad_(True)
                    with autocast:
                        identity_loss = criterion(identical, ref)
                    identity_loss.backward()
                    self.assertLess(abs(identity_loss.item()), 1e-6)
                    self.assertLess(identical.grad.norm().item(), 1e-5)
