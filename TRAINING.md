# Training and validation

The model trained on **7,702 source MRI volumes** with **1,711 held-out validation volumes**. Modalities count separately when the same participant has more than one scan.

| Dataset | Modalities | Training volumes |
| --- | --- | ---: |
| openBHB | T1w | 3,227 |
| ADNI3 | FLAIR | 1,553 |
| ADNI4 | FLAIR | 138 |
| HCPA | T1w, T2w | 812 |
| HCPD | T1w, T2w | 726 |
| HCPY | T1w, T2w | 1,246 |
| **Total** | | **7,702** |

The archived loader reoriented volumes to RAS, applied 0.5th–99.5th percentile intensity clipping and nonzero-voxel z-score normalization, and sampled axial slices from a center-cropped or padded 192³ grid. The loader disabled MONAI spatial metadata before its nominal 1 mm spacing transform. The archived cache-build and training logs show MONAI falling back to an identity affine, which leaves native sampling unchanged in the reproduced transform. Source header spot checks showed 0.7 mm, 0.8 mm, and 1 mm isotropic images across the training datasets. The archived trainer used VINN's default scale factor of 1.0. Upstream preprocessing varied by dataset. Training and validation used FastSurfer-derived labels.

Training mixed real axial slices with synthetic slices generated on the fly. The synthesis transform used the label map to assign voxel intensities drawn from per-label Gaussian distributions, using statistics from the paired source MRI with random perturbations. The training configuration sets an 80% probability of retaining the real slice and a 20% probability of synthesizing it. The 7,702 count refers to the source volumes listed above. [SynthSeg method](https://doi.org/10.1016/j.media.2023.102789) · [Axial training loader](fastsurfer_vinn/data/dataset2d.py)

The table reports the peak validation result from the archived trainer metrics.

| Validation metric | Dice |
| --- | ---: |
| Macro-average across all 35 classes | 0.9002 |
| Macro-average across 34 foreground classes | 0.8973 |
| Cerebral white matter, mean of left and right Dice | 0.9398 |
| Cortical gray matter, mean of left and right Dice | 0.8986 |

The validation figures use per-class voxel counts accumulated over randomly sampled axial slices in the validation volumes. The slice loader samples Z indices 30–150 after volume preprocessing.

See the [training config](fastsurfer_vinn/configs/vinn.yaml) and [preprocessing code](fastsurfer_vinn/data/dataset.py).

## Train the model

Set `data_root`, `train_datasets`, and `val_datasets` in the [training config](fastsurfer_vinn/configs/vinn.yaml). For T1w, put each aligned scan and label pair at `<data_root>/<dataset>/T1w_split/train/subject_T1w.nii.gz` and `<data_root>/<dataset>/fastsurfer_seg/subject_T1w_seg.nii.gz`. The [dataset discovery code](fastsurfer_vinn/data/discovery.py) defines the validation, T2w, and FLAIR naming rules.

```bash
python -m fastsurfer_vinn.train --config fastsurfer_vinn/configs/vinn.yaml
```

## Use the segmentation consistency loss

`VINNSegLoss` adds anatomical supervision to MRI super-resolution, image reconstruction, and cross-modal synthesis. Its default soft KL mode matches the generated image's class probabilities to the frozen VINN probabilities on the aligned reference MRI.

- Prepare both images before calling the loss. Use the same aligned brain mask on the generated image and target, set voxels outside it to zero, and extract axial tensors of shape `(B, 1, H, W)` on the same device and output grid.
- Apply nonzero-voxel z-score normalization per volume before extracting slices. To match the CLI's `--clip` workflow, clip each masked volume at its 0.5th–99.5th percentiles first. Keep the prediction connected to the training graph through any masking and normalization applied to it.
- Pass `scale_factor` explicitly using `1 mm / output_in_plane_spacing`. The loss receives tensors and uses the scale supplied by the caller. An output with 0.8 mm axial in-plane spacing uses 1.25, including when the Z spacing differs. The constructor's default of 2.0 corresponds to 0.5 mm. Use one spacing per batch and pass the scale at each call when the spacing varies across batches.

```python
from fastsurfer_vinn.segmentation_loss import VINNSegLoss

criterion = VINNSegLoss("checkpoints/best.pth", scale_factor=1.0 / in_plane_spacing_mm, mode="soft")
loss_seg = criterion(pred_axial, ref_axial)
loss = image_loss + lambda_seg * loss_seg
```

Soft KL sums over all 35 classes and averages over the batch and spatial positions. The probability calculation uses FP32 under mixed precision. Set `mode="hard"` to use the reference argmax with the original Dice + cross-entropy loss. VINN remains in evaluation mode during training. The reference pass runs without gradients, and backpropagation reaches the generator through its predicted image.

For evaluation, use the same soft objective under `torch.no_grad()` to report agreement in VINN's class distributions. Use independent reference labels to measure anatomical accuracy.

Run `python -m unittest discover -s tests -v` after downloading `checkpoints/best.pth` with Git LFS. The tests cover gradient flow, the frozen VINN state, and near-zero soft KL loss and gradient for identical inputs.
