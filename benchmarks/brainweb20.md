# BrainWeb-20 evaluation

| Model | Tissue | Mean Dice | Sample SD | Median | Min–max |
| --- | --- | ---: | ---: | ---: | ---: |
| Axial VINN | WM | 0.9141 | 0.0070 | 0.9154 | 0.8988–0.9260 |
| Axial VINN | GM | 0.9253 | 0.0087 | 0.9268 | 0.9013–0.9392 |
| SynthSeg 2.0 | WM | 0.9084 | 0.0104 | 0.9082 | 0.8905–0.9238 |
| SynthSeg 2.0 | GM | 0.9086 | 0.0098 | 0.9127 | 0.8820–0.9223 |

## Evaluation protocol

All 20 precomputed 1 mm isotropic simulated T1-weighted volumes in the BrainWeb normal-anatomy cohort were evaluated.
The 0.5 mm discrete tissue model for each phantom was resampled to its T1w grid with nearest-neighbor interpolation.
SynthStrip v1 generated the brain mask from the T1w volume. The axial VINN then used the EMA weights of the default model, 0.5th–99.5th percentile intensity clipping, and nonzero-voxel z-score normalization. Inference ran on an RTX 4060 Ti with a batch of 8 axial slices.
SynthSeg 2.0 (FreeSurfer 8.1.0) used the same 20 skull-stripped T1w images with its default segmentation and intensity preprocessing. SynthSeg outputs were aligned to the input T1w grid with nearest-neighbor interpolation when needed.
Dice was calculated over the entire image grid per phantom. BrainWeb tissue IDs 3 and 2 define the WM and GM reference masks, respectively.
Predicted WM comprises FreeSurfer aseg IDs 2, 7, 41, and 46. Predicted GM comprises 3, 8, 10–13, 17, 18, 26, 42, 47, 49–54, and 58.

[BrainWeb data and tissue definitions](https://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html) · [Phantom paper](https://doi.org/10.1109/TMI.2006.883453) · [SynthStrip](https://doi.org/10.1016/j.neuroimage.2022.119474) · [SynthSeg](https://github.com/BBillot/SynthSeg)

## Reproduce the benchmark

From the repository root, install the project dependencies, fetch the checkpoint with Git LFS, and make the official SynthStrip v1 command available on your path.

```bash
python -m benchmarks.evaluate_brainweb
```

The evaluator downloads the source volumes from BrainWeb and writes the per-phantom Dice and voxel counts to `outputs/brainweb-eval/dice-per-case.csv`. Use `--subject 4` to run a single phantom, `--device cuda` to select CUDA for VINN, or `--strip-device cuda` to run SynthStrip on CUDA as well. Use `--synthstrip` and `--synthstrip-model` with a standalone SynthStrip script and weights.

With FreeSurfer 8.1.0 configured, pass the brain images from that run to SynthSeg and score its outputs with the same evaluator.

```bash
run_dir="$PWD/outputs/brainweb-synthseg"
mkdir -p "$run_dir/predictions"
printf '%s\n' "$PWD"/outputs/brainweb-eval/subject-*/t1w-brain.nii.gz > "$run_dir/inputs.txt"
while IFS= read -r image; do
    subject="$(basename "$(dirname "$image")")"
    printf '%s\n' "$run_dir/predictions/$subject.nii.gz"
done < "$run_dir/inputs.txt" > "$run_dir/outputs.txt"
mri_synthseg --i "$run_dir/inputs.txt" --o "$run_dir/outputs.txt" --cpu --threads 6
python -m benchmarks.evaluate_brainweb --cache-dir outputs/brainweb-eval \
    --predictions-dir "$run_dir/predictions" --output-dir "$run_dir"
```

## Per-phantom Dice

| Phantom | VINN WM | VINN GM | SynthSeg WM | SynthSeg GM |
| --- | ---: | ---: | ---: | ---: |
| 04 | 0.9093 | 0.9204 | 0.9028 | 0.9044 |
| 05 | 0.9110 | 0.9278 | 0.9023 | 0.9094 |
| 06 | 0.9151 | 0.9223 | 0.9002 | 0.8971 |
| 18 | 0.9199 | 0.9333 | 0.9238 | 0.9174 |
| 20 | 0.9196 | 0.9392 | 0.9170 | 0.9223 |
| 38 | 0.9082 | 0.9252 | 0.9058 | 0.9142 |
| 41 | 0.9260 | 0.9301 | 0.9220 | 0.9145 |
| 42 | 0.9166 | 0.9345 | 0.9131 | 0.9177 |
| 43 | 0.9212 | 0.9220 | 0.9233 | 0.8959 |
| 44 | 0.9041 | 0.9271 | 0.8969 | 0.9131 |
| 45 | 0.9199 | 0.9353 | 0.9166 | 0.9164 |
| 46 | 0.9076 | 0.9240 | 0.9017 | 0.9139 |
| 47 | 0.9205 | 0.9234 | 0.9180 | 0.9143 |
| 48 | 0.8988 | 0.9128 | 0.8908 | 0.9015 |
| 49 | 0.9058 | 0.9013 | 0.8905 | 0.8820 |
| 50 | 0.9168 | 0.9141 | 0.9078 | 0.8968 |
| 51 | 0.9221 | 0.9276 | 0.9150 | 0.9086 |
| 52 | 0.9106 | 0.9265 | 0.8973 | 0.9058 |
| 53 | 0.9157 | 0.9304 | 0.9147 | 0.9152 |
| 54 | 0.9135 | 0.9293 | 0.9087 | 0.9123 |
