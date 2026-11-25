# ResCUT

This repository is the official implementation of the paper "ResCUT: Domain-Preserving Restoration of Pelvic CBCT"
## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python -m model.prepare \
  --input-dir /path/to/cbct_nifti \
  --label-dir /path/to/ct_nifti \
  --output prepared_data
python -m model.train --dataroot prepared_data --name rescut
```

## Inference

```bash
python -m model.infer \
  --checkpoint /path/to/rescut_generator.pth \
  --input /path/to/cbct_nifti \
  --output /path/to/output
```

## Evaluation

SynthRAD2023 masks and ROI coordinates are included in `annotations/synthrad2023/`.

```bash
python -m evaluation.prepare_synthrad \
  --native /path/to/pelvis --output images
```

Run inference on `images/`, then add the prediction paths to the `images` fields in `annotations/synthrad2023/manifest.json`. Paths are relative to the manifest.

```bash
python -m evaluation.evaluate \
  --manifest annotations/synthrad2023/manifest.json \
  --output results
```
