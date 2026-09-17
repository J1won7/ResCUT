# ResCUT

This repository is the official implementation of the paper "ResCUT: Domain-Preserving Restoration of Pelvic CBCT," which is currently under review.

## Checkpoint

The pretrained ResCUT generator is available [here](https://drive.google.com/file/d/1nWiBch30e6CgPOhLjVHiMaGLIWGME89v/view?usp=drive_link).

## Training

```bash
python prepare_and_train.py \
  --input-dir /path/to/cbct_nifti \
  --label-dir /path/to/ct_nifti
```

## Inference

```bash
python infer.py \
  --input-dir /path/to/cbct_nifti \
  --checkpoint /path/to/rescut_generator.pth \
  --output-dir /path/to/output
```
