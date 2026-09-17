#!/usr/bin/env python3
"""Run ResCUT directly on a directory of NIfTI volumes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import networks  # noqa: E402
from util.nifti_preprocessing import (  # noqa: E402
    load_preprocessed_case,
    prepare_input_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--slice-batch-size", type=int, default=32)
    parser.add_argument("--clip-min-hu", type=float, default=-1000.0)
    parser.add_argument("--clip-max-hu", type=float, default=1000.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_generator(args: argparse.Namespace):
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint}")
    requested = [int(value) for value in args.gpu_ids.split(",") if int(value) >= 0]
    if requested and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
    gpu_ids = requested[:1]
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
    opt = SimpleNamespace(use_residual_learning=True)
    generator = networks.define_G(
        input_nc=1, output_nc=1, ngf=64, netG="resnet_9blocks",
        norm="instance", use_dropout=False, init_type="xavier", init_gain=0.02,
        no_antialias=False, no_antialias_up=False, gpu_ids=gpu_ids, opt=opt,
    )
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    generator.load_state_dict(state_dict, strict=True)
    generator.eval()
    return generator, device


def save_volume(path: Path, values_czyx: np.ndarray, reference: nib.spatialimages.SpatialImage) -> None:
    xyz = np.asarray(values_czyx[0].transpose(2, 1, 0), dtype=np.float32)
    if xyz.shape != reference.shape:
        raise ValueError(f"Output/reference shape mismatch: {xyz.shape} != {reference.shape}")
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(xyz, reference.affine, header), str(path))


def main() -> None:
    args = parse_args()
    if args.clip_max_hu <= args.clip_min_hu:
        raise ValueError("--clip-max-hu must be greater than --clip-min-hu")
    cache_dir = args.cache_dir or (args.output_dir / ".rescut_cache")
    manifest_path = prepare_input_dataset(args.input_dir, cache_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    domain = manifest["domains"]["a"]
    identifiers = list(domain["identifiers"])
    if args.max_cases:
        identifiers = identifiers[: args.max_cases]
    cache_folder = cache_dir / domain["folder"]
    generator, device = load_generator(args)
    records = []

    for number, identifier in enumerate(identifiers, start=1):
        output = args.output_dir / "fake_B" / f"{identifier}.nii.gz"
        if output.is_file() and not args.overwrite:
            print(f"[skip] {number}/{len(identifiers)} {identifier}", flush=True)
            records.append({"identifier": identifier, "status": "skipped_exists", "output": str(output.resolve())})
            continue
        source = load_preprocessed_case(cache_folder, identifier)["image"]
        reference_path = Path(domain["sources"][identifier]["path"])
        reference = nib.load(str(reference_path))
        source_cut = source.transpose(1, 0, 3, 2) * 2.0 - 1.0
        predictions: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, source_cut.shape[0], args.slice_batch_size):
                batch = torch.from_numpy(source_cut[start:start + args.slice_batch_size]).to(device)
                predictions.append((batch + generator(batch)).cpu().numpy())
        fake_czyx = np.concatenate(predictions, axis=0).transpose(1, 0, 3, 2)
        fake_hu = np.clip(fake_czyx * 1000.0, args.clip_min_hu, args.clip_max_hu)
        source_hu = (source * 2.0 - 1.0) * 1000.0
        save_volume(output, fake_hu, reference)
        save_volume(args.output_dir / "residual_B" / f"{identifier}.nii.gz", fake_hu - source_hu, reference)
        save_volume(args.output_dir / "real_A" / f"{identifier}.nii.gz", source_hu, reference)
        records.append({"identifier": identifier, "status": "written", "output": str(output.resolve())})
        print(f"[done] {number}/{len(identifiers)} {identifier}", flush=True)

    summary = {
        "format": "rescut_nifti_inference_v2",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "input_dir": str(args.input_dir.expanduser().resolve()),
        "device": str(device),
        "case_count": len(records),
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
