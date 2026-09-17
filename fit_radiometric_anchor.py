"""Fit a fixed radiometric anchor using the historical MICCAI train split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.medimg_miccai2d_dataset import _case_key, _load_case


QUANTILES = np.linspace(0.20, 0.80, 13, dtype=np.float64)


def fit_affine(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    design = np.stack((source, np.ones_like(source)), axis=1)
    weights = np.ones_like(source)
    params = np.array((1.0, 0.0), dtype=np.float64)
    for _ in range(8):
        params = np.linalg.lstsq(design * np.sqrt(weights[:, None]), target * np.sqrt(weights), rcond=None)[0]
        residual = design @ params - target
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-8
        normalized = np.abs(residual) / (1.345 * scale)
        weights = np.minimum(1.0, 1.0 / np.maximum(normalized, 1e-8))
    affine_scale = float(np.clip(params[0], 0.25, 4.0))
    return affine_scale, float(np.mean(target - affine_scale * source))


def load_volume(folder: Path, available: set[str], fallback: Path | None, identifier: str, domain: str) -> np.ndarray:
    if identifier in available:
        return _load_case(str(folder), identifier)[0] * 2.0 - 1.0
    if fallback is None:
        raise FileNotFoundError(f"Preprocessed case is missing: {identifier}")
    split_folder = "trainCBCT" if domain == "source" else "trainDCT"
    raw = np.load(fallback / split_folder / f"{_case_key(identifier)}.npy").astype(np.float32, copy=False)
    return np.clip(raw, -1000.0, 1000.0).transpose(2, 1, 0) / 1000.0


def curve(folder: Path, identifiers: list[str], available: set[str], fallback: Path | None, domain: str) -> np.ndarray:
    values = []
    for index, identifier in enumerate(identifiers, start=1):
        volume = load_volume(folder, available, fallback, identifier, domain)
        foreground = volume[volume > -0.99]
        if foreground.size < 1024:
            raise ValueError(f"Too few foreground voxels: {identifier}")
        values.append(np.quantile(foreground, QUANTILES))
        if index == 1 or index % 50 == 0 or index == len(identifiers):
            print(f"[anchor] {domain} {index}/{len(identifiers)}", flush=True)
    return np.median(np.stack(values), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--fallback-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.dataroot)
    manifest = json.loads((root / "preprocessing_manifest.json").read_text())
    split = json.loads(Path(args.split_file).read_text())
    source = curve(root / manifest["domains"]["a"]["folder"], split["domains"]["a"]["train"], set(manifest["domains"]["a"]["identifiers"]), Path(args.fallback_root), "source")
    target = curve(root / manifest["domains"]["b"]["folder"], split["domains"]["b"]["train"], set(manifest["domains"]["b"]["identifiers"]), Path(args.fallback_root), "target")
    reference = 0.5 * (source + target)
    source_scale, source_shift = fit_affine(source, reference)
    target_scale, target_shift = fit_affine(target, reference)
    payload = {
        "format": "rescut_radiometric_anchor_v1",
        "quantiles": QUANTILES.tolist(),
        "source_curve": source.tolist(),
        "target_curve": target.tolist(),
        "reference_curve": reference.tolist(),
        "affine": {
            "source": {"scale": source_scale, "shift": source_shift},
            "target": {"scale": target_scale, "shift": target_shift},
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[anchor] saved {output}")
    print(f"[anchor] source=({source_scale:.6f}, {source_shift:+.6f}) target=({target_scale:.6f}, {target_shift:+.6f})")


if __name__ == "__main__":
    main()
