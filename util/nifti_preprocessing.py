"""Small, cached NIfTI preprocessing layer used by ResCUT."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

MANIFEST_NAME = "preprocessing_manifest.json"
SUPPORTED_SUFFIXES = (".nii.gz", ".nii")


def nifti_identifier(path: Path) -> str:
    name = path.name
    for suffix in SUPPORTED_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Unsupported NIfTI filename: {path}")


def scan_nifti(folder: str | Path) -> dict[str, Path]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"NIfTI directory does not exist: {root}")
    files: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or not path.name.endswith(SUPPORTED_SUFFIXES):
            continue
        identifier = nifti_identifier(path)
        if identifier in files:
            raise ValueError(f"Duplicate NIfTI identifier '{identifier}' in {root}")
        files[identifier] = path
    if not files:
        raise ValueError(f"No .nii or .nii.gz files found in {root}")
    return files


def _source_record(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _preprocess_volume(source: Path, output: Path) -> None:
    image = nib.load(str(source))
    values_xyz = np.asarray(image.dataobj, dtype=np.float32)
    if values_xyz.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, got {values_xyz.shape}: {source}")
    if not np.isfinite(values_xyz).all():
        raise ValueError(f"NIfTI contains NaN or infinite values: {source}")
    normalized_xyz = (np.clip(values_xyz, -1000.0, 1000.0) + 1000.0) / 2000.0
    image_czyx = normalized_xyz.transpose(2, 1, 0)[None].astype(np.float32, copy=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, image=image_czyx)


def _prepare_domain(raw_dir: Path, cache_dir: Path, previous: dict | None) -> dict:
    sources = scan_nifti(raw_dir)
    previous_sources = (previous or {}).get("sources", {})
    current_sources: dict[str, dict] = {}
    for index, (identifier, source) in enumerate(sources.items(), start=1):
        record = _source_record(source)
        output = cache_dir / f"{identifier}.npz"
        if previous_sources.get(identifier) != record or not output.is_file():
            _preprocess_volume(source, output)
            print(f"[preprocess] {index}/{len(sources)} {source.name}", flush=True)
        current_sources[identifier] = record
    return {
        "folder": cache_dir.name,
        "identifiers": list(sources),
        "sources": current_sources,
    }


def prepare_unpaired_dataset(
    input_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    previous = None
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("format") != "rescut_nifti_cache_v1":
            previous = None
    root.mkdir(parents=True, exist_ok=True)
    domains = {
        "a": _prepare_domain(Path(input_dir), root / "input", (previous or {}).get("domains", {}).get("a")),
        "b": _prepare_domain(Path(label_dir), root / "label", (previous or {}).get("domains", {}).get("b")),
    }
    manifest = {
        "format": "rescut_nifti_cache_v1",
        "dataset_kind": "unpaired_domains",
        "normalization": {"clip_hu": [-1000.0, 1000.0], "output_range": [0.0, 1.0]},
        "domains": domains,
    }
    serialized = json.dumps(manifest, indent=2) + "\n"
    if not manifest_path.is_file() or manifest_path.read_text(encoding="utf-8") != serialized:
        manifest_path.write_text(serialized, encoding="utf-8")
    return manifest_path


def prepare_input_dataset(input_dir: str | Path, output_dir: str | Path) -> Path:
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    previous = None
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("format") != "rescut_nifti_cache_v1":
            previous = None
    root.mkdir(parents=True, exist_ok=True)
    domain = _prepare_domain(
        Path(input_dir), root / "input", (previous or {}).get("domains", {}).get("a")
    )
    manifest = {
        "format": "rescut_nifti_cache_v1",
        "dataset_kind": "single_domain",
        "normalization": {"clip_hu": [-1000.0, 1000.0], "output_range": [0.0, 1.0]},
        "domains": {"a": domain},
    }
    serialized = json.dumps(manifest, indent=2) + "\n"
    if not manifest_path.is_file() or manifest_path.read_text(encoding="utf-8") != serialized:
        manifest_path.write_text(serialized, encoding="utf-8")
    return manifest_path


def load_preprocessed_case(folder: str | Path, identifier: str) -> dict[str, np.ndarray]:
    path = Path(folder) / f"{identifier}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached ResCUT volume: {path}")
    with np.load(path, allow_pickle=False) as payload:
        image = np.asarray(payload["image"], dtype=np.float32)
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected cached [1,Z,Y,X], got {image.shape}: {path}")
    return {"image": image}
