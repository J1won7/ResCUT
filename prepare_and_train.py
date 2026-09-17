"""Prepare raw NIfTI domains and train ResCUT with one command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from util.nifti_preprocessing import prepare_unpaired_dataset


def _run(command: list[str]) -> None:
    print("[ResCUT] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _write_split(preprocessed_dir: Path, split_file: Path) -> None:
    manifest = json.loads((preprocessed_dir / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    identifiers_a = list(manifest["domains"]["a"]["identifiers"])
    identifiers_b = list(manifest["domains"]["b"]["identifiers"])
    payload = {"domains": {
        "a": {"train": identifiers_a, "test": []},
        "b": {"train": identifiers_b, "test": []},
    }}
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[ResCUT] training split: {len(identifiers_a)} input, {len(identifiers_b)} label volumes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory containing input CBCT NIfTI files")
    parser.add_argument("--label-dir", required=True, help="Directory containing unpaired target CT NIfTI files")
    parser.add_argument("--work-dir", default="./rescut_workspace", help="Reusable preprocessing and anchor directory")
    parser.add_argument("--name", default="rescut", help="Training experiment name")
    parser.add_argument("--checkpoints-dir", default="./checkpoints")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--num-workers", type=int, default=4)
    args, training_args = parser.parse_known_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    label_dir = Path(args.label_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    preprocessed_dir = work_dir / "preprocessed"
    split_file = work_dir / "train_split.json"
    anchor_file = work_dir / "radiometric_anchor.json"

    prepare_unpaired_dataset(input_dir, label_dir, preprocessed_dir)
    _write_split(preprocessed_dir, split_file)
    if anchor_file.is_file() and anchor_file.stat().st_mtime_ns >= (preprocessed_dir / "preprocessing_manifest.json").stat().st_mtime_ns:
        print(f"[ResCUT] reusing radiometric anchor: {anchor_file}", flush=True)
    else:
        _run([
            sys.executable, str(Path(__file__).with_name("fit_radiometric_anchor.py")),
            "--dataroot", str(preprocessed_dir), "--split-file", str(split_file),
            "--fallback-root", str(preprocessed_dir), "--output", str(anchor_file),
        ])

    command = [
        sys.executable, str(Path(__file__).with_name("train.py")),
        "--dataroot", str(preprocessed_dir),
        "--dataset_mode", "medimg_miccai2d",
        "--medimg_split_file", str(split_file),
        "--name", args.name, "--checkpoints_dir", args.checkpoints_dir,
        "--gpu_ids", args.gpu_ids, "--num_threads", str(args.num_workers),
        "--input_nc", "1", "--output_nc", "1",
        "--use_residual_learning", "--use_monotonic_D", "--use_drc",
        "--radiometric_anchor_path", str(anchor_file),
        "--no_html", "--display_id", "-1",
    ]
    _run(command + training_args)


if __name__ == "__main__":
    main()
