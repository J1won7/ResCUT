"""MICCAI-compatible 2D adapter for cached ResCUT NIfTI volumes.

The historical ResCUT checkpoint was trained from NPY volumes stored as
``[X, Y, Z]``.  Current medimg cases store the identical voxels as
``[C, Z, Y, X]``.  This loader restores the old 2D orientation by transposing
each axial slice to ``[C, X, Y]`` after the usual [0, 1] -> [-1, 1] adapter.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from data.base_dataset import BaseDataset


def _load_case(folder: str, identifier: str) -> np.ndarray:
    from util.nifti_preprocessing import load_preprocessed_case
    case = load_preprocessed_case(folder, identifier)
    image = np.asarray(case["image"], dtype=np.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected [C,D,Y,X], got {image.shape} for {identifier}")
    return image


def _case_key(identifier: str) -> str:
    return identifier.replace("_CBCT_", "_").replace("_dCT_", "_")


class MedimgMiccai2dDataset(BaseDataset):
    """Use the historical 466/40 MICCAI split with current medimg storage."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--medimg_split_file", required=True, help="MICCAI split JSON created by build_miccai_medimg_split.py")
        parser.add_argument(
            "--medimg_missing_npy_root",
            default="",
            help="Optional historical NPY root used only for cases absent from the medimg root.",
        )
        parser.add_argument("--medimg_slice_axis", type=int, default=0, choices=[0], help="The historical model uses axial Z slices only.")
        parser.add_argument(
            "--medimg_slices_per_volume",
            type=int,
            default=1,
            help="Independent random axial slices drawn from each volume in one epoch.",
        )
        parser.add_argument("--medimg_volume_depth", type=int, default=80, help="Number of stored axial slices per volume.")
        parser.add_argument("--medimg_cache_size", type=int, default=4, help="Per-worker volume cache size.")
        parser.add_argument("--medimg_random_target_seed", type=int, default=42)
        parser.set_defaults(
            preprocess="none",
            no_flip=True,
            serial_batches=True,
            input_nc=1,
            output_nc=1,
            netG="resnet_9blocks",
            netD="basic",
            nce_layers="0,4,8,12,16",
        )
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        root = Path(self.root)
        manifest_path = root / "preprocessing_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing medimg manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("dataset_kind") != "unpaired_domains":
            raise ValueError("medimg_miccai2d requires an unpaired_domains medimg root")

        split_payload = json.loads(Path(opt.medimg_split_file).read_text(encoding="utf-8"))
        split_name = "train" if opt.phase == "train" else "test"
        try:
            self.identifiers_a = tuple(split_payload["domains"]["a"][split_name])
            self.identifiers_b = tuple(split_payload["domains"]["b"][split_name])
        except KeyError as error:
            raise KeyError(f"Split file lacks domains.a/b.{split_name}") from error
        if not self.identifiers_a or not self.identifiers_b:
            raise ValueError(f"MICCAI {split_name} split is empty")

        domains = self.manifest["domains"]
        self.folder_a = str(root / domains["a"]["folder"])
        self.folder_b = str(root / domains["b"]["folder"])
        self.available_a = set(domains["a"].get("identifiers", []))
        self.available_b = set(domains["b"].get("identifiers", []))
        self.npy_root = Path(opt.medimg_missing_npy_root) if opt.medimg_missing_npy_root else None
        self.cache: OrderedDict[tuple[str, str, str], torch.Tensor] = OrderedDict()
        self.slices_per_volume = int(opt.medimg_slices_per_volume)
        self.volume_depth = int(opt.medimg_volume_depth)
        if self.slices_per_volume <= 0 or self.slices_per_volume > self.volume_depth:
            raise ValueError("--medimg_slices_per_volume must be positive")
        self._validate_availability(split_name)

        print(
            f"[{opt.phase.upper()}] MedimgMiccai2dDataset initialized. split={split_name} "
            f"volumes_a={len(self.identifiers_a)} volumes_b={len(self.identifiers_b)} "
            f"sampled_slices_per_volume={self.slices_per_volume} "
            f"axial_depth={self.volume_depth} "
            f"medimg_a={sum(item in self.available_a for item in self.identifiers_a)} "
            f"medimg_b={sum(item in self.available_b for item in self.identifiers_b)}",
            flush=True,
        )

    def _validate_availability(self, split_name: str) -> None:
        for domain, identifiers, available in (
            ("a", self.identifiers_a, self.available_a),
            ("b", self.identifiers_b, self.available_b),
        ):
            missing = [identifier for identifier in identifiers if identifier not in available]
            if not missing:
                continue
            if self.npy_root is None:
                raise FileNotFoundError(
                    f"{len(missing)} MICCAI {split_name} domain-{domain} cases are absent from medimg; "
                    "provide --medimg_missing_npy_root. First: " + missing[0]
                )
            for identifier in missing:
                key = _case_key(identifier)
                folder = "trainCBCT" if domain == "a" else "trainDCT"
                if split_name == "test":
                    folder = "testCBCT" if domain == "a" else "testDCT"
                path = self.npy_root / folder / f"{key}.npy"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing NPY fallback for {identifier}: {path}")

    def _epoch_permutation(self, count: int, salt: int) -> np.ndarray:
        epoch = int(getattr(self, "current_epoch", 0))
        seed = int(self.opt.medimg_random_target_seed) + 1_000_003 * epoch + salt
        return np.random.default_rng(seed).permutation(count)

    def _load_npy_fallback(self, identifier: str, domain: str) -> torch.Tensor:
        if self.npy_root is None:
            raise RuntimeError("NPY fallback was requested without --medimg_missing_npy_root")
        split_name = "train" if self.opt.phase == "train" else "test"
        folder = f"{split_name}{'CBCT' if domain == 'a' else 'DCT'}"
        raw = np.load(self.npy_root / folder / f"{_case_key(identifier)}.npy").astype(np.float32, copy=False)
        if raw.ndim != 3:
            raise ValueError(f"Expected [X,Y,Z] NPY array, got {raw.shape} for {identifier}")
        normalized = (np.clip(raw, -1000.0, 1000.0) + 1000.0) / 2000.0
        # [X,Y,Z] -> [C,Z,Y,X], exactly matching the medimg orientation.
        return torch.from_numpy(normalized.transpose(2, 1, 0)[None].copy())

    def _load_volume(self, identifier: str, domain: str) -> torch.Tensor:
        available = self.available_a if domain == "a" else self.available_b
        folder = self.folder_a if domain == "a" else self.folder_b
        source = "medimg" if identifier in available else "npy"
        cache_key = (source, domain, identifier)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache.move_to_end(cache_key)
            return cached
        if source == "medimg":
            volume = torch.from_numpy(_load_case(folder, identifier)).float()
        else:
            volume = self._load_npy_fallback(identifier, domain)
        if volume.shape[0] != 1 or volume.ndim != 4:
            raise ValueError(f"Expected [1,D,Y,X], got {tuple(volume.shape)} for {identifier}")
        self.cache[cache_key] = volume
        while len(self.cache) > int(self.opt.medimg_cache_size):
            self.cache.popitem(last=False)
        return volume

    @staticmethod
    def _to_cut_range(volume: torch.Tensor) -> torch.Tensor:
        # Current medimg values are exactly the old NPY HU clip mapped to [0,1].
        return volume.clamp(0.0, 1.0).mul(2.0).sub(1.0)

    def _slice(self, volume: torch.Tensor, index: int) -> torch.Tensor:
        # [C,Y,X] -> [C,X,Y] matches the old NPY loader's in-plane orientation.
        return self._to_cut_range(volume[:, index]).transpose(-1, -2).contiguous()

    def _random_slice_index(self, volume_index: int, sample_slot: int, salt: int, depth: int) -> int:
        epoch = int(getattr(self, "current_epoch", 0))
        seed = int(self.opt.medimg_random_target_seed) + 10_000_019 * epoch + 1_009 * volume_index + salt
        return int(np.random.default_rng(seed).permutation(depth)[sample_slot % depth])

    def __getitem__(self, index: int):
        volume_index = int(index) // self.slices_per_volume
        sample_slot = int(index) % self.slices_per_volume
        count_a = len(self.identifiers_a)
        count_b = len(self.identifiers_b)
        source_index = int(self._epoch_permutation(count_a, 17)[volume_index % count_a])
        target_index = int(self._epoch_permutation(count_b, 31)[volume_index % count_b])
        id_a = self.identifiers_a[source_index]
        id_b = self.identifiers_b[target_index]
        volume_a = self._load_volume(id_a, "a")
        volume_b = self._load_volume(id_b, "b")
        slice_a = self._random_slice_index(source_index, sample_slot, 43, volume_a.shape[1])
        slice_b = self._random_slice_index(target_index, sample_slot, 59, volume_b.shape[1])
        image_a = self._slice(volume_a, slice_a)
        image_b = self._slice(volume_b, slice_b)
        return {
            "A": image_a,
            "B": image_b,
            "A_paths": f"{id_a}::slice{slice_a}",
            "B_paths": f"{id_b}::slice{slice_b}",
        }

    def __len__(self):
        return max(len(self.identifiers_a), len(self.identifiers_b)) * self.slices_per_volume
