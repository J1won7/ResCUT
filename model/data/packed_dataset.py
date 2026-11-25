"""Memory-mapped axial slices with independent domain permutations."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .base_dataset import BaseDataset


class PackedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--seed', type=int, default=20261024)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        root = Path(self.root)
        manifest_path, split_path = root / 'cache/manifest.json', root / 'split.json'
        manifest = json.loads(manifest_path.read_text())
        split = json.loads(split_path.read_text())
        if not opt.isTrain or opt.phase != 'train':
            raise ValueError('This adapter reads only the training split')
        assert manifest['array_order'] == 'Z,Y,X' and manifest['value_range'] == [-1, 1]
        digest = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
        assert manifest['split_sha256'] == digest
        self.fingerprint = hashlib.sha256(manifest_path.read_bytes() + split_path.read_bytes()).hexdigest()
        records = {r['patient']: r for r in manifest['cases']}
        self.records = [records[c] for c in split['splits']['train']]
        if not self.records:
            raise ValueError('Empty training split')
        self.records.sort(key=lambda r: r['packed_start'])
        self.stops = np.asarray([r['packed_stop'] for r in self.records])
        starts = [r['packed_start'] for r in self.records]
        assert starts == [0] + self.stops[:-1].tolist()
        self.length = int(self.stops[-1])
        assert sum(r['depth'] for r in self.records) == self.length
        self.arrays = {d: np.load(root / self.records[0]['packed_files'][d], mmap_mode='r')
                       for d in ('cbct', 'ct')}
        assert all(a.shape == (self.length, 256, 256) and a.dtype == np.float32 for a in self.arrays.values())
        self.use_views = opt.use_radiometric_views
        self.mask_arrays = {}
        self.plan_epoch = None
        self.target_plan = None
        subjects = {r['subject_id'] for r in self.records}
        heldout = {records[c]['subject_id'] for name, ids in split['splits'].items()
                   if name != 'train' for c in ids}
        if subjects & heldout:
            raise ValueError('Training and held-out subjects overlap')

    def __len__(self):
        return self.length

    def describe(self, index):
        record = self.records[int(np.searchsorted(self.stops, index, side='right'))]
        return record, int(index - record['packed_start'])

    def tensor(self, domain, index):
        image = np.array(self.arrays[domain][index:index+1], copy=True)
        if not np.isfinite(image).all() or image.min() < -1.00001 or image.max() > 1.00001:
            raise ValueError(('Invalid normalized slice', domain, index))
        return torch.from_numpy(image)

    def foreground(self, domain, index):
        record, z = self.describe(index)
        key = record['patient'], domain
        if key not in self.mask_arrays:
            path = Path(self.root) / 'calibration/foreground' / f'{key[0]}_{domain}.npy'
            array = np.load(path, mmap_mode='r')
            assert array.dtype == bool and array.shape == (record['depth'], 256, 256)
            self.mask_arrays[key] = array
        return torch.from_numpy(np.array(self.mask_arrays[key][z:z+1], copy=True))

    def __getitem__(self, index):
        if self.plan_epoch != self.current_epoch:
            self.plan_epoch = self.current_epoch
            self.target_plan = np.random.default_rng([self.opt.seed, self.current_epoch, 97]).permutation(self.length)
        target = int(self.target_plan[index])
        a, az = self.describe(index)
        b, bz = self.describe(target)
        result = dict(A=self.tensor('cbct', index), B=self.tensor('ct', target),
            A_paths=f'{a["patient"]}:{a["source_start_z"]+az}',
            B_paths=f'{b["patient"]}:{b["source_start_z"]+bz}')
        if self.use_views:
            result.update(A_foreground=self.foreground('cbct', index), B_foreground=self.foreground('ct', target))
        return result
