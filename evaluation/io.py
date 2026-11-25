"""NIfTI geometry checks and portable manifest I/O."""
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path, reference=None, mask=False):
    image = nib.load(str(path))
    values = image.get_fdata(dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f'Expected a finite 3D image: {path}')
    if reference is not None and (image.shape != reference.shape or
            not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4)):
        raise ValueError(f'Geometry mismatch: {path}; no implicit resampling is performed')
    return image, values > 0 if mask else np.clip(values, -1000, 1000)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def read_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    records = []
    seen = set()
    for original in data['cases']:
        record = dict(original)
        for name in (record['case'], record.get('dataset', 'dataset')):
            if not name or name in ('.', '..') or Path(name).name != name:
                raise ValueError('Case and dataset labels must be single path components')
        key = (record.get('dataset', 'dataset'), record['case'])
        if key in seen:
            raise ValueError(f'Duplicate case: {key}')
        seen.add(key)
        for field in ('source', 'artifact_mask', 'reference_mask', 'body_mask', 'rois'):
            if record.get(field):
                record[field] = str((path.parent / record[field]).resolve())
        record['images'] = {name: str((path.parent / p).resolve())
                            for name, p in record.get('images', {}).items()}
        record.setdefault('dataset', 'dataset')
        records.append(record)
    return records
