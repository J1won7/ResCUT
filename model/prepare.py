"""Pack training-only NIfTI slices and estimate fixed foreground CDF calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
from scipy import ndimage

from evaluation.io import save_json, sha256


def directory_cases(input_dir, label_dir):
    domains = {}
    for domain, directory in (('cbct', input_dir), ('ct', label_dir)):
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise ValueError(f'Not a directory: {directory}')
        domains[domain] = {p.name: p for p in directory.iterdir()
                           if p.is_file() and (p.name.endswith('.nii.gz') or p.name.endswith('.nii'))}
        if not domains[domain]:
            raise ValueError(f'No NIfTI files found: {directory}')
    missing_ct = sorted(domains['cbct'].keys() - domains['ct'].keys())
    missing_cbct = sorted(domains['ct'].keys() - domains['cbct'].keys())
    if missing_ct or missing_cbct:
        raise ValueError(f'Filenames must match exactly. Missing CT: {missing_ct}; missing CBCT: {missing_cbct}')
    entries, seen = [], set()
    for name in sorted(domains['cbct']):
        case = name[:-7] if name.endswith('.nii.gz') else name[:-4]
        if not case or case in seen:
            raise ValueError(f'Empty or duplicate case name: {name}')
        seen.add(case)
        sources = {d: str(files[name]) for d, files in domains.items()}
        images = {d: nib.load(p) for d, p in sources.items()}
        if any(im.ndim != 3 or im.shape[:2] != (256, 256) for im in images.values()):
            raise ValueError(f'Expected preoriented 256x256 axial volumes: {name}')
        if images['cbct'].shape != images['ct'].shape:
            raise ValueError(f'CBCT and CT shapes must match to use every slice: {name}')
        entries.append(dict(case=case, subject=case, **sources,
                            start_z=0, stop_z=images['cbct'].shape[2]))
    return entries


def foreground(image):
    candidate = ndimage.binary_closing(image > -.8, iterations=2)
    labels, count = ndimage.label(candidate)
    if not count:
        return np.zeros_like(image, dtype=bool)
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    largest = np.argsort(areas)[::-1][:2]
    return ndimage.binary_fill_holes(np.isin(labels,
        [i for i in largest if areas[i] >= .2*areas[largest[0]]]))


def fit_calibration(cdfs, grid):
    probability = np.linspace(0, 1, 10001)
    pooled = {d: np.mean(values, axis=0) for d, values in cdfs.items()}
    common = np.mean([np.interp(probability, cdf, grid) for cdf in pooled.values()], axis=0)
    maps = {d: np.interp(cdf, probability, common) for d, cdf in pooled.items()}
    if any(np.any(np.diff(v) <= 0) for v in maps.values()):
        raise ValueError('Calibration has flat intervals. Inspect training foreground distributions; '
                         'no unreported smoothing or slope floor is applied.')
    return maps


def prepare(manifest_path, output):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new cache directory')
    entries = json.loads(manifest_path.read_text())['cases']
    if not entries or len({r['case'] for r in entries}) != len(entries):
        raise ValueError('Provide a nonempty training-only manifest with unique case names')
    planned = []
    offset = 0
    for row in entries:
        if Path(row['case']).name != row['case'] or row['case'] in ('.', '..'):
            raise ValueError('Case names must be filenames, not paths')
        sources = {d: manifest_path.parent/row[d] for d in ('cbct', 'ct')}
        images = {d: nib.load(p) for d, p in sources.items()}
        start = int(row.get('start_z', 0))
        stop = int(row.get('stop_z', min(im.shape[2] for im in images.values())))
        if any(im.ndim != 3 or im.shape[:2] != (256, 256) or
               not 0 <= start < stop <= im.shape[2] for im in images.values()):
            raise ValueError('Supply preoriented 256x256 axial volumes and valid zero-based slice ranges')
        affine = images['cbct'].affine.copy()
        affine[:3, 3] += affine[:3, 2]*start
        record = dict(patient=row['case'], subject_id=row['subject'], depth=stop-start,
            packed_start=offset, packed_stop=offset+stop-start, source_start_z=start,
            affine=affine.tolist(), packed_files={d:f'packed/{d}.npy' for d in sources})
        planned.append((record, sources))
        offset += stop-start
    (output/'packed').mkdir(parents=True)
    (output/'calibration/foreground').mkdir(parents=True)
    arrays = {d: np.lib.format.open_memmap(output/f'packed/{d}.npy', mode='w+',
              dtype=np.float32, shape=(offset, 256, 256)) for d in ('cbct', 'ct')}
    grid = np.linspace(-1, 1, 4097)
    cdfs = {d: [] for d in arrays}
    source_hashes = []
    for record, sources in planned:
        for domain, path in sources.items():
            image = nib.load(path)
            start, depth = record['source_start_z'], record['depth']
            raw = np.asarray(image.dataobj[:, :, start:start+depth], dtype=np.float32)
            if not np.isfinite(raw).all():
                raise ValueError(f'Nonfinite training input: {path}')
            volume = np.ascontiguousarray(np.clip(raw, -1000, 1000).transpose(2, 1, 0)/np.float32(1000))
            masks = np.stack([foreground(plane) for plane in volume])
            histogram = np.histogram(volume[masks], bins=grid)[0]
            if histogram.sum() == 0:
                raise ValueError(f'Empty training foreground: {path}')
            cdfs[domain].append(np.r_[0., np.cumsum(histogram)]/histogram.sum())
            arrays[domain][record['packed_start']:record['packed_stop']] = volume
            np.save(output/'calibration/foreground'/f'{record["patient"]}_{domain}.npy', masks)
            source_hashes.append(dict(case=record['patient'], domain=domain, sha256=sha256(path)))
        print(record['patient'], record['depth'], flush=True)
    for array in arrays.values():
        array.flush()
    maps = fit_calibration(cdfs, grid)
    np.savez(output/'calibration/calibration.npz', grid=grid, **maps)
    split = dict(splits=dict(train=[r['patient'] for r, _ in planned], validation=[], test=[]))
    save_json(output/'split.json', split)
    save_json(output/'cache/manifest.json', dict(array_order='Z,Y,X', value_range=[-1, 1],
        cases=[r for r, _ in planned],
        split_sha256=hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()))
    save_json(output/'calibration/provenance.json', dict(training_manifest_sha256=sha256(manifest_path),
        source_hashes=source_hashes, volume_count_per_domain=len(entries), slices_per_domain=offset,
        calibration='Equal-volume mean CDFs; common mean inverse-CDF reference; piecewise linear maps.',
        grid_points=4097, probability_points=10001,
        pairing_used_for_calibration=False, training_only=True))
    save_json(output/'input_manifest.json', dict(cases=[dict(row,
        **{d: str((manifest_path.parent/row[d]).resolve()) for d in ('cbct', 'ct')}) for row in entries]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input-dir', help='Training CBCT NIfTI directory; filenames must match label-dir')
    inputs.add_argument('--manifest', help='Optional JSON for explicit cases and slice ranges')
    parser.add_argument('--label-dir', help='Training CT NIfTI directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.input_dir:
        if not args.label_dir:
            parser.error('--label-dir is required with --input-dir')
        entries = directory_cases(args.input_dir, args.label_dir)
        with tempfile.TemporaryDirectory(prefix='rescut-prepare-') as temp:
            manifest = Path(temp)/'train.json'
            save_json(manifest, dict(cases=entries))
            prepare(manifest, args.output)
    else:
        if args.label_dir:
            parser.error('--label-dir cannot be combined with --manifest')
        prepare(args.manifest, args.output)


if __name__ == '__main__':
    main()
