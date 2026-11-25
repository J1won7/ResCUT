"""One fully contained 16x16 source-NTV ROI per region and eligible slice."""
import argparse
from pathlib import Path

import numpy as np

from .io import load, read_manifest, write_csv

FIELDS = ('case_id', 'region', 'roi_id', 'slice_z', 'x_start', 'x_stop',
          'y_start', 'y_stop', 'size_x', 'size_y', 'rectangle_pixels',
          'ntv', 'candidate_count')


def window_sum(array, height, width):
    summed = np.pad(array, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    return (summed[height:, width:] - summed[:-height, width:]
            - summed[height:, :-width] + summed[:-height, :-width])


def choose(plane, mask, size=16, maximum=True):
    if size < 2:
        raise ValueError('ROI side must be at least two pixels')
    if min(mask.shape) < size:
        return None, 0
    valid = window_sum(mask.astype(np.int64), size, size) == size * size
    count = int(valid.sum())
    if not count:
        return None, 0
    normalized = (np.clip(np.asarray(plane, dtype=np.float64), -1000, 1000) + 1000) / 2000.
    variation = (window_sum(np.abs(np.diff(normalized, axis=0)), size-1, size)
                 + window_sum(np.abs(np.diff(normalized, axis=1)), size, size-1)) / (size*size)
    extreme = (np.max if maximum else np.min)(variation[valid])
    x, y = np.argwhere(valid & (np.abs(variation-extreme) <= 1e-12))[0]
    return (int(x), int(y), float(variation[x, y])), count


def select_rois(raw, artifact, reference, case, size=16):
    rows = []
    for region, mask in (('artifact', artifact), ('reference', reference)):
        eligible = mask & (raw > -900)
        for z in range(raw.shape[2]):
            location, count = choose(raw[:, :, z], eligible[:, :, z], size,
                                     maximum=region == 'artifact')
            if location is None:
                continue
            x, y, ntv = location
            rows.append(dict(case_id=case, region=region, roi_id=len(rows)+1,
                slice_z=z, x_start=x, x_stop=x+size, y_start=y, y_stop=y+size,
                size_x=size, size_y=size, rectangle_pixels=size*size,
                ntv=ntv, candidate_count=count))
    return rows


def validate_rois(rows, raw, artifact, reference):
    seen = set()
    for row in rows:
        region = row['region']
        if region not in ('artifact', 'reference'):
            raise ValueError('Unknown ROI region')
        x0, x1, y0, y1, z = [int(row[k]) for k in
                            ('x_start', 'x_stop', 'y_start', 'y_stop', 'slice_z')]
        if not (0 <= x0 < x1 <= raw.shape[0] and 0 <= y0 < y1 <= raw.shape[1]
                and 0 <= z < raw.shape[2] and x1-x0 == y1-y0 == 16):
            raise ValueError('Invalid 16x16 ROI bounds')
        if (region, z) in seen:
            raise ValueError('At most one ROI per region per slice is allowed')
        seen.add((region, z))
        mask = artifact if region == 'artifact' else reference
        if not (mask[x0:x1, y0:y1, z] & (raw[x0:x1, y0:y1, z] > -900)).all():
            raise ValueError('ROI is not fully contained in the source-derived eligible mask')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    for record in read_manifest(args.manifest):
        geometry, raw = load(record['source'])
        artifact = load(record['artifact_mask'], geometry, mask=True)[1]
        reference = load(record['reference_mask'], geometry, mask=True)[1]
        rows = select_rois(raw, artifact, reference, record['case'])
        dest = Path(args.output) / record['dataset'] / (record['case']+'.csv')
        write_csv(dest, rows, FIELDS)
        print(dest, len(rows), flush=True)


if __name__ == '__main__':
    main()
