"""Prepare official validation CBCTs on the exact released annotation grids."""
import argparse
import json
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from .io import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', required=True, help='Root containing CASE/cbct.nii.gz')
    parser.add_argument('--output', required=True)
    parser.add_argument('--metadata', default=str(Path(__file__).resolve().parents[1]/
        'annotations/synthrad2023/metadata.json'))
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Use a new output directory')
    output.mkdir(parents=True)
    for record in json.loads(Path(args.metadata).read_text())['cases']:
        path = Path(args.native)/record['case']/'cbct.nii.gz'
        if sha256(path) != record['native_cbct_sha256']:
            raise ValueError(f'Official source file mismatch: {record["case"]}')
        image = nib.load(path)
        affine = np.asarray(record['affine'])
        adjustment = record['intensity_adjustment']
        values = np.clip(image.get_fdata(dtype=np.float32)*adjustment['scale']
                         + adjustment['offset'], -1000, 1000)
        sampled = resample_from_to(nib.Nifti1Image(values, image.affine),
            (tuple(record['shape']), affine), order=1, cval=-1000)
        target = output/(record['case']+'.nii.gz')
        nib.save(nib.Nifti1Image(sampled.get_fdata(dtype=np.float32), affine), target)
        print(target, flush=True)


if __name__ == '__main__':
    main()
