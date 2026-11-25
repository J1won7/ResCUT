"""Evaluate all outputs jointly using frozen source ROIs and source masks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import numpy as np

from .io import load, read_manifest, save_json, sha256, write_csv
from .metrics import artifact_metrics, make_body_mask, preservation_metrics, preservation_support
from .roi import select_rois, validate_rois


def evaluate_case(record):
    geometry, raw = load(record['source'])
    artifact = load(record['artifact_mask'], geometry, mask=True)[1]
    reference = load(record['reference_mask'], geometry, mask=True)[1]
    body = (load(record['body_mask'], geometry, mask=True)[1]
            if record.get('body_mask') else make_body_mask(raw))
    rois = (list(csv.DictReader(Path(record['rois']).open())) if record.get('rois')
            else select_rois(raw, artifact, reference, record['case']))
    validate_rois(rois, raw, artifact, reference)
    images = {'Input': raw}
    for method, path in record.get('images', {}).items():
        if method == 'Input':
            if Path(path).resolve() != Path(record['source']).resolve():
                raise ValueError('Input must refer to the original source')
        else:
            images[method] = load(path, geometry)[1]
    indices, scores = artifact_metrics(images, rois)
    support = preservation_support(body, artifact, geometry.header.get_zooms()[:3])
    rows = [dict(dataset=record['dataset'], case=record['case'], method=method,
                 artifact_status='annotated' if artifact.any() else 'no_annotated_artifact',
                 **indices[method], **preservation_metrics(raw, output, support))
            for method, output in images.items()]
    hashes = {k: sha256(record[k]) for k in
              ('source', 'artifact_mask', 'reference_mask', 'body_mask', 'rois') if record.get(k)}
    hashes['images'] = {m: sha256(p) for m, p in record.get('images', {}).items()}
    return dict(rows=rows, raw_direction_scores=scores, hashes=hashes)


def summarize(rows):
    result = []
    metrics = ('ai_sd', 'rho', 'sigma_artifact', 'sigma_reference',
               'lambda_artifact', 'lambda_reference', 'outside_mae', 'outside_ssim', 'outside_lncc')
    for dataset, method in sorted({(r['dataset'], r['method']) for r in rows}):
        group = [r for r in rows if (r['dataset'], r['method']) == (dataset, method)]
        row = dict(dataset=dataset, method=method, cases=len(group))
        for metric in metrics:
            values = [r[metric] for r in group if r[metric] is not None]
            row[metric+'_n'] = len(values)
            row[metric+'_mean'] = float(np.mean(values)) if values else None
            row[metric+'_variance'] = float(np.var(values, ddof=1)) if len(values) > 1 else None
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    records = read_manifest(args.manifest)
    output = Path(args.output)
    if output.exists():
        parser.error('Use a new output directory to avoid mixing evaluation runs')
    output.mkdir(parents=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record, result in zip(records, pool.map(evaluate_case, records)):
            rows.extend(result['rows'])
            save_json(output/'cases'/record['dataset']/(record['case']+'.json'), result)
            print(record['dataset'], record['case'], flush=True)
    write_csv(output/'case_metrics.csv', rows)
    write_csv(output/'summary_mean_variance.csv', summarize(rows))
    save_json(output/'protocol.json', dict(protocol='source_ntv16_four_direction_rho_lncc5_v1',
        manifest_sha256=sha256(args.manifest), case_count=len(records),
        aggregation='Equal-volume mean and sample variance (ddof=1); missing metrics are not zero.'))


if __name__ == '__main__':
    main()
