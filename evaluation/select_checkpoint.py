"""Select the lowest validation mean AI_SD after the constant-LR phase."""
import argparse
import csv
from pathlib import Path

import numpy as np

from .io import save_json


def select(rows, decay_start=50):
    groups = {}
    for row in rows:
        epoch = int(row['epoch'])
        if epoch >= decay_start:
            groups.setdefault(epoch, []).append(row)
    cohorts, scores = [], []
    for epoch, group in sorted(groups.items()):
        ids = {(r.get('dataset', ''), r['case']) for r in group}
        if len(ids) != len(group):
            raise ValueError('One model and one row per validation case are required')
        if any(r['ai_sd'] in ('', None) or not np.isfinite(float(r['ai_sd'])) for r in group):
            raise ValueError('Missing validation AI_SD; do not silently change the selection cohort')
        values = [float(r['ai_sd']) for r in group]
        cohorts.append(ids)
        scores.append(dict(epoch=epoch, n=len(values), mean=float(np.mean(values)),
                           variance=float(np.var(values, ddof=1)) if len(values)>1 else None))
    if not scores or any(ids != cohorts[0] for ids in cohorts):
        raise ValueError('All candidate checkpoints must use the same nonempty validation cohort')
    return dict(best=min(scores, key=lambda r: (r['mean'], r['epoch'])), candidates=scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', required=True, help='CSV: epoch,case,ai_sd; one model only')
    parser.add_argument('--output', required=True)
    parser.add_argument('--decay-start', type=int, default=50)
    args = parser.parse_args()
    result = select(list(csv.DictReader(Path(args.metrics).open())), args.decay_start)
    save_json(args.output, result)
    print(result['best'])


if __name__ == '__main__':
    main()
