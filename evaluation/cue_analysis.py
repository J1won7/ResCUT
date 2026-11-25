"""Mean-PatchGAN input-gradient saliency on annotated artifact slices."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model.models import networks
from model.models.radiometric_views import RadiometricViews
from model.prepare import foreground
from .io import load, read_manifest, save_json, sha256, write_csv


def case_cues(net, views, raw, artifact, case, device, batch_size):
    body = raw > -900
    zs_all = [z for z in range(raw.shape[2]) if (artifact[:, :, z] & body[:, :, z]).any()
              and (body[:, :, z] & ~artifact[:, :, z]).any()]
    rows = []
    for start in range(0, len(zs_all), batch_size):
        zs = zs_all[start:start+batch_size]
        arrays = np.stack([raw[:, :, z].T/1000 for z in zs])[:, None]
        images = torch.from_numpy(arrays.copy()).to(device)
        if views is not None:
            masks = torch.from_numpy(np.stack([foreground(a[0]) for a in arrays])[:, None]).to(device)
            controls = []
            for z in zs:
                seed = int(hashlib.sha256(f'media-cue-20261109:{case}:{z}'.encode()).hexdigest()[:16], 16)
                controls.append(np.random.default_rng(seed).uniform(size=4))
            lookup = torch.from_numpy(views.make_lut(np.asarray(controls))).to(device)
            with torch.no_grad():
                images = views(views.calibrate(images, 'cbct', masks), lookup, masks)
        u = images.detach().requires_grad_(True)
        response = net(u).flatten(1).mean(1)
        gradient, = torch.autograd.grad(response.sum(), u)
        saliency = gradient.detach().abs()[:, 0].cpu().numpy()
        for i, z in enumerate(zs):
            a, normal = (artifact[:, :, z] & body[:, :, z]).T, (body[:, :, z] & ~artifact[:, :, z]).T
            sa, sn = (float(saliency[i][mask].mean(dtype=np.float64)) for mask in (a, normal))
            if sn <= 0 or not np.isfinite([sa, sn]).all():
                raise ValueError('Undefined saliency ratio')
            rows.append(dict(slice_z=z, s_artifact=sa, s_normal=sn, enrichment=sa/sn))
    summary = {key: float(np.mean([r[key] for r in rows])) if rows else None
               for key in ('s_artifact', 's_normal', 'enrichment')}
    return dict(valid_slices=len(rows), **summary), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True, help='Discriminator state_dict, not generator')
    parser.add_argument('--config', required=True, help='Training config.json for this exact checkpoint')
    parser.add_argument('--calibration', help='Required for variants trained with radiometric views')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    output = Path(args.output)
    if output.exists():
        parser.error('Use a new output directory')
    config = json.loads(Path(args.config).read_text())['model_options']
    net = networks.define_D(1, config['ndf'], config['netD'], config['n_layers_D'],
        norm=config['normD'], init_type=config['init_type'], init_gain=config['init_gain'],
        no_antialias=config['no_antialias'], gpu_ids=[])
    weights = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    weights = {k.removeprefix('module.'): v for k, v in weights.items()}
    net.load_state_dict(weights, strict=True)
    net.to(args.device).eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    views = None
    if config['use_radiometric_views']:
        if not args.calibration:
            parser.error('Supply the calibration used to train this discriminator')
        views = RadiometricViews(args.calibration).to(args.device)
    output.mkdir(parents=True)
    rows = []
    for record in read_manifest(args.manifest):
        geometry, raw = load(record['source'])
        artifact = load(record['artifact_mask'], geometry, mask=True)[1]
        summary, slices = case_cues(net, views, raw, artifact, record['case'], args.device, args.batch_size)
        row = dict(dataset=record['dataset'], case=record['case'], **summary)
        rows.append(row)
        save_json(output/'cases'/record['dataset']/(record['case']+'.json'), dict(summary=row, slices=slices))
    write_csv(output/'case_metrics.csv', rows)
    summaries = []
    for dataset in sorted({r['dataset'] for r in rows}):
        group = [r for r in rows if r['dataset'] == dataset and r['enrichment'] is not None]
        row = dict(dataset=dataset, n=len(group))
        for key in ('s_artifact', 's_normal', 'enrichment'):
            values = [r[key] for r in group]
            row[key+'_mean'] = float(np.mean(values)) if values else None
            row[key+'_variance'] = float(np.var(values, ddof=1)) if len(values)>1 else None
        summaries.append(row)
    write_csv(output/'summary_mean_variance.csv', summaries)
    save_json(output/'protocol.json', dict(checkpoint_sha256=sha256(args.checkpoint),
        config_sha256=sha256(args.config), calibration_sha256=sha256(args.calibration) if views else None,
        gradient='Absolute gradient of mean PatchGAN response with respect to native D-input view.',
        aggregation='Slice S_A/S_N, then volume mean, then equal-volume mean/sample variance.',
        region='Raw > -900; artifact intersection and its foreground complement; no dilation.'))


if __name__ == '__main__':
    main()
