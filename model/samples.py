"""Training sample export; no contribution to losses or checkpoint selection."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
import torch


def record_run(opt, dataset):
    options = vars(opt).copy()
    if not np.isfinite(options['max_dataset_size']):
        options['max_dataset_size'] = None
    folder = Path(opt.checkpoints_dir) / opt.name
    folder.mkdir(parents=True, exist_ok=True)
    record = dict(model_options=options, precision='FP32', train_cases=len(dataset.records),
        train_slices=len(dataset), cache_fingerprint=dataset.fingerprint,
        residual_prediction=opt.use_residual,
        residual_l1=opt.use_residual_l1, drc=opt.use_drc,
        radiometric_views=dataset.use_views, background_masking=dataset.use_views,
        samples_every_epochs=2, checkpoints_every_epochs=opt.save_epoch_freq)
    (folder / 'config.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')


def save_samples(model, dataset, opt, epoch):
    folder = Path(opt.checkpoints_dir) / opt.name / 'samples' / f'epoch_{epoch:04d}'
    folder.mkdir(parents=True, exist_ok=False)
    net = model.netG
    was_training = net.training
    net.eval()
    subjects, selected = set(), []
    for record in sorted(dataset.records, key=lambda r: (-r['depth'], r['patient'])):
        if record['subject_id'] not in subjects:
            selected.append(record)
            subjects.add(record['subject_id'])
        if len(selected) == 2:
            break
    with torch.inference_mode():
        for i, record in enumerate(selected):
            z = record['depth'] // 2
            source = dataset.tensor('cbct', record['packed_start'] + z)[None].to(model.device)
            predicted = net(source)
            restored = source + predicted if getattr(model, 'residual_learning', False) else predicted
            affine = np.asarray(record['affine']).copy()
            affine[:3, 3] += affine[:3, 2] * z
            for name, tensor in (('real_A', source), ('fake_B', restored), ('residual_B', restored-source)):
                values = tensor[0, 0].float().cpu().numpy()
                nib.save(nib.Nifti1Image((values*1000).T[:, :, None], affine), folder / f'{i}_{name}.nii.gz')
                shown = np.clip((values+1)*127.5, 0, 255).astype(np.uint8)
                Image.fromarray(np.rot90(shown, 2)).save(folder / f'{i}_{name}.png')
            (folder / f'{i}_source.json').write_text(json.dumps(dict(case=record['patient'],
                source_z=record['source_start_z']+z, split='train')) + '\n')
    net.train(was_training)
