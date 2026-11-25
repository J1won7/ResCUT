"""Generator-only full-slice NIfTI inference (no discriminator preprocessing)."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model.models import networks


def load_generator(checkpoint, device, residual=None):
    if residual is None:
        config = Path(checkpoint).parent/'config.json'
        metadata = json.loads(config.read_text()) if config.is_file() else {}
        residual = metadata.get('model_options', {}).get('use_residual',
                    metadata.get('residual_prediction', True))
    if not isinstance(residual, bool):
        raise ValueError('Residual prediction must be a Boolean setting')
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if weights and all(key.startswith('module.') for key in weights):
        weights = {key.removeprefix('module.'): value for key, value in weights.items()}
    width = weights['model.1.weight'].shape[0]
    net = networks.define_G(1, 1, width, 'resnet_9blocks', norm='instance',
                            use_dropout=False, init_type='xavier', gpu_ids=[])
    if residual:
        net.model[-1] = nn.Identity()
    net.load_state_dict(weights, strict=True)
    net.residual_learning = residual
    return net.to(device).eval()


def restore_volume(net, volume, device, batch_size=8):
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError('Expected a finite X,Y,Z volume')
    if batch_size < 1 or min(volume.shape) < 1 or min(volume.shape[:2]) < 32:
        raise ValueError('Use a positive batch size and axial dimensions at least 32')
    slices = np.clip(volume, -1000, 1000).transpose(2, 1, 0).astype(np.float32) / 1000
    result = np.empty_like(slices)
    height, width = slices.shape[1:]
    pad_h, pad_w = (-height) % 4, (-width) % 4
    with torch.inference_mode():
        for start in range(0, len(slices), batch_size):
            source = torch.from_numpy(slices[start:start+batch_size, None]).to(device)
            padded = F.pad(source, (0, pad_w, 0, pad_h), mode='reflect')
            predicted = net(padded)
            restored = padded + predicted if getattr(net, 'residual_learning', True) else predicted
            result[start:start+batch_size] = restored[:, 0, :height, :width].cpu().numpy()
    return np.clip(result.transpose(2, 1, 0)*1000, -1000, 1000)


def infer_file(net, source, destination, device, batch_size=8):
    image = nib.load(source)
    restored = restore_volume(net, image.get_fdata(dtype=np.float32), device, batch_size)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1, 0)
    output = nib.Nifti1Image(restored, image.affine, header)
    for form in ('qform', 'sform'):
        matrix, code = getattr(image, 'get_'+form)(coded=True)
        getattr(output, 'set_'+form)(matrix, int(code))
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, help='One NIfTI file or a directory of NIfTI files')
    parser.add_argument('--output', required=True, help='Output file or directory, matching input type')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--residual', action=argparse.BooleanOptionalAction, default=None,
                        help='Override config.json; bare checkpoints default to residual prediction')
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    files = sorted([*source.glob('*.nii'), *source.glob('*.nii.gz')]) if source.is_dir() else [source]
    if not files or any(not path.is_file() for path in files):
        parser.error('No input NIfTI files found')
    if source.is_dir() and output.suffix in ('.nii', '.gz'):
        parser.error('Directory input requires a directory output')
    destinations = [output / path.name if source.is_dir() else output for path in files]
    if any(path.resolve() == dest.resolve() for path, dest in zip(files, destinations)):
        parser.error('Output must not overwrite input images')
    net = load_generator(args.checkpoint, args.device, args.residual)
    for path, destination in zip(files, destinations):
        infer_file(net, path, destination, args.device, args.batch_size)
        print(destination, flush=True)


if __name__ == '__main__':
    main()
