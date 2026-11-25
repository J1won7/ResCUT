"""Frozen monotonic calibration and increasing Models Genesis intensity views.

Zhou et al., Medical Image Analysis 67 (2021), 101840.
Reference: MrGiovanni/ModelsGenesis, pytorch/utils.py, nonlinear_transformation.
Only its increasing branch is used. Differentiable LUT interpolation replaces
the reference's image-side NumPy interpolation; out-of-range values use linear
endpoint continuation rather than hard clipping.
"""

import numpy as np
import torch
from torch import nn


class RadiometricViews(nn.Module):
    def __init__(self, calibration):
        super().__init__()
        with np.load(calibration, allow_pickle=False) as payload:
            grid = payload['grid']
            if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
                    or not np.allclose(grid, np.linspace(-1, 1, len(grid)), rtol=0, atol=1e-6)):
                raise ValueError('Calibration grid must be uniform on [-1,1]')
            for domain in ('cbct', 'ct'):
                values = payload[domain]
                if (values.shape != grid.shape or not np.isfinite(values).all()
                        or np.any(np.diff(values) <= 0)):
                    raise ValueError('Calibration maps must be finite and strictly increasing')
                self.register_buffer(domain + '_map', torch.tensor(values, dtype=torch.float32))
        t = np.linspace(0, 1, 100000)
        self.basis = np.stack(((1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3))
        self.lookup_grid = np.linspace(0, 1, 4097)

    def make_lut(self, controls):
        controls = np.asarray(controls, dtype=np.float64)
        if (controls.ndim != 2 or controls.shape[1] != 4
                or not np.isfinite(controls).all() or np.any((controls < 0) | (controls > 1))):
            raise ValueError('Expected two interior (x,y) control points in [0,1] per image')
        curves = []
        for x1, y1, x2, y2 in controls:
            x = np.array((0., x1, x2, 1.)) @ self.basis
            y = np.array((0., y1, y2, 1.)) @ self.basis
            curves.append(np.interp(self.lookup_grid, np.sort(x), np.sort(y)))
        return np.asarray(curves, dtype=np.float32)

    def sample_pair(self, image):
        def sample():
            controls = torch.rand(image.shape[0], 4, device='cpu').numpy()
            return torch.from_numpy(self.make_lut(controls)).to(image.device)
        return sample(), sample()

    @staticmethod
    def interpolate(image, lookup):
        if lookup.ndim != 2 or image.shape[0] != lookup.shape[0]:
            raise ValueError('Each image requires one lookup table')
        with torch.autocast(device_type=image.device.type, enabled=False):
            lookup = lookup.float()
            position = (image.float()+1)*((lookup.shape[1]-1)/2)
            index = position.floor().long().clamp(0, lookup.shape[1]-2)
            flat = index.flatten(1)
            lower = lookup.gather(1, flat).reshape_as(image)
            upper = lookup.gather(1, flat+1).reshape_as(image)
            return lower + (position-index)*(upper-lower)

    def calibrate(self, image, domain, foreground):
        if domain not in ('cbct', 'ct'):
            raise ValueError('Domain must be cbct or ct')
        lookup = getattr(self, domain+'_map').expand(image.shape[0], -1)
        corrected = self.interpolate(image, lookup)
        return torch.where(foreground, corrected, image)

    def forward(self, calibrated, params, foreground):
        transformed = 2*self.interpolate(calibrated, params)-1
        # Only discriminator views are masked; restored images remain untouched.
        return torch.where(foreground, transformed, transformed.new_full((), -1.))
