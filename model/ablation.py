"""Ablation presets and explicit switches for the four ResCUT additions."""

import argparse
from pathlib import Path


COMPONENTS = ('use_residual', 'use_residual_l1', 'use_radiometric_views', 'use_drc')
PRESETS = {
    'cut': (False, False, False, False),
    'cut_views': (False, False, True, False),
    'cut_residual': (True, True, False, False),
    'cut_residual_views': (True, True, True, False),
    'rescut': (True, True, True, True),
}


def add_ablation_options(parser):
    parser.add_argument('--ablation', choices=PRESETS, default='rescut')
    for name, destination in zip(('residual', 'residual-l1', 'radiometric-views', 'drc'), COMPONENTS):
        parser.add_argument('--'+name, dest=destination, action=argparse.BooleanOptionalAction,
                            default=None, help='Override the selected ablation preset')
    return parser


def resolve_ablation(opt, parser):
    explicit = {key: getattr(opt, key) for key in COMPONENTS}
    for key, default in zip(COMPONENTS, PRESETS[opt.ablation]):
        setattr(opt, key, default if explicit[key] is None else explicit[key])
    # Disabling a parent also disables its regularizer unless explicitly requested.
    for parent, child in (('use_residual', 'use_residual_l1'), ('use_radiometric_views', 'use_drc')):
        if not getattr(opt, parent):
            if explicit[child] is True:
                parser.error(f'{child} requires {parent}')
            setattr(opt, child, False)
    if opt.use_radiometric_views and not opt.calibration:
        opt.calibration = str(Path(opt.dataroot)/'calibration/calibration.npz')
    return opt
