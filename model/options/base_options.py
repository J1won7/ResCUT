"""Small ResCUT interface around the unchanged CUT network implementation."""

import argparse
from pathlib import Path

import torch

from model.models import get_option_setter
from model.data import get_option_setter as dataset_options
from model.ablation import resolve_ablation


class BaseOptions:
    def __init__(self, cmd_line=None):
        self.cmd_line = None if cmd_line is None else cmd_line.split()

    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True, help='Cache created by prepare.py')
        parser.add_argument('--name', default='rescut')
        parser.add_argument('--checkpoints_dir', default='./checkpoints')
        parser.add_argument('--gpu_ids', default='0', help='Comma-separated CUDA indices; -1 for CPU')
        parser.add_argument('--model', choices=['rescut'], default='rescut')
        parser.add_argument('--dataset_mode', choices=['packed'], default='packed')
        parser.add_argument('--ngf', type=int, default=64)
        parser.add_argument('--ndf', type=int, default=64)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--num_threads', type=int, default=4)
        parser.add_argument('--max_dataset_size', type=int, default=float('inf'))
        parser.add_argument('--serial_batches', action='store_true')
        parser.add_argument('--epoch', default='latest')
        parser.add_argument('--verbose', action='store_true')
        # Architecture and preprocessing are fixed, not independent model variants.
        parser.set_defaults(input_nc=1, output_nc=1, netG='resnet_9blocks', netD='basic',
                            n_layers_D=3, normG='instance', normD='instance',
                            no_dropout=True, no_antialias=False, no_antialias_up=False,
                            init_type='xavier', init_gain=.02, direction='AtoB',
                            preprocess='none', crop_size=256, load_size=256,
                            no_flip=True, display_winsize=256)
        return parser

    def gather_options(self):
        parser = self.initialize(argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter))
        parser = get_option_setter('rescut')(parser, self.isTrain)
        parser = dataset_options('packed')(parser, self.isTrain)
        self.parser = parser
        return resolve_ablation(parser.parse_args(self.cmd_line), parser)

    def print_options(self, opt):
        message = '\n'.join(f'{key}: {value}' for key, value in sorted(vars(opt).items()))
        print(message)
        folder = Path(opt.checkpoints_dir)/opt.name
        folder.mkdir(parents=True, exist_ok=True)
        (folder/f'{opt.phase}_opt.txt').write_text(message+'\n')

    def parse(self):
        opt = self.gather_options()
        opt.isTrain = self.isTrain
        self.print_options(opt)
        opt.gpu_ids = [int(i) for i in opt.gpu_ids.split(',') if int(i) >= 0]
        if opt.gpu_ids:
            torch.cuda.set_device(opt.gpu_ids[0])
        self.opt = opt
        return opt
