"""Official CUT with residual prediction, residual L1, radiometric views and DRC."""

import torch
from torch import nn
from torch.nn import functional as F

from .cut_model import CUTModel
from .radiometric_views import RadiometricViews
from model.ablation import add_ablation_options


class ResCUTModel(CUTModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = add_ablation_options(parser)
        parser.add_argument('--lambda_GAN', type=float, default=1.)
        parser.add_argument('--lambda_NCE', type=float, default=1.)
        parser.add_argument('--nce_layers', default='0,4,8,12,16')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=.07)
        parser.add_argument('--num_patches', type=int, default=256)
        parser.set_defaults(CUT_mode='CUT', nce_idt=True, netF='mlp_sample',
                            flip_equivariance=False,
                            nce_includes_all_negatives_from_minibatch=False)
        if is_train:
            parser.add_argument('--calibration',
                                help='Frozen training-only calibration.npz with grid, cbct and ct arrays')
            parser.add_argument('--lambda_residual', type=float, default=1.)
            parser.add_argument('--lambda_DRC', type=float, default=.05)
        return parser

    def __init__(self, opt):
        if (opt.use_residual_l1 and not opt.use_residual) or (opt.use_drc and not opt.use_radiometric_views):
            raise ValueError('Residual L1 requires residual prediction; DRC requires radiometric views')
        if (opt.CUT_mode.lower() != 'cut' or opt.flip_equivariance
                or opt.direction != 'AtoB' or not opt.nce_idt
                or opt.netG != 'resnet_9blocks' or opt.netF != 'mlp_sample'
                or opt.input_nc != 1 or opt.output_nc != 1):
            raise ValueError('ResCUT requires grayscale AtoB CUT with ResNet-9, MLP F and NCE-X/Y')
        if opt.isTrain and (opt.gan_mode != 'lsgan' or opt.netD != 'basic'
                            or opt.lambda_NCE <= 0 or opt.lambda_GAN <= 0
                            or opt.lambda_residual < 0 or opt.lambda_DRC < 0):
            raise ValueError('Use mean LSGAN/PatchGAN, positive GAN/NCE and nonnegative regularizers')
        super().__init__(opt)
        if not isinstance(self.netG.model[-1], nn.Tanh):
            raise ValueError('Expected the original CUT generator to end with tanh')
        self.residual_learning = opt.use_residual
        if self.residual_learning:
            self.netG.model[-1] = nn.Identity()
        self.visual_names = ['real_A', 'fake_B', 'residual_B']
        if self.isTrain:
            if opt.use_radiometric_views:
                self.views = RadiometricViews(opt.calibration).to(self.device)
            self.loss_names = ['G_GAN', 'G_resid', 'NCE', 'NCE_Y', 'G',
                               'D_real', 'D_fake', 'D_DRC', 'D']
        self.view_params = None

    def set_input(self, data):
        self.real_A = data['A'].to(self.device)
        self.image_paths = data['A_paths']
        if self.isTrain:
            self.real_B = data['B'].to(self.device)
            if self.real_A.shape != self.real_B.shape:
                raise ValueError('Unpaired domain batches must have equal shapes')
            if self.opt.use_radiometric_views:
                self.foreground_A = data['A_foreground'].to(self.device)
                self.foreground_B = data['B_foreground'].to(self.device)
                for image, mask in ((self.real_A, self.foreground_A),
                                    (self.real_B, self.foreground_B)):
                    if mask.dtype != torch.bool or mask.shape != image.shape:
                        raise ValueError('Provide a Boolean foreground mask for each original input')

    def data_dependent_initialize(self, data):
        # CUT trims images per GPU during lazy initialization; trim their masks too.
        count = data['A'].size(0) // max(len(self.gpu_ids), 1)
        if count < 1:
            raise ValueError('Batch size must be at least the number of GPUs')
        self.set_input(data)
        self.real_A = self.real_A[:count]
        if self.isTrain:
            self.real_B = self.real_B[:count]
            if self.opt.use_radiometric_views:
                self.foreground_A = self.foreground_A[:count]
                self.foreground_B = self.foreground_B[:count]
        self.forward()
        if self.isTrain:
            self.compute_D_loss().backward()
            self.compute_G_loss().backward()
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                               betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_F)

    def forward(self):
        self.real = torch.cat((self.real_A, self.real_B)) if self.isTrain else self.real_A
        predicted = self.netG(self.real)
        self.fake = self.real + predicted if self.residual_learning else predicted
        self.residual = predicted if self.residual_learning else predicted-self.real
        self.fake_B = self.fake[:len(self.real_A)]
        self.residual_B = self.residual[:len(self.real_A)]
        if self.isTrain:
            self.idt_B = self.fake[len(self.real_A):]
        self.view_params = None

    def _scores(self, image, domain):
        if not self.opt.use_radiometric_views:
            return [self.netD(image)]
        foreground = self.foreground_A if domain == 'cbct' else self.foreground_B
        calibrated = self.views.calibrate(image, domain, foreground)
        return [self.netD(self.views(calibrated, params, foreground))
                for params in self.view_params]

    def compute_D_loss(self):
        if self.opt.use_radiometric_views:
            self.view_params = self.views.sample_pair(self.real_A)
        fake = self._scores(self.fake_B.detach(), 'cbct')
        real = self._scores(self.real_B, 'ct')
        self.loss_D_real = sum(self.criterionGAN(s, True) for s in real) / len(real)
        self.loss_D_fake = sum(self.criterionGAN(s, False) for s in fake) / len(fake)
        self.loss_D_DRC = self.fake_B.new_zeros(())
        if self.opt.use_drc:
            consistency = (F.l1_loss(real[1], real[0].detach())
                           + F.l1_loss(fake[1], fake[0].detach())) / 2
            self.loss_D_DRC = self.opt.lambda_DRC * consistency
        self.loss_D = (self.loss_D_real + self.loss_D_fake) / 2 + self.loss_D_DRC
        return self.loss_D

    def compute_G_loss(self):
        if self.opt.use_radiometric_views and self.view_params is None:
            self.view_params = self.views.sample_pair(self.real_A)
        fake = self._scores(self.fake_B, 'cbct')
        self.loss_G_GAN = self.opt.lambda_GAN * sum(
            self.criterionGAN(s, True) for s in fake) / len(fake)
        self.loss_G_resid = (self.opt.lambda_residual * self.residual.float().abs().mean()
                             if self.opt.use_residual_l1 else self.fake_B.new_zeros(()))
        self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
        self.loss_G = (self.loss_G_GAN + self.loss_G_resid
                       + (self.loss_NCE + self.loss_NCE_Y) / 2)
        return self.loss_G
