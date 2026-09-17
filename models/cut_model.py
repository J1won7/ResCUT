import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from util.radiometric_views import (
    apply_brion,
    body_mask,
    load_anchor,
    masked_affine,
    sample_brion,
    sample_fan_shading,
)

class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_residual', type=float, default=1.0, help='weight for Residual L1 loss: ||G(X)||')
        parser.add_argument('--use_monotonic_D', action='store_true')
        parser.add_argument('--use_drc', action='store_true', help='enable anchored discriminator consistency across radiometric nuisance views')
        parser.add_argument('--lambda_DRC', type=float, default=0.05, help='weight of anchored discriminator consistency')
        parser.add_argument('--lambda_drc_gan', type=float, default=0.10, help='relative GAN weight of the perturbed radiometric branch')
        parser.add_argument('--radiometric_anchor_path', type=str, default='', help='fixed source/target radiometric anchor JSON')
        parser.add_argument('--radiometric_brion_scale_range', type=float, default=0.10)
        parser.add_argument('--radiometric_brion_shift_range', type=float, default=0.05)
        parser.add_argument('--radiometric_brion_piecewise_range', type=float, default=0.05)
        parser.add_argument('--radiometric_fan_shading_strength', type=float, default=0.04)
        parser.add_argument('--use_residual_learning', action='store_true', help='Use Residual Learning structure for the generator')

        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'G_resid']
        if self.isTrain and self.opt.use_drc:
            self.loss_names += ['G_DRC_GAN', 'D_DRC_GAN', 'D_DRC']
        if self.opt.use_residual_learning:
            self.visual_names = ['real_A', 'fake_B', 'real_B', 'residual_B']
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        self.radiometric_anchor = None
        if self.isTrain and self.opt.use_drc:
            if not self.opt.radiometric_anchor_path:
                raise ValueError('--use_drc requires --radiometric_anchor_path')
            self.radiometric_anchor = load_anchor(self.opt.radiometric_anchor_path)
            if self.opt.lambda_drc_gan < 0:
                raise ValueError('--lambda_drc_gan must be non-negative')
            print(
                '[DRC] anchored canonical/perturbed D with '
                'lambda_gan=%.4f lambda_consistency=%.4f' % (
                    self.opt.lambda_drc_gan,
                    self.opt.lambda_DRC,
                )
            )
        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1Residual = torch.nn.L1Loss().to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def _sample_monotonic_params(self, x):
        bs, c = x.shape[:2]
        spatial_rank = x.dim() - 2
        view_shape = [bs, c] + [1] * spatial_rank
        gamma = torch.rand(*view_shape, device=x.device) * 1.5 + 0.5
        scale = torch.rand(*view_shape, device=x.device) * 0.4 + 0.8
        shift = torch.rand(*view_shape, device=x.device) * 0.2 - 0.1
        return gamma, scale, shift

    def preprocess_for_discriminator(self, x, params=None):
        """Apply a coordinate-preserving monotonic radiometric transform."""

        if not self.opt.use_monotonic_D:
            return x, None
        # Gamma is evaluated in [0, 1] and mapped back to the model range.
        x_norm = torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)

        if params is None:
            params = self._sample_monotonic_params(x)
        gamma, scale, shift = params

        x_pow = torch.clamp(x_norm, 1e-6, 1.0)
        x_aug = torch.pow(x_pow, gamma)
        x_aug = x_aug * scale + shift

        x_aug = torch.clamp(x_aug, 0.0, 1.0)
        x_out = x_aug * 2.0 - 1.0
        return x_out, params

    def _radiometric_views(self, x, domain, monotonic_params=None):
        """Preserve the legacy monotonic-D branch and attach one anchored nuisance view."""
        mask = body_mask(x)
        if self.radiometric_anchor is not None:
            affine = self.radiometric_anchor['affine'][domain]
            x = masked_affine(x, mask, affine['scale'], affine['shift'])
        canonical, monotonic_params = self.preprocess_for_discriminator(x, monotonic_params)
        if not self.opt.use_drc:
            return canonical, None, monotonic_params
        params = sample_brion(
            canonical,
            self.opt.radiometric_brion_scale_range,
            self.opt.radiometric_brion_shift_range,
            self.opt.radiometric_brion_piecewise_range,
        )
        perturbed = apply_brion(canonical, mask, params)
        perturbed = perturbed + sample_fan_shading(canonical, mask, self.opt.radiometric_fan_shading_strength)
        return canonical, perturbed, monotonic_params

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [-1])

        if self.opt.use_residual_learning:
            self.residual_B = self.netG(self.real)
            self.fake = self.real + self.residual_B
        else:
            self.fake = self.netG(self.real)
        
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        monotonic_params = self._sample_monotonic_params(fake) if self.opt.use_monotonic_D else None
        fake_canonical, fake_perturbed, _ = self._radiometric_views(fake, 'source', monotonic_params)
        real_canonical, real_perturbed, _ = self._radiometric_views(self.real_B, 'target', monotonic_params)
        pred_fake = self.netD(fake_canonical)
        self.pred_real = self.netD(real_canonical)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.loss_D_real = self.criterionGAN(self.pred_real, True).mean()

        if self.opt.use_drc:
            pred_fake_perturbed = self.netD(fake_perturbed)
            pred_real_perturbed = self.netD(real_perturbed)
            perturbed_fake_loss = self.criterionGAN(pred_fake_perturbed, False).mean()
            perturbed_real_loss = self.criterionGAN(pred_real_perturbed, True).mean()
            self.loss_D_DRC_GAN = 0.5 * (
                perturbed_fake_loss + perturbed_real_loss
            ) * self.opt.lambda_drc_gan
            self.loss_D_DRC = 0.5 * (
                F.l1_loss(pred_fake_perturbed, pred_fake.detach()) +
                F.l1_loss(pred_real_perturbed, self.pred_real.detach())
            ) * self.opt.lambda_DRC
        else:
            self.loss_D_DRC_GAN = torch.zeros((), device=fake.device, dtype=fake.dtype)
            self.loss_D_DRC = torch.zeros((), device=fake.device, dtype=fake.dtype)

        self.loss_D = (
            (self.loss_D_fake + self.loss_D_real) * 0.5
            + self.loss_D_DRC_GAN
            + self.loss_D_DRC
        )
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            fake_canonical, fake_perturbed, _ = self._radiometric_views(fake, 'source')
            self.loss_G_GAN = self.criterionGAN(self.netD(fake_canonical), True).mean() * self.opt.lambda_GAN
            if self.opt.use_drc:
                self.loss_G_DRC_GAN = (
                    self.criterionGAN(self.netD(fake_perturbed), True).mean()
                    * self.opt.lambda_GAN
                    * self.opt.lambda_drc_gan
                )
            else:
                self.loss_G_DRC_GAN = torch.zeros((), device=fake.device, dtype=fake.dtype)
        else:
            self.loss_G_GAN = 0.0
            self.loss_G_DRC_GAN = torch.zeros((), device=fake.device, dtype=fake.dtype)

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        if self.opt.use_residual_learning:
            # Residual L1 Regularization
            self.loss_G_resid = self.criterionL1Residual(self.residual_B, torch.zeros_like(self.residual_B)) * self.opt.lambda_residual
        else:
            self.loss_G_resid = torch.zeros((), device=fake.device, dtype=fake.dtype)

        self.loss_G = self.loss_G_GAN + self.loss_G_DRC_GAN + loss_NCE_both + self.loss_G_resid
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [-1]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
