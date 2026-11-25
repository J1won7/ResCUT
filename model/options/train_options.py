from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument('--n_epochs', type=int, default=50, help='Constant learning rate epochs')
        parser.add_argument('--n_epochs_decay', type=int, default=50, help='Linear decay epochs')
        parser.add_argument('--lr', type=float, default=.0002)
        parser.add_argument('--beta1', type=float, default=.5)
        parser.add_argument('--beta2', type=float, default=.999)
        parser.add_argument('--save_epoch_freq', type=int, default=10)
        parser.add_argument('--save_latest_freq', type=int, default=5000)
        parser.add_argument('--print_freq', type=int, default=100)
        parser.add_argument('--continue_train', action='store_true', help='Reload network weights only (CUT semantics)')
        parser.add_argument('--epoch_count', type=int, default=1)
        parser.set_defaults(phase='train', gan_mode='lsgan', pool_size=0,
                            lr_policy='linear', lr_decay_iters=50, pretrained_name=None,
                            save_by_iter=False)
        self.isTrain = True
        return parser
