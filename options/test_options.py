from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.add_argument('--nifti_recursive', action='store_true', help='recursively scan dataroot for .nii/.nii.gz files')
        parser.add_argument('--input_dir', type=str, default='', help='raw nifti directory to run inference on')
        parser.add_argument('--manifest_path', type=str, default='', help='explicit path to preprocessing_manifest.json. Overrides automatic lookup from --dataroot when set')
        parser.add_argument('--nifti_reader', type=str, default='nibabel_reorient', choices=['nibabel', 'nibabel_reorient'], help='reader used for raw nifti loading/writing')
        parser.add_argument('--infer_patch_size', type=str, default='', help='optional override for 3D sliding-window patch size as D,H,W. If omitted, infer from saved train options first and then from preprocessing_manifest.json')
        parser.add_argument('--infer_overlap', type=float, default=0.5, help='sliding-window overlap ratio in [0, 1)')
        parser.add_argument('--infer_batch_size', type=int, default=1, help='number of 3D windows per forward pass during nifti inference')

        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
