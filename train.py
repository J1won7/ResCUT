import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer


def _save_nifti(path, values_czyx):
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(values_czyx[0].transpose(2, 1, 0), dtype=np.float32)
    nib.save(nib.Nifti1Image(xyz, np.eye(4)), str(path))


def save_miccai_nifti_samples(model, opt, epoch):
    """Save deterministic full-volume test samples without changing training state."""
    if opt.dataset_mode != 'medimg_miccai2d' or opt.save_nifti_epoch_freq <= 0:
        return
    if epoch % opt.save_nifti_epoch_freq != 0:
        return

    from util.nifti_preprocessing import load_preprocessed_case

    split = json.loads(Path(opt.medimg_split_file).read_text(encoding='utf-8'))
    identifiers = list(split['domains']['a']['test'])[:opt.nifti_sample_count]
    manifest = json.loads((Path(opt.dataroot) / 'preprocessing_manifest.json').read_text(encoding='utf-8'))
    source_folder = Path(opt.dataroot) / manifest['domains']['a']['folder']
    output_root = Path(opt.checkpoints_dir) / opt.name / 'nifti_samples' / f'epoch_{epoch:03d}'

    model.eval()
    with torch.inference_mode():
        for number, identifier in enumerate(identifiers):
            source = np.asarray(
                load_preprocessed_case(str(source_folder), identifier)['image'],
                dtype=np.float32,
            )
            source_cut = source.transpose(1, 0, 3, 2) * 2.0 - 1.0
            generated, residuals = [], []
            for start in range(0, source_cut.shape[0], opt.batch_size):
                batch = torch.from_numpy(source_cut[start:start + opt.batch_size]).to(model.device)
                residual = model.netG(batch)
                fake = batch + residual if opt.use_residual_learning else residual
                generated.append(fake.cpu().numpy())
                residuals.append(residual.cpu().numpy())
            fake_czyx = np.concatenate(generated, axis=0).transpose(1, 0, 3, 2)
            residual_czyx = np.concatenate(residuals, axis=0).transpose(1, 0, 3, 2)
            stem = f'sample_{number:02d}_{identifier}'
            _save_nifti(output_root / f'{stem}_real_A.nii.gz', (source * 2.0 - 1.0) * 1000.0)
            _save_nifti(output_root / f'{stem}_fake_B.nii.gz', fake_czyx * 1000.0)
            _save_nifti(output_root / f'{stem}_residual_B.nii.gz', residual_czyx * 1000.0)
            print(f'[nifti] saved {identifier} at epoch {epoch}', flush=True)
    model.netG.train()


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        save_miccai_nifti_samples(model, opt, epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
