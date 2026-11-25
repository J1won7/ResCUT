"""Train the CUT-derived model without changing its optimizer update order."""
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from model.data import create_dataset
from model.models import create_model
from model.options.train_options import TrainOptions
from model.samples import record_run, save_samples


def main():
    opt = TrainOptions().parse()
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = create_dataset(opt)
    record_run(opt, dataset.dataset)
    model = create_model(opt)
    initialized = False
    total_iters = 0
    log = Path(opt.checkpoints_dir)/opt.name/'losses.jsonl'
    for epoch in range(opt.epoch_count, opt.n_epochs+opt.n_epochs_decay+1):
        started = time.monotonic()
        dataset.set_epoch(epoch)
        for data in dataset:
            opt.batch_size = data['A'].size(0)
            total_iters += opt.batch_size
            if not initialized:
                model.data_dependent_initialize(data)
                model.setup(opt)
                model.parallelize()
                initialized = True
            model.set_input(data)
            model.optimize_parameters()
            if total_iters % opt.print_freq == 0:
                row = dict(epoch=epoch, images_seen=total_iters, **model.get_current_losses())
                with log.open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                print(row, flush=True)
            if total_iters % opt.save_latest_freq == 0:
                model.save_networks('latest')
        if epoch % opt.save_epoch_freq == 0:
            model.save_networks('latest')
            model.save_networks(epoch)
        if epoch % 2 == 0:
            save_samples(model, dataset.dataset, opt, epoch)
        model.update_learning_rate()
        print(f'Epoch {epoch}/{opt.n_epochs+opt.n_epochs_decay}: {time.monotonic()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
