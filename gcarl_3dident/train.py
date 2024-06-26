"""GCaRL training"""


from datetime import datetime
import os.path
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
import faiss

from gcarl_3dident import gcarl
from gcarl_3dident.utils import find_device
from subfunc.threedident_dataset import RawThreeDIdentDataset
from subfunc.showdata import *


# =============================================================
# =============================================================
def train(s,
          num_hz_nodes,
          num_hp_nodes,
          initial_learning_rate,
          momentum,
          max_steps,
          decay_steps,
          decay_factor,
          batch_size,
          train_dir,
          data_dir,
          weight_decay=0,
          phi_type='maxout',
          moving_average_decay=0.999,
          summary_steps=500,
          checkpoint_steps=10000,
          save_file='model.pt',
          load_file=None,
          device=None,
          random_seed=None):
    """Build and train a model
    Args:
        s: data [data, group, dim] (without observational mixing)
        num_hz_nodes:
        num_hp_nodes:
        initial_learning_rate: initial learning rate
        momentum: momentum parameter
        max_steps: number of iterations (mini-batches)
        decay_steps: decay steps
        decay_factor: decay factor
        batch_size: mini-batch size
        train_dir: save directory
        data_dir: 3dIdent data directory
        weight_decay: weight decay
        phi_type: model type of phi (needs to be consistent with the source model)
        moving_average_decay: (option) moving average decay of variables to be saved
        summary_steps: (option) interval to save summary
        checkpoint_steps: (option) interval to save checkpoint
        save_file: (option) name of model file to save
        load_file: (option) name of model file to load
        device: device to be used
        random_seed: (option) random seed
    Returns:
    """

    # set random_seed
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    num_data, num_group, num_dim = s.shape

    # load image-factors and assign an image to each sample
    z = np.load(os.path.join(data_dir, "raw_latents.npy"))
    s_range = np.percentile(s.reshape(-1), [0.01, 99.99])
    s_norm = 2 * ((s - s_range[0]) / (s_range[1] - s_range[0]) - 0.5)
    index_fn = faiss.IndexFlatL2(z.shape[1])
    index_fn.add(z)
    distance_s_to_z, index_s_to_z = index_fn.search(s_norm.reshape([-1, s_norm.shape[-1]]), 1)
    index_s_to_z = index_s_to_z.reshape([s_norm.shape[0], s_norm.shape[1]])

    # load whole images
    dataset = RawThreeDIdentDataset(root=data_dir,
                                    transform=transforms.Compose(
                                        [
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                mean=[0.4363, 0.2818, 0.3045], std=[0.1197, 0.0734, 0.0919]  # Yao2023
                                                # mean=[0.3292, 0.3278, 0.3215], std=[0.0778, 0.0776, 0.0771]  # CL-ICA
                                            ),
                                        ]))
    data_loader = DataLoader(dataset,
                             batch_size=int(2**10),
                             shuffle=False,
                             drop_last=False,
                             num_workers=0)
    x_img = torch.zeros([len(z), 3, 224, 224], device='cpu')
    for step, batch in enumerate(data_loader):
        sys.stdout.write('\rLoading x... %d / %d' % (step * data_loader.batch_size, len(z)))
        sys.stdout.flush()
        idx = batch[1].detach().cpu().numpy()
        x_img[idx, :] = batch[0]
    sys.stdout.write('\r\n')

    # define network
    model = gcarl.Net(num_group=num_group,
                      num_hdim=num_dim,
                      hz_sizes=num_hz_nodes if num_hz_nodes is not None else [num_dim],
                      hp_sizes=num_hp_nodes,
                      phi_type=phi_type)
    if device is None:
        device = find_device()
    model = model.to(device)
    model.train()

    # define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum, weight_decay=weight_decay)

    if type(decay_steps) == list:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_steps, gamma=decay_factor)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=decay_factor)
    writer = SummaryWriter(log_dir=train_dir)

    state_dict_ema = model.state_dict()

    trained_step = 0
    if load_file is not None:
        print('Load trainable parameters from %s...' % load_file)
        checkpoint = torch.load(load_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trained_step = checkpoint['step']
        state_dict_ema = checkpoint['ema_state_dict']

    # training iteration
    for step in range(trained_step, max_steps):
        start_time = time.time()

        idx = np.random.choice(num_data, batch_size)
        x0 = torch.stack([x_img[index_s_to_z[idx, m], :] for m in range(num_group)], dim=1)
        xast = x0.clone()
        fix_group = np.random.randint(num_group)  # single group has the same value to x0
        for mi, m in enumerate(np.setdiff1d(np.arange(num_group), fix_group)):
            idx_ast = np.random.choice(num_data, batch_size)
            xast[:, m, :] = x_img[index_s_to_z[idx_ast, m], :]

        x_torch = torch.cat([x0, xast], dim=0).to(device)
        y_torch = torch.cat([torch.ones([batch_size]), torch.zeros([batch_size])]).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits, h, phi = model(x_torch)
        loss = criterion(logits, y_torch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if phi_type.startswith('lap'):
            model.w.data = model.w.data.clamp(max=0)

        # moving average of parameters
        state_dict_n = model.state_dict()
        for key in state_dict_ema:
            state_dict_ema[key] = moving_average_decay * state_dict_ema[key] \
                                  + (1.0 - moving_average_decay) * state_dict_n[key]

        # accuracy
        predicted = (logits > 0.0).float()
        accu_val = (predicted == y_torch).sum().item()/(batch_size*2)
        loss_val = loss.item()
        lr = scheduler.get_last_lr()[0]

        duration = time.time() - start_time

        assert not np.isnan(loss_val), 'Model diverged with loss = NaN'

        # display stats
        if step % 100 == 0:
            num_examples_per_step = batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            format_str = '%s: step %d, lr = %f, loss = %.2f, accuracy = %3.2f (%.1f examples/sec; %.3f sec/batch)'
            print(format_str % (datetime.now(), step, lr, loss_val, accu_val * 100,
                                examples_per_sec, sec_per_batch))

        # save summary
        if step % summary_steps == 0:
            writer.add_scalar('scalar/lr', lr, step)
            writer.add_scalar('scalar/loss', loss_val, step)
            writer.add_scalar('scalar/accu', accu_val, step)
            h_val = h.cpu().detach().numpy()
            h_comp = np.split(h_val, indices_or_sections=h.shape[1], axis=1)
            for (i, cm) in enumerate(h_comp):
                writer.add_histogram('h/h%d' % i, cm)
            for k, v in state_dict_n.items():
                writer.add_histogram('w/%s' % k, v)

        # save the model checkpoint periodically.
        if step % checkpoint_steps == 0:
            checkpoint_path = os.path.join(train_dir, save_file)
            torch.save({'step': step,
                        'model_state_dict': model.state_dict(),
                        'ema_state_dict': state_dict_ema,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()}, checkpoint_path)

    # save trained model
    save_path = os.path.join(train_dir, save_file)
    print('Save model in file: %s' % save_path)
    torch.save({'step': max_steps,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': state_dict_ema,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()}, save_path)
