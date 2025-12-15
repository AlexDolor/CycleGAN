# CSC 321, Assignment 4
#
# This is the main training file for the CycleGAN part of the assignment.
#
# Usage:
# ======
#    To train with the default hyperparamters (saves results to samples_cyclegan/):
#       python cycle_gan.py
#
#    To train with cycle consistency loss (saves results to samples_cyclegan_cycle/):
#       python cycle_gan.py --use_cycle_consistency_loss
#
#
#    For optional experimentation:
#    -----------------------------
#    If you have a powerful computer (ideally with a GPU), then you can obtain better results by
#    increasing the number of filters used in the generator and/or discriminator, as follows:
#      python cycle_gan.py --g_conv_dim=64 --d_conv_dim=64

import os
import pdb
import pickle
import csv
# import argparse

import warnings
warnings.filterwarnings("ignore")

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Numpy & Matplotlib imports
import numpy as np
# import scipy
# import scipy.misc
import matplotlib.pyplot as plt
# import imageio

from tqdm import tqdm

# Local imports
import utils
from data_loader import get_loaders
from models import CycleGenerator, DCDiscriminator
from parser import create_parser, print_opts


SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


def print_models(G_XtoY, G_YtoX, D_X, D_Y):
    """Prints model information for the generators and discriminators.
    """
    print("                 G_XtoY                ")
    print("---------------------------------------")
    print(G_XtoY)
    print("---------------------------------------")

    print("                 G_YtoX                ")
    print("---------------------------------------")
    print(G_YtoX)
    print("---------------------------------------")

    print("                  D_X                  ")
    print("---------------------------------------")
    print(D_X)
    print("---------------------------------------")

    print("                  D_Y                  ")
    print("---------------------------------------")
    print(D_Y)
    print("---------------------------------------")


def create_model(opts):
    """Builds the generators and discriminators.
    """
    G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    G_YtoX = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    D_X = DCDiscriminator(conv_dim=opts.d_conv_dim)
    D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)

    print_models(G_XtoY, G_YtoX, D_X, D_Y)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to mps.')
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU.')

    return G_XtoY, G_YtoX, D_X, D_Y


def checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts):
    """Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_X, D_Y.
    """
    filenames = {
        'G_XtoY': 'G_XtoY.pkl',
        'G_YtoX': 'G_YtoX.pkl',
        'D_X': 'D_X.pkl', 
        'D_Y': 'D_Y.pkl'
        }
    if iteration % opts.secure_checkpoint_every == 0:
        for name, file in filenames.items():
            filenames[name] = name + f'_{iteration}.pkl' 
    G_XtoY_path = os.path.join(opts.checkpoint_dir, filenames['G_XtoY'])
    G_YtoX_path = os.path.join(opts.checkpoint_dir, filenames['G_YtoX'])
    D_X_path = os.path.join(opts.checkpoint_dir, filenames['D_X'])
    D_Y_path = os.path.join(opts.checkpoint_dir, filenames['D_Y'])
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)


def load_checkpoint(opts):
    """Loads the generator and discriminator models from checkpoints.
    """
    G_XtoY_path = os.path.join(opts.load, 'G_XtoY.pkl')
    G_YtoX_path = os.path.join(opts.load, 'G_YtoX.pkl')
    D_X_path = os.path.join(opts.load, 'D_X.pkl')
    D_Y_path = os.path.join(opts.load, 'D_Y.pkl')

    G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    G_YtoX = CycleGenerator(conv_dim=opts.g_conv_dim, init_zero_weights=opts.init_zero_weights)
    D_X = DCDiscriminator(conv_dim=opts.d_conv_dim)
    D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)

    G_XtoY.load_state_dict(torch.load(G_XtoY_path, map_location=lambda storage, loc: storage))
    G_YtoX.load_state_dict(torch.load(G_YtoX_path, map_location=lambda storage, loc: storage))
    D_X.load_state_dict(torch.load(D_X_path, map_location=lambda storage, loc: storage))
    D_Y.load_state_dict(torch.load(D_Y_path, map_location=lambda storage, loc: storage))

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to mps.')
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU.')

    return G_XtoY, G_YtoX, D_X, D_Y


def merge_images(sources, targets, opts, k=10):
    """Creates a grid consisting of pairs of columns, where the first column in
    each pair contains images source images and the second column in each pair
    contains images generated by the CycleGAN from the corresponding images in
    the first column.
    """
    _, _, h, w = sources.shape
    row = int(np.sqrt(opts.batch_size))
    merged = np.zeros([3, row*h, row*w*2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
        merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
    merged = merged.transpose(1, 2, 0)
    # normalize image for saving
    merged = merged - merged.min()
    merged = merged/merged.max()
    return merged


def save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts):
    """Saves samples from both generators X->Y and Y->X.
    """
    fake_X = G_YtoX(fixed_Y)
    fake_Y = G_XtoY(fixed_X)

    X, fake_X = utils.to_data(fixed_X), utils.to_data(fake_X)
    Y, fake_Y = utils.to_data(fixed_Y), utils.to_data(fake_Y)

    merged = merge_images(X, fake_Y, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}-X-Y.png'.format(iteration))
    # scipy.misc.imsave(path, merged)
    # print(f'169, {merged.max(), merged.min()}')
    # imageio.imwrite(path, merged)
    plt.imsave(path, merged)
    print('Saved {}'.format(path))

    merged = merge_images(Y, fake_X, opts)
    path = os.path.join(opts.sample_dir, 'sample-{:06d}-Y-X.png'.format(iteration))
    # scipy.misc.imsave(path, merged)
    plt.imsave(path, merged)
    print('Saved {}'.format(path))

def calc_discr_loss(loss, img, target):
    '''casts target to img.shape to calculate loss'''
    if target:
        target = torch.ones_like(img).float()
    else:
        target = torch.zeros_like(img).float()
    return loss(img, target)

def lambda_rule(epoch):
    '''rule to update lr in optimizers'''
    if epoch <= 100:
        return 1.0
    else:
        return (1.0 - (float(epoch-100) / 100))

def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    if opts.load:
        G_XtoY, G_YtoX, D_X, D_Y = load_checkpoint(opts)
    else:
        G_XtoY, G_YtoX, D_X, D_Y = create_model(opts)

    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters
    d_params = list(D_X.parameters()) + list(D_Y.parameters())  # Get discriminator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr, [opts.beta1, opts.beta2])

    # Create schedulers with custom rule
    g_scheduler = optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lambda_rule)
    d_scheduler = optim.lr_scheduler.LambdaLR(d_optimizer, lr_lambda=lambda_rule)

    #create criterion for the generators and discriminators
    mse_criterion = nn.MSELoss()
    cycle_criterion = torch.nn.L1Loss()

    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    #create imagepool for discriminators
    X_fake_pool = utils.ImagePool(50)
    Y_fake_pool = utils.ImagePool(50)

    # make csv file for metrics
    metrics_f = open('metrics.csv', 'w', newline='', encoding='utf-8')
    metrics_writer = csv.writer(metrics_f)
    metrics_writer.writerow(['epoch', 'iteration', 'g_lr', 'd_lr', 'd_X_loss', 'd_Y_loss', 'g_loss'])

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = utils.to_device(next(test_iter_X)[0])
    fixed_Y = utils.to_device(next(test_iter_Y)[0])

    iter_per_epoch = min(len(iter_X), len(iter_Y))
    epoch = 1
    if opts.train_epochs > 0:
        opts.train_iters = opts.train_epochs * iter_per_epoch 

    for iteration in tqdm(range(1, opts.train_iters+1)):

        # Reset data_iter for each epoch
        if iteration % iter_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)
            epoch += 1
            g_scheduler.step()
            d_scheduler.step()

        images_X, labels_X = next(iter_X)
        images_X, labels_X = utils.to_device(images_X), utils.to_device(labels_X).long().squeeze()

        images_Y, labels_Y = next(iter_Y)
        images_Y, labels_Y = utils.to_device(images_Y), utils.to_device(labels_Y).long().squeeze()

        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================

        g_optimizer.zero_grad()

        fake_X = G_YtoX(images_Y)
        # print(f'282, {fake_X.shape=}')
        # assert 1 == 2
        fake_Y = G_XtoY(images_X)

        for p in D_X.parameters():
            p.requires_grad_(False)
        for p in D_Y.parameters():
            p.requires_grad_(False)

        
        D_X_fake = D_X(fake_X)
        D_Y_fake = D_Y(fake_Y)
        g_loss = calc_discr_loss(mse_criterion, D_X_fake, True)
        g_loss += calc_discr_loss(mse_criterion, D_Y_fake, True)

        if opts.use_cycle_consistency_loss:
            reconstructed_Y = G_XtoY(fake_X)
            cycle_consistency_loss = cycle_criterion(reconstructed_Y, images_Y) * opts.cycle_lambda
            g_loss += cycle_consistency_loss

            reconstructed_X = G_YtoX(fake_Y)
            cycle_consistency_loss = cycle_criterion(reconstructed_X, images_X) * opts.cycle_lambda
            g_loss += cycle_consistency_loss

        g_loss.backward()
        g_optimizer.step()

        for p in D_X.parameters():
            p.requires_grad_(True)
        for p in D_Y.parameters():
            p.requires_grad_(True)

        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        # UPDATE D_X
        d_optimizer.zero_grad()
        
        D_X_real = D_X(images_X)
        # print(f'236, {images_X.shape=}, {labels_X.shape=}, {D_X_real.shape=}, {torch.ones_like(labels_X).shape=}')
        # assert 1==2
        D_X_loss_real = calc_discr_loss(mse_criterion, D_X_real, True)
        
        # with torch.no_grad():
        #     fake_X = G_YtoX(images_Y)
        fake_X = X_fake_pool.query(fake_X.detach())
        D_X_fake = D_X(fake_X)
        D_X_loss_fake = calc_discr_loss(mse_criterion, D_X_fake, False)
        
        d_X_loss = (D_X_loss_real + D_X_loss_fake) * 0.5
        d_X_loss.backward()
        d_optimizer.step()

        # UPDATE D_Y
        d_optimizer.zero_grad()

        D_Y_real = D_Y(images_Y)
        D_Y_loss_real = calc_discr_loss(mse_criterion, D_Y_real, True)

        # with torch.no_grad():
        #     fake_Y = G_XtoY(images_X)
        fake_Y = Y_fake_pool.query(fake_Y.detach())
        D_Y_fake = D_Y(fake_Y)
        D_Y_loss_fake = calc_discr_loss(mse_criterion, D_Y_fake, False)

        d_Y_loss = (D_Y_loss_real + D_Y_loss_fake) * 0.5
        d_Y_loss.backward()
        d_optimizer.step()

        # Print the log info
        if iteration % opts.log_step == 0:
            g_lr = g_scheduler.get_last_lr()[0]
            d_lr = d_scheduler.get_last_lr()[0]
            log = (
                f'\nIteration [{iteration:5d}/{opts.train_iters:5d}] | Current epoch [{epoch}] '
                f'current lr: g_optim_lr: {g_lr:0.5f} d_optim_lr: {d_lr:0.5f} '
                f'| d_X_loss: {d_X_loss.item():6.4f} | d_Y_loss: {d_Y_loss.item():6.4f} | g_loss: {g_loss.item():6.4f}'
            )
            print(log)

            metrics_writer.writerow([
                epoch, iteration,
                g_lr, d_lr,
                d_X_loss.item(), d_Y_loss.item(), g_loss.item()
            ])
            metrics_f.flush()

        # Save the generated samples
        if iteration % opts.sample_every == 0:
            save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts)


        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts)

        # Secure save the model paremeters
        if iteration % opts.secure_checkpoint_every == 0:
            checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts)
    metrics_f.close()
    
def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create train and test dataloaders for images from the two domains X and Y
    dataloader_X, test_dataloader_X = get_loaders(img_type=opts.X, opts=opts)
    dataloader_Y, test_dataloader_Y = get_loaders(img_type=opts.Y, opts=opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    # Start training
    training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts)




if __name__ == '__main__':

    parser = create_parser()
    # opts = parser.parse_args()
    opts, _ = parser.parse_known_args()

    # if opts.use_cycle_consistency_loss:
    #     opts.sample_dir = 'samples_cyclegan_cycle'

    # if opts.load:
    #     opts.sample_dir = '{}_pretrained'.format(opts.sample_dir)
    #     opts.sample_every = 20

    print_opts(opts)
    main(opts)
