import argparse

def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=128, help='The side length N to convert images to NxN.')
    parser.add_argument('--g_conv_dim', type=int, default=32)
    parser.add_argument('--d_conv_dim', type=int, default=32)
    parser.add_argument('--use_cycle_consistency_loss', action='store_true', default=True, help='Choose whether to include the cycle consistency term in the loss.')
    parser.add_argument('--init_zero_weights', action='store_true', default=False, help='Choose whether to initialize the generator conv weights to 0 (implements the identity function).')

    # Training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=14000, help='The number of training iterations to run (you can Ctrl-C out earlier if you want).')
    parser.add_argument('--train_epochs', type=int, default=None, help='Alternative way of specifing training iters. Epoch size is equal to minimal train set length. Will override --train_iters')
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate (default 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--cycle_lambda', type=float, default=10, help='parameter for cycle loss')

    # Data sources
    parser.add_argument('--X', type=str, default='A', choices=['A', 'B'], help='Choose the type of images for domain X.')
    parser.add_argument('--Y', type=str, default='B', choices=['A', 'B'], help='Choose the type of images for domain Y.')

    # Saving directories and checkpoint/sample iterations
    parser.add_argument('--data_dir', type=str, default='data\horse2zebra')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_cyclegan')
    parser.add_argument('--sample_dir', type=str, default='samples_cyclegan')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--log_step', type=int , default=100)
    parser.add_argument('--sample_every', type=int , default=500)
    parser.add_argument('--checkpoint_every', type=int , default=500)
    parser.add_argument('--secure_checkpoint_every', type=int , default=1000, help='Saves that would not be overwritten')

    return parser