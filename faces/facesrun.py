import argparse
import os
import time

from facesdata import YouTubeFacesSetsDataset
from facesmodel import Statistician
from facesplot import save_test_grid
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm

# command line args
parser = argparse.ArgumentParser(description='Neural Statistician Synthetic Experiment')

# required
parser.add_argument('--data-dir', required=True, type=str, default=None,
                    help='location of formatted Omniglot data')
parser.add_argument('--output-dir', required=True, type=str, default=None,
                    help='output directory for checkpoints and figures')

# optional
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size (of datasets) for training (default: 64)')
parser.add_argument('--sample-size', type=int, default=5,
                    help='number of sample images per dataset (default: 5)')
parser.add_argument('--c-dim', type=int, default=512,
                    help='dimension of c variables (default: 512)')
parser.add_argument('--n-hidden-statistic', type=int, default=1,
                    help='number of hidden layers in statistic network modules '
                         '(default: 1)')
parser.add_argument('--hidden-dim-statistic', type=int, default=1000,
                    help='dimension of hidden layers in statistic network (default: 1000)')
parser.add_argument('--n-stochastic', type=int, default=1,
                    help='number of z variables in hierarchy (default: 1)')
parser.add_argument('--z-dim', type=int, default=16,
                    help='dimension of z variables (default: 16)')
parser.add_argument('--n-hidden', type=int, default=1,
                    help='number of hidden layers in modules outside statistic network '
                         '(default: 1)')
parser.add_argument('--hidden-dim', type=int, default=1000,
                    help='dimension of hidden layers in modules outside statistic network '
                         '(default: 1000)')
parser.add_argument('--print-vars', type=bool, default=False,
                    help='whether to print all trainable parameters for sanity check '
                         '(default: False)')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='learning rate for Adam optimizer (default: 1e-3).')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs for training (default: 300)')
parser.add_argument('--viz-interval', type=int, default=-1,
                    help='number of epochs between visualizing context space '
                         '(default: -1 (only visualize last epoch))')
parser.add_argument('--save_interval', type=int, default=-1,
                    help='number of epochs between saving model '
                         '(default: -1 (save on last epoch))')
parser.add_argument('--clip-gradients', type=bool, default=True,
                    help='whether to clip gradients to range [-0.5, 0.5] '
                         '(default: True)')
args = parser.parse_args()
assert (args.data_dir is not None) and (args.output_dir is not None)
os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

# experiment start time
time_stamp = time.strftime("%d-%m-%Y-%H:%M:%S")


def run(model, optimizer, loaders, datasets):
    train_dataset, test_dataset = datasets
    train_loader, test_loader = loaders
    test_batch = next(iter(test_loader))

    viz_interval = args.epochs if args.viz_interval == -1 else args.viz_interval
    save_interval = args.epochs if args.save_interval == -1 else args.save_interval

    # initial weighting for two term loss
    alpha = 1
    # main training loop
    tbar = tqdm(range(args.epochs))
    for epoch in tbar:

        # train step (iterate once over training data)
        model.train()
        running_vlb = 0
        for batch in train_loader:
            inputs = Variable(batch.cuda())
            vlb = model.step(inputs, alpha, optimizer, clip_gradients=args.clip_gradients)
            running_vlb += vlb

        # update running lower bound
        running_vlb /= (len(train_dataset) // args.batch_size)
        s = "VLB: {:.3f}".format(running_vlb)
        tbar.set_description(s)

        # reduce weight
        alpha *= 0.5

        # evaluate on test set by sampling conditioned on contexts
        model.eval()
        if (epoch + 1) % 1 == 0:
            filename = time_stamp + '-grid-{}.png'.format(epoch + 1)
            save_path = os.path.join(args.output_dir, 'figures/' + filename)
            inputs = Variable(test_batch.cuda(), volatile=True)
            samples = model.sample_conditioned(inputs)
            save_test_grid(inputs, samples, save_path)

        # checkpoint model at intervals
        if (epoch + 1) % save_interval == 0:
            filename = time_stamp + '-{}.m'.format(epoch + 1)
            save_path = os.path.join(args.output_dir, '/checkpoints/' + filename)
            model.save(optimizer, save_path)


def main():
    # create datasets
    train_dataset = YouTubeFacesSetsDataset(data_dir=args.data_dir, split='train',
                                            n_frames_per_set=5)
    test_dataset = YouTubeFacesSetsDataset(data_dir=args.data_dir, split='valid',
                                           n_frames_per_set=5)
    datasets = (train_dataset, test_dataset)

    # create loaders
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=4, drop_last=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, drop_last=True)
    loaders = (train_loader, test_loader)

    # create model
    n_features = 256 * 4 * 4  # output shape of convolutional encoder
    model_kwargs = {
        'batch_size': args.batch_size,
        'sample_size': args.sample_size,
        'n_features': n_features,
        'c_dim': args.c_dim,
        'n_hidden_statistic': args.n_hidden_statistic,
        'hidden_dim_statistic': args.hidden_dim_statistic,
        'n_stochastic': args.n_stochastic,
        'z_dim': args.z_dim,
        'n_hidden': args.n_hidden,
        'hidden_dim': args.hidden_dim,
        'nonlinearity': F.elu,
        'print_vars': args.print_vars
    }
    model = Statistician(**model_kwargs)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    run(model, optimizer, loaders, datasets)


if __name__ == '__main__':
    main()
