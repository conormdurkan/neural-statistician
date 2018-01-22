import argparse
import os
import time

from spatialdata import SpatialMNISTDataset
from spatialmodel import Statistician
from spatialplot import grid
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
parser.add_argument('--c-dim', type=int, default=64,
                    help='dimension of c variables (default: 64)')
parser.add_argument('--n-hidden-statistic', type=int, default=3,
                    help='number of hidden layers in statistic network modules '
                         '(default: 3)')
parser.add_argument('--hidden-dim-statistic', type=int, default=256,
                    help='dimension of hidden layers in statistic network (default: 256)')
parser.add_argument('--n-stochastic', type=int, default=3,
                    help='number of z variables in hierarchy (default: 3)')
parser.add_argument('--z-dim', type=int, default=2,
                    help='dimension of z variables (default: 2)')
parser.add_argument('--n-hidden', type=int, default=3,
                    help='number of hidden layers in modules outside statistic network '
                         '(default: 3)')
parser.add_argument('--hidden-dim', type=int, default=256,
                    help='dimension of hidden layers in modules outside statistic network '
                         '(default: 256)')
parser.add_argument('--print-vars', type=bool, default=False,
                    help='whether to print all trainable parameters for sanity check '
                         '(default: False)')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='learning rate for Adam optimizer (default: 1e-3).')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs for training (default: 100)')
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
assert args.output_dir is not None
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

    # initial weighting for loss terms is (1 + alpha)
    alpha = 1

    # main training loop
    tbar = tqdm(range(args.epochs))
    for epoch in tbar:

        # train step
        model.train()
        running_vlb = 0
        for batch in train_loader:
            inputs = Variable(batch[0].cuda())
            vlb = model.step(inputs, alpha, optimizer, clip_gradients=args.clip_gradients)
            running_vlb += vlb

        running_vlb /= (len(train_dataset) // args.batch_size)
        s = "VLB: {:.3f}".format(running_vlb)
        tbar.set_description(s)

        # reduce weight
        alpha *= 0.5

        # show samples conditioned on test batch at intervals
        model.eval()
        if (epoch + 1) % viz_interval == 0:
            inputs = Variable(test_batch[0].cuda(), volatile=True)
            samples = model.sample_conditioned(inputs)
            filename = time_stamp + '-{}.png'.format(epoch + 1)
            save_path = os.path.join(args.output_dir, 'figures/' + filename)
            grid(inputs, samples, save_path=save_path, ncols=10)

        # checkpoint model at intervals
        if (epoch + 1) % save_interval == 0:
            filename = time_stamp + '-{}.m'.format(epoch + 1)
            save_path = os.path.join(args.output_dir, 'checkpoints/' + filename)
            model.save(optimizer, save_path)

    # we're already in eval mode, but let's be explicit
    model.eval()
    # summarize test batch at end of training
    n = 10  # number of datasets to summarize
    inputs = Variable(test_batch[0].cuda(), volatile=True)
    print("Summarizing...")
    summaries = model.summarize_batch(inputs[:n], output_size=6)
    print("Summary complete!")

    # plot summarized datasets
    samples = model.sample_conditioned(inputs)
    filename = time_stamp + '-summary.png'
    save_path = os.path.join(args.output_dir, 'figures/' + filename)
    grid(inputs, samples, summaries=summaries, save_path=save_path, ncols=n)


def main():
    train_dataset = SpatialMNISTDataset(data_dir=args.data_dir, split='train')
    test_dataset = SpatialMNISTDataset(data_dir=args.data_dir, split='test')
    datasets = (train_dataset, test_dataset)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0, drop_last=True)

    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, drop_last=True)
    loaders = (train_loader, test_loader)

    # hardcoded sample_size and n_features when making Spatial MNIST dataset
    sample_size = 50
    n_features = 2
    model_kwargs = {
        'batch_size': args.batch_size,
        'sample_size': sample_size,
        'n_features': n_features,
        'c_dim': args.c_dim,
        'n_hidden_statistic': args.n_hidden_statistic,
        'hidden_dim_statistic': args.hidden_dim_statistic,
        'n_stochastic': args.n_stochastic,
        'z_dim': args.z_dim,
        'n_hidden': args.n_hidden,
        'hidden_dim': args.hidden_dim,
        'nonlinearity': F.relu,
        'print_vars': args.print_vars
    }
    model = Statistician(**model_kwargs)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    run(model, optimizer, loaders, datasets)


if __name__ == '__main__':
    main()
