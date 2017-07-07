import argparse
import socket
import time

from synthdata import SyntheticSetsDataset
from synthmodel import Statistician
from synthplot import scatter_contexts, contexts_by_moment
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm

host = socket.gethostname()
if host == 'coldingham':
    output_dir = '/home/conor/Dropbox/msc/thesis/ns/ns-pytorch/synthetic' \
                        '/output/'
else:
    output_dir = '/disk/scratch/conor/output/ns/ns-pytorch/synthetic/'


parser = argparse.ArgumentParser(description='Neural Statistician Synthetic Experiment')
parser.add_argument('--n-datasets', type=int, default=10000, metavar='N',
                    help='number of synthetic datasets in collection (default: 10000)')
parser.add_argument('--batch-size', type=int, default=16,
                    help='batch size (of datasets) for training (default: 16)')
parser.add_argument('--sample-size', type=int, default=200,
                    help='number of samples per dataset (default: 200)')
parser.add_argument('--n-features', type=int, default=1,
                    help='number of features per sample (default: 1)')
parser.add_argument('--distributions', type=str, default='easy',
                    help='which distributions to use for synthetic data '
                         '(easy: (Gaussian, Uniform, Laplacian, Exponential), '
                         'hard: (Bimodal mixture of Gaussians, Laplacian, '
                         'Exponential, Reverse Exponential) '
                         '(default: easy)')
parser.add_argument('--c-dim', type=int, default=3,
                    help='dimension of c variables (default: 3)')
parser.add_argument('--n-hidden-statistic', type=int, default=3,
                    help='number of hidden layers in statistic network modules '
                         '(default: 3)')
parser.add_argument('--hidden-dim-statistic', type=int, default=128,
                    help='dimension of hidden layers in statistic network (default: 128)')
parser.add_argument('--n-stochastic', type=int, default=1,
                    help='number of z variables in hierarchy (default: 1)')
parser.add_argument('--z-dim', type=int, default=32,
                    help='dimension of z variables (default: 32)')
parser.add_argument('--n-hidden', type=int, default=3,
                    help='number of hidden layers in modules outside statistic network '
                         '(default: 10000)')
parser.add_argument('--hidden-dim', type=int, default=128,
                    help='dimension of hidden layers in modules outside statistic network '
                         '(default: 10000)')
parser.add_argument('--print-vars', type=bool, default=False,
                    help='number of synthetic datasets in collection (default: False)')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='learning rate for Adam optimizer (default: 1e-3).')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs for training (default: 50)')
parser.add_argument('--viz-interval', type=int, default=-1,
                    help='number of epochs between visualizing context space '
                         '(default: -1 (only visualize last epoch))')
parser.add_argument('--save_interval', type=int, default=-1,
                    help='number of epochs between saving model '
                         '(default: -1 (save on last epoch))')
args = parser.parse_args()

time_stamp = time.strftime("%d-%m-%Y-%H:%M:%S")


def run(model, optimizer, loaders, datasets):
    train_loader, test_loader = loaders
    train_dataset, test_dataset = datasets

    viz_interval = args.epochs if args.viz_interval == -1 else args.viz_interval
    save_interval = args.epochs if args.save_interval == -1 else args.save_interval

    alpha = 1
    tbar = tqdm(range(args.epochs))
    # main training loop
    for epoch in tbar:

        # train step
        model.train()
        running_vlb = 0
        for batch in train_loader:
            vlb = model.step(batch, alpha, optimizer, clip_gradients=True)
            running_vlb += vlb

        running_vlb /= (len(train_dataset) // 16)
        s = "VLB: {:.3f}".format(running_vlb)
        tbar.set_description(s)

        # reduce weight
        alpha *= 0.5

        # show test set in context space at intervals
        if (epoch + 1) % viz_interval == 0:
            model.eval()
            contexts = []
            for batch in test_loader:
                inputs = Variable(batch.cuda(), volatile=True)
                context_means, _ = model.statistic_network(inputs)
                contexts.append(context_means.data.cpu().numpy())

            # show clusters
            path = output_dir + '/figures/' + time_stamp + '-{}.pdf'.format(epoch + 1)
            scatter_contexts(contexts, test_dataset.data['labels'],
                             test_dataset.data['distributions'], savepath=path)

            # show coloured by mean
            path = output_dir + '/figures/' + time_stamp \
                   + '-{}-mean.pdf'.format(epoch + 1)
            contexts_by_moment(contexts, moments=test_dataset.data['means'],
                               savepath=path)

            # show coloured by variance
            path = output_dir + '/figures/' + time_stamp \
                   + '-{}-variance.pdf'.format(epoch + 1)
            contexts_by_moment(contexts, moments=test_dataset.data['variances'],
                               savepath=path)

        # checkpoint model at intervals
        if (epoch + 1) % save_interval == 0:
            save_path = output_dir + '/checkpoints/' + time_stamp \
                        + '-{}.m'.format(epoch + 1)
            model.save(optimizer, save_path)


def main():
    if args.distributions == 'easy':
        distributions = [
            'gaussian',
            'uniform',
            'laplacian',
            'exponential'
        ]
    else:
        distributions = [
            'mixture of gaussians',
            'laplacian',
            'exponential',
            'reverse exponential'
        ]
    train_dataset = SyntheticSetsDataset(n_datasets=args.n_datasets,
                                         sample_size=args.sample_size,
                                         n_features=args.n_features,
                                         distributions=distributions)

    n_test_datasets = args.n_datasets // 10
    test_dataset = SyntheticSetsDataset(n_datasets=n_test_datasets,
                                        sample_size=args.sample_size,
                                        n_features=args.n_features,
                                        distributions=distributions)
    datasets = (train_dataset, test_dataset)

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=4, drop_last=True)

    test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4, drop_last=True)
    loaders = (train_loader, test_loader)

    model_kwargs = {
        'batch_size': args.batch_size,
        'sample_size': args.sample_size,
        'n_features': args.n_features,
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
