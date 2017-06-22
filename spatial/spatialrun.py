import torch

from spatialdata import SpatialMNISTDataset
from spatialmodel import Statistician
from spatialplot import grid
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm


def create_datasets():
    train_dataset = SpatialMNISTDataset(split='train')
    test_dataset = SpatialMNISTDataset(split='test')
    return train_dataset, test_dataset


def create_loaders(datasets, batch_size):
    train_dataset, test_dataset = datasets

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=4, drop_last=True)

    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=4, drop_last=True)

    return train_loader, test_loader


def create_model(kwargs):
    model = Statistician(**kwargs)
    model.cuda()
    return model


def train(model, optimizer, loaders, datasets, epochs, viz_interval=-1, save_interval=-1):
    train_loader, test_loader = loaders
    train_dataset, test_dataset = datasets

    viz_interval = epochs if viz_interval == -1 else viz_interval
    save_interval = epochs if save_interval == -1 else save_interval

    alpha = 1
    tbar = tqdm(range(epochs))
    # main training loop
    for epoch in tbar:

        # train step
        model.train()
        running_vlb = 0
        for batch in train_loader:
            inputs = Variable(batch[0].cuda())
            outputs = model(inputs)
            loss, vlb = model.loss(outputs, weight=(alpha + 1))

            running_vlb += vlb.data[0]

            # perform (clipped) gradient update
            optimizer.zero_grad()
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)
            optimizer.step()

        running_vlb /= (len(train_dataset) // 16)
        s = "VLB: {:.3f}".format(running_vlb)
        tbar.set_description(s)

        # reduce weight
        alpha *= 0.5

        # show test set in context space at intervals
        # if (epoch + 1) % viz_interval == 0:
        if True:
            model.eval()
            for batch in test_loader:
                inputs = Variable(batch[0].cuda())
                outputs = model(inputs)
                reconstructions = outputs[-1][1]
                grid(inputs, reconstructions.view(64, 50, 2), ncols=8)
                break

        # checkpoint model at intervals
        if (epoch + 1) % save_interval == 0:
            path = '/home/conor/Dropbox/msc/thesis/ns/ns-pytorch/spatial' \
                   '/output/checkpoints/{}.m'.format(epoch + 1)
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, path)


def main():
    batch_size = 64
    sample_size = 50
    n_features = 2

    datasets = create_datasets()
    loaders = create_loaders(datasets, batch_size)

    model_kwargs = {
        'batch_size': batch_size,
        'sample_size': sample_size,
        'n_features': n_features,
        'c_dim': 64,
        'n_hidden_statistic': 3,
        'hidden_dim_statistic': 256,
        'n_stochastic': 3,
        'z_dim': 2,
        'n_hidden': 3,
        'hidden_dim': 256,
        'nonlinearity': F.relu,
        'print_vars': False
    }
    model = create_model(model_kwargs)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    restore = False
    if restore:
        # assume we're restoring just for visualization purposes
        path = 'home/conor/Dropbox/msc/thesis/ns/ns-pytorch/spatial/' \
               'output/checkpoints/50.m'
        state = torch.load(path)
        model.load_state_dict(state['model_state'])
    else:
        epochs = 50
        train(model, optimizer, loaders, datasets, epochs,
              viz_interval=-1, save_interval=-1)


if __name__ == '__main__':
    main()
