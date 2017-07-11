import torch

from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


# Module for residual/skip connections
class FCResBlock(nn.Module):
    def __init__(self, dim, n, nonlinearity, batch_norm=True):
        """

        :param dim:
        :param n:
        :param nonlinearity:
        """
        super(FCResBlock, self).__init__()
        self.n = n
        self.nonlinearity = nonlinearity
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.block = nn.ModuleList(
                [nn.ModuleList([nn.Linear(dim, dim), nn.BatchNorm1d(num_features=dim)])
                 for _ in range(self.n)]
            )
        else:
            self.block = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n)])

    def forward(self, x):
        e = x + 0

        if self.batch_norm:
            for i, pair in enumerate(self.block):
                fc, bn = pair
                e = fc(e)
                e = bn(e)
                if i < (self.n - 1):
                    e = self.nonlinearity(e)

        else:
            for i, layer in enumerate(self.block):
                e = layer(e)
                if i < (self.n - 1):
                    e = self.nonlinearity(e)

        return self.nonlinearity(e + x)


# Building block for convolutional encoder with same padding
class Conv2d3x3(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(Conv2d3x3, self).__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=1, stride=stride)

    def forward(self, x):
        return self.conv(x)


# SHARED CONVOLUTIONAL ENCODER
class SharedConvolutionalEncoder(nn.Module):
    def __init__(self, nonlinearity):
        super(SharedConvolutionalEncoder, self).__init__()
        self.nonlinearity = nonlinearity

        self.conv_layers = nn.ModuleList([
            Conv2d3x3(in_channels=3, out_channels=32),
            Conv2d3x3(in_channels=32, out_channels=32),
            Conv2d3x3(in_channels=32, out_channels=32, downsample=True),
            # shape is now (-1, 32, 32, 32)
            Conv2d3x3(in_channels=32, out_channels=64),
            Conv2d3x3(in_channels=64, out_channels=64),
            Conv2d3x3(in_channels=64, out_channels=64, downsample=True),
            # shape is now (-1, 64, 16, 16)
            Conv2d3x3(in_channels=64, out_channels=128),
            Conv2d3x3(in_channels=128, out_channels=128),
            Conv2d3x3(in_channels=128, out_channels=128, downsample=True),
            # shape is now (-1, 128, 8, 8)
            Conv2d3x3(in_channels=128, out_channels=256),
            Conv2d3x3(in_channels=256, out_channels=256),
            Conv2d3x3(in_channels=256, out_channels=256, downsample=True)
            # shape is now (-1, 256, 4, 4)
        ])

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(num_features=32),
            nn.BatchNorm2d(num_features=32),
            nn.BatchNorm2d(num_features=32),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
        ])

    def forward(self, x):
        h = x.view(-1, 3, 64, 64)
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            h = conv(h)
            h = bn(h)
            h = self.nonlinearity(h)
        return h


# PRE-POOLING FOR STATISTIC NETWORK
class PrePool(nn.Module):
    """

    """

    def __init__(self, batch_size, n_features, n_hidden, hidden_dim, nonlinearity):
        super(PrePool, self).__init__()
        self.batch_size = batch_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc = nn.Linear(self.n_features, self.hidden_dim)
        self.bn = nn.BatchNorm1d(self.hidden_dim)

    def forward(self, h):
        # reshape and affine
        e = h.view(-1, self.n_features)
        e = self.fc(e)
        e = self.bn(e)
        e = self.nonlinearity(e)

        return e


# POST POOLING FOR STATISTIC NETWORK
class PostPool(nn.Module):
    """

    """

    def __init__(self, n_hidden, hidden_dim, c_dim, nonlinearity):
        super(PostPool, self).__init__()
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim),
                                        nn.Linear(self.hidden_dim, self.hidden_dim)])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(self.hidden_dim),
                                        nn.BatchNorm1d(self.hidden_dim)])

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.c_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, e):
        for fc, bn in zip(self.fc_layers, self.bn_layers):
            e = fc(e)
            e = bn(e)
            e = self.nonlinearity(e)

        # affine transformation to parameters
        e = self.fc_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.c_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.c_dim)

        mean, logvar = e[:, :self.c_dim], e[:, self.c_dim:]

        return mean, logvar


# STATISTIC NETWORK q(c|D)
class StatisticNetwork(nn.Module):
    """

    """

    def __init__(self, batch_size, sample_size, n_features,
                 n_hidden, hidden_dim, c_dim, nonlinearity):
        super(StatisticNetwork, self).__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.nonlinearity = nonlinearity

        # modules
        self.prepool = PrePool(self.batch_size, self.n_features,
                               self.n_hidden, self.hidden_dim, self.nonlinearity)
        self.postpool = PostPool(self.n_hidden, self.hidden_dim,
                                 self.c_dim, self.nonlinearity)

    def forward(self, h):
        e = self.prepool(h)
        e = self.pool(e)
        e = self.postpool(e)
        return e

    def pool(self, e):
        e = e.view(self.batch_size, self.sample_size, self.hidden_dim)
        e = e.mean(1).view(self.batch_size, self.hidden_dim)
        return e


class InferenceNetwork(nn.Module):
    """
    Inference network q(z|h, z, c) gives approximate posterior over latent variables.
    """
    def __init__(self, batch_size, sample_size, n_features,
                 n_hidden, hidden_dim, c_dim, z_dim, nonlinearity):
        super(InferenceNetwork, self).__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.z_dim = z_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_h = nn.Linear(self.n_features, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim)

        self.fc_res_block = FCResBlock(dim=self.hidden_dim, n=self.n_hidden,
                                       nonlinearity=self.nonlinearity, batch_norm=True)

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, h, z, c):
        # combine h, z, and c
        # embed h
        eh = h.view(-1, self.n_features)
        eh = self.fc_h(eh)
        eh = eh.view(self.batch_size, self.sample_size, self.hidden_dim)

        # embed z if we have more than one stochastic layer
        if z is not None:
            ez = z.view(-1, self.z_dim)
            ez = self.fc_z(ez)
            ez = ez.view(self.batch_size, self.sample_size, self.hidden_dim)
        else:
            ez = Variable(torch.zeros(eh.size()).cuda())

        # embed c and expand for broadcast addition
        ec = self.fc_c(c)
        ec = ec.view(self.batch_size, 1, self.hidden_dim).expand_as(eh)

        # sum and reshape
        e = eh + ez + ec
        e = e.view(self.batch_size * self.sample_size, self.hidden_dim)
        e = self.nonlinearity(e)

        # for layer in self.fc_block:
        e = self.fc_res_block(e)

        # affine transformation to parameters
        e = self.fc_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.z_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim)

        mean, logvar = e[:, :self.z_dim].contiguous(), e[:, self.z_dim:].contiguous()

        return mean, logvar


# LATENT DECODER p(z|z, c)
class LatentDecoder(nn.Module):
    """

    """

    def __init__(self, batch_size, sample_size, n_features,
                 n_hidden, hidden_dim, c_dim, z_dim, nonlinearity):
        super(LatentDecoder, self).__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.z_dim = z_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim)

        self.fc_res_block = FCResBlock(dim=self.hidden_dim, n=self.n_hidden,
                                       nonlinearity=self.nonlinearity, batch_norm=True)

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, z, c):
        # combine z and c
        # embed z if we have more than one stochastic layer
        if z is not None:
            ez = z.view(-1, self.z_dim)
            ez = self.fc_z(ez)
            ez = ez.view(self.batch_size, self.sample_size, self.hidden_dim)
        else:
            ez = Variable(torch.zeros(self.batch_size, 1, self.hidden_dim).cuda())

        # embed c and expand for broadcast addition
        ec = self.fc_c(c)
        ec = ec.view(self.batch_size, 1, self.hidden_dim).expand_as(ez)

        # sum and reshape
        e = ez + ec
        e = e.view(-1, self.hidden_dim)
        e = self.nonlinearity(e)

        # for layer in self.fc_block:
        e = self.fc_res_block(e)

        # affine transformation to parameters
        e = self.fc_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.z_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim)

        mean, logvar = e[:, :self.z_dim].contiguous(), e[:, self.z_dim:].contiguous()

        return mean, logvar


# Observation Decoder p(x|z, c)
class ObservationDecoder(nn.Module):
    """

    """
    def __init__(self, batch_size, sample_size, n_features,
                 n_hidden, hidden_dim, c_dim, n_stochastic, z_dim,
                 nonlinearity):
        super(ObservationDecoder, self).__init__()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.n_stochastic = n_stochastic
        self.z_dim = z_dim

        self.nonlinearity = nonlinearity

        # shared learnable log variance parameter
        self.logvar = nn.Parameter(torch.randn(1, 3, 64, 64).cuda())

        # modules
        self.fc_zs = nn.Linear(self.n_stochastic * self.z_dim, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)

        self.fc_initial = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_linear = nn.Linear(self.hidden_dim, self.n_features)

        self.conv_layers = nn.ModuleList([
            Conv2d3x3(in_channels=256, out_channels=256),
            Conv2d3x3(in_channels=256, out_channels=256),
            nn.ConvTranspose2d(in_channels=256, out_channels=256,
                               kernel_size=2, stride=2),
            Conv2d3x3(in_channels=256, out_channels=128),
            Conv2d3x3(in_channels=128, out_channels=128),
            nn.ConvTranspose2d(in_channels=128, out_channels=128,
                               kernel_size=2, stride=2),
            Conv2d3x3(in_channels=128, out_channels=64),
            Conv2d3x3(in_channels=64, out_channels=64),
            nn.ConvTranspose2d(in_channels=64, out_channels=64,
                               kernel_size=2, stride=2),
            Conv2d3x3(in_channels=64, out_channels=32),
            Conv2d3x3(in_channels=32, out_channels=32),
            nn.ConvTranspose2d(in_channels=32, out_channels=32,
                               kernel_size=2, stride=2)
        ])

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=256),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=128),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=64),
            nn.BatchNorm2d(num_features=32),
            nn.BatchNorm2d(num_features=32),
            nn.BatchNorm2d(num_features=32),
        ])

        self.conv_mean = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, zs, c):
        ezs = self.fc_zs(zs)
        ezs = ezs.view(self.batch_size, self.sample_size, self.hidden_dim)

        ec = self.fc_c(c)
        ec = ec.view(self.batch_size, 1, self.hidden_dim).expand_as(ezs)

        e = ezs + ec
        e = self.nonlinearity(e)
        e = e.view(-1, self.hidden_dim)

        e = self.fc_initial(e)
        e = self.nonlinearity(e)
        e = self.fc_linear(e)
        e = e.view(-1, 256, 4, 4)

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            e = conv(e)
            e = bn(e)
            e = self.nonlinearity(e)

        mean = self.conv_mean(e)
        mean = F.sigmoid(mean)

        return mean, self.logvar.expand_as(mean)