import torch

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


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
            Conv2d3x3(in_channels=1, out_channels=64),
            Conv2d3x3(in_channels=64, out_channels=64),
            Conv2d3x3(in_channels=64, out_channels=64, downsample=True),
            # shape is now (-1, 64, 14 , 14)
            Conv2d3x3(in_channels=64, out_channels=128),
            Conv2d3x3(in_channels=128, out_channels=128),
            Conv2d3x3(in_channels=128, out_channels=128, downsample=True),
            # shape is now (-1, 128, 7, 7)
            Conv2d3x3(in_channels=128, out_channels=256),
            Conv2d3x3(in_channels=256, out_channels=256),
            Conv2d3x3(in_channels=256, out_channels=256, downsample=True)
            # shape is now (-1, 256, 4, 4)
        ])

        self.bn_layers = nn.ModuleList([
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
        h = x.view(-1, 1, 28, 28)
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
        # e = self.sample_dropout(e)
        e = self.pool(e)
        e = self.postpool(e)
        return e

    def pool(self, e):
        e = e.view(self.batch_size, self.sample_size, self.hidden_dim)
        e = e.mean(1).view(self.batch_size, self.hidden_dim)
        return e

    def sample_dropout(self, e):
        # create mask
        a = Variable(torch.ones((self.batch_size, 1, 1)).cuda())
        p = 0.5 if self.training else 1
        b = Variable(torch.bernoulli(p * torch.ones((self.batch_size,
                                                     self.sample_size - 1, 1)).cuda()))
        mask = torch.cat([a, b], 1)

        # zero out samples
        e = e.view(self.batch_size, self.sample_size, self.hidden_dim)
        e = e * mask.expand_as(e)

        # take mean across sample dimension
        extra_feature = torch.sum(mask, 1)
        e = torch.sum(e, 1)
        e /= extra_feature.expand_as(e)

        # add number of retained samples as extra feature
        e = torch.cat([e, extra_feature], 2).squeeze(1)

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

        # modules
        self.fc_zs = nn.Linear(self.n_stochastic * self.z_dim, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)

        self.fc_initial = nn.Linear(self.hidden_dim, 256 * 4 * 4)

        self.conv_layers = nn.ModuleList([
            Conv2d3x3(256, 256),
            Conv2d3x3(256, 256),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            Conv2d3x3(256, 128),
            Conv2d3x3(128, 128),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            Conv2d3x3(128, 64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        ])
        
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
        ])

        self.conv_final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, zs, c):
        # concatenate zs and c
        ezs = self.fc_zs(zs)
        ezs = ezs.view(self.batch_size, self.sample_size, self.hidden_dim)

        ec = self.fc_c(c)
        ec = ec.view(self.batch_size, 1, self.hidden_dim).expand_as(ezs)

        e = ezs + ec
        e = self.nonlinearity(e)
        e = e.view(-1, self.hidden_dim)

        e = self.fc_initial(e)
        e = e.view(-1, self.hidden_dim, 4, 4)

        for conv, bn in zip(self.conv_layers, self.bn_layers):
            e = conv(e)
            e = bn(e)
            e = self.nonlinearity(e)

        e = self.conv_final(e)
        e = F.sigmoid(e)

        return e