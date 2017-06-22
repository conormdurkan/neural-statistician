import torch

from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F, init
from utils import (kl_diagnormal_diagnormal, kl_diagnormal_stdnormal,
                   bernoulli_log_likelihood)


# Module for residual/skip connections
class FCResBlock(nn.Module):
    def __init__(self, dim, n, nonlinearity):
        """

        :param dim: 
        :param n: 
        :param nonlinearity: 
        """
        super(FCResBlock, self).__init__()
        self.n = n
        self.nonlinearity = nonlinearity
        self.block = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.n)])

    def forward(self, x):
        e = x + 0
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

    def forward(self, x):
        h = x.view(-1, 1, 28, 28)
        for layer in self.conv_layers:
            h = layer(h)
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

    def forward(self, h):
        # reshape and affine
        e = h.view(-1, self.n_features)
        e = self.fc(e)
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
        self.fc_block = nn.ModuleList([
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        ])

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.c_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, e):
        for layer in self.fc_block:
            e = layer(e)
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
                                       nonlinearity=self.nonlinearity)

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
                                       nonlinearity=self.nonlinearity)

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

        for layer in self.conv_layers:
            e = layer(e)
            e = self.nonlinearity(e)

        e = self.conv_final(e)
        e = F.sigmoid(e)

        return e


# Model
class Statistician(nn.Module):
    def __init__(self, batch_size=16, sample_size=200, n_features=1,
                 c_dim=3, n_hidden_statistic=128, hidden_dim_statistic=3,
                 n_stochastic=1, z_dim=16, n_hidden=3, hidden_dim=128,
                 nonlinearity=F.relu, print_vars=False):
        """

        :param batch_size: 
        :param sample_size: 
        :param n_features: 
        :param c_dim: 
        :param n_hidden_statistic: 
        :param hidden_dim_statistic: 
        :param n_stochastic: 
        :param z_dim: 
        :param n_hidden: 
        :param hidden_dim: 
        :param nonlinearity: 
        :param print_vars: 
        """
        super(Statistician, self).__init__()
        # data shape
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.n_features = n_features

        # context
        self.c_dim = c_dim
        self.n_hidden_statistic = n_hidden_statistic
        self.hidden_dim_statistic = hidden_dim_statistic

        # latent
        self.n_stochastic = n_stochastic
        self.z_dim = z_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim

        self.nonlinearity = nonlinearity

        # modules
        # convolutional encoder
        self.shared_convolutional_encoder = SharedConvolutionalEncoder(self.nonlinearity)

        # statistic network
        statistic_args = (self.batch_size, self.sample_size, self.n_features,
                          self.n_hidden_statistic, self.hidden_dim_statistic,
                          self.c_dim, self.nonlinearity)
        self.statistic_network = StatisticNetwork(*statistic_args)

        z_args = (self.batch_size, self.sample_size, self.n_features,
                  self.n_hidden, self.hidden_dim, self.c_dim, self.z_dim,
                  self.nonlinearity)
        # inference networks
        self.inference_networks = nn.ModuleList([InferenceNetwork(*z_args)
                                                 for _ in range(self.n_stochastic)])

        # latent decoders
        self.latent_decoders = nn.ModuleList([LatentDecoder(*z_args)
                                              for _ in range(self.n_stochastic)])

        # observation decoder
        observation_args = (self.batch_size, self.sample_size, self.n_features,
                            self.n_hidden, self.hidden_dim, self.c_dim,
                            self.n_stochastic, self.z_dim, self.nonlinearity)
        self.observation_decoder = ObservationDecoder(*observation_args)

        # initialize weights
        self.apply(self.weights_init)

        # print variables for sanity check and debugging
        if print_vars:
            for i, pair in enumerate(self.named_parameters()):
                name, param = pair
                print("{} --> {}, {}".format(i + 1, name, param.size()))
            print()

    def forward(self, x):
        # convolutional encoder
        h = self.shared_convolutional_encoder(x)

        # statistic network
        c_mean, c_logvar = self.statistic_network(h)
        if self.training:
            c = self.reparameterize_gaussian(c_mean, c_logvar)
        else:  # sampling conditioned on inputs
            c = c_mean

        # inference networks
        qz_samples = []
        qz_params = []
        z = None
        for inference_network in self.inference_networks:
            z_mean, z_logvar = inference_network(h, z, c)
            qz_params.append([z_mean, z_logvar])
            z = self.reparameterize_gaussian(z_mean, z_logvar)
            qz_samples.append(z)

        # latent decoders
        pz_params = []
        z = None
        for i, latent_decoder in enumerate(self.latent_decoders):
            z_mean, z_logvar = latent_decoder(z, c)
            pz_params.append([z_mean, z_logvar])
            z = qz_samples[i]

        # observation decoder
        zs = torch.cat(qz_samples, dim=1)
        x_mean = self.observation_decoder(zs, c)

        outputs = (
            (c_mean, c_logvar),
            (qz_params, pz_params),
            (x, x_mean)
        )

        return outputs

    def loss(self, outputs, weight):
        c_outputs, z_outputs, x_outputs = outputs

        # 1. Reconstruction loss
        x, x_mean = x_outputs
        recon_loss = bernoulli_log_likelihood(x, x_mean)
        recon_loss /= (self.batch_size * self.sample_size)

        # 2. KL Divergence terms
        kl = 0

        # a) Context divergence
        c_mean, c_logvar = c_outputs
        kl_c = kl_diagnormal_stdnormal(c_mean, c_logvar)
        kl += kl_c

        # b) Latent divergences
        qz_params, pz_params = z_outputs
        shapes = (
            (self.batch_size, self.sample_size, self.z_dim),
            (self.batch_size, 1, self.z_dim)
        )
        for i in range(self.n_stochastic):
            args = (qz_params[i][0].view(shapes[0]),
                    qz_params[i][1].view(shapes[0]),
                    pz_params[i][0].view(shapes[1] if i == 0 else shapes[0]),
                    pz_params[i][1].view(shapes[1] if i == 0 else shapes[0]))
            kl_z = kl_diagnormal_diagnormal(*args)
            kl += kl_z

        kl /= (self.batch_size * self.sample_size)

        # Variational lower bound and weighted loss
        vlb = recon_loss - kl
        loss = - ((weight * recon_loss) - (kl / weight))

        return loss, vlb

    def step(self, batch, alpha, optimizer, clip_gradients=True):
        assert self.training is True

        inputs = Variable(batch.cuda())
        outputs = self.forward(inputs)
        loss, vlb = self.loss(outputs, weight=(alpha + 1))

        # perform gradient update
        optimizer.zero_grad()
        loss.backward()
        if clip_gradients:
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)
        optimizer.step()

        # output variational lower bound
        return vlb.data[0]

    def sample_conditioned(self, batch):
        assert self.training is False

        inputs = Variable(batch.cuda())
        outputs = self.forward(inputs)
        samples = outputs[-1][1]

        return inputs, samples

    def save(self, optimizer, path):
        torch.save({
            'model_state': self.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, path)

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = Variable(torch.randn(std.size()).cuda())
        return mean + std * eps

    @staticmethod
    def weights_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.xavier_normal(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass
