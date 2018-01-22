import os
import sys
import torch

from facesnets import (SharedConvolutionalEncoder, StatisticNetwork, InferenceNetwork,
                       LatentDecoder, ObservationDecoder)
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F, init
try:
    from utils import (kl_diagnormal_diagnormal, kl_diagnormal_stdnormal,
                       gaussian_log_likelihood)
except ModuleNotFoundError:
    # put parent directory in path for utils
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import (kl_diagnormal_diagnormal, kl_diagnormal_stdnormal,
                       gaussian_log_likelihood)


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
        x_mean, x_logvar = self.observation_decoder(zs, c)

        outputs = (
            (c_mean, c_logvar),
            (qz_params, pz_params),
            (x, x_mean, x_logvar)
        )

        return outputs

    def loss(self, outputs, weight):
        c_outputs, z_outputs, x_outputs = outputs

        # 1. Reconstruction loss
        x, x_mean, x_logvar = x_outputs
        recon_loss = gaussian_log_likelihood(x.view(-1, 3, 64, 64), x_mean, x_logvar, clip=True)
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

    def step(self, inputs, alpha, optimizer, clip_gradients=True):
        assert self.training is True

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

        # output variational lower bound for batch
        return vlb.data[0]

    def sample(self):
        c = torch.randn(self.batch_size, self.c_dim)

        # latent decoders
        pz_samples = []
        z = None
        for i, latent_decoder in enumerate(self.latent_decoders):
            z_mean, z_logvar = latent_decoder(z, c)
            z = self.reparameterize_gaussian(z_mean, z_logvar)
            pz_samples.append(z)

        # observation decoder
        zs = torch.cat(pz_samples, dim=1)
        x_mean, x_logvar = self.observation_decoder(zs, c)

        return x_mean

    def sample_conditioned(self, inputs):
        h = self.shared_convolutional_encoder(inputs)
        c, _ = self.statistic_network(h)

        # latent decoders
        pz_samples = []
        z = None
        for i, latent_decoder in enumerate(self.latent_decoders):
            z_mean, z_logvar = latent_decoder(z, c)
            if i == 0:
                z_mean = z_mean.repeat(self.sample_size, 1)
                z_logvar = z_logvar.repeat(self.sample_size, 1)
            z = self.reparameterize_gaussian(z_mean, z_logvar)
            pz_samples.append(z)

        # observation decoder
        zs = torch.cat(pz_samples, dim=1)
        x_mean, x_logvar = self.observation_decoder(zs, c)

        return x_mean

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
