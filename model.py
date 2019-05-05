"""pytorch Modules for UVAENet

Contains building blocks for UVAENet.

ResBlock: convolutional additive skip connection
Encoder: emits bottleneck layer for input images
VAEDecoder: samples from bottleneck to create latent distribution, reconstructs image
SemanticDecoder: upsamples bottleneck into softmax classes
UVAENet: constructor for creating full network
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, block_channels, kernel_size=3, activation=nn.ReLU,
                 regularization=nn.GroupNorm, n_groups=1, imdim=2):
        super().__init__()
        self.channels = block_channels
        self.kernel = kernel_size
        self.n_groups = n_groups
        if imdim == 2:
            Conv = nn.Conv2d
        elif imdim == 3:
            Conv = nn.Conv3d
        else:
            raise ValueError('imdim must be 2 or 3')
        self.padding = kernel_size - 2

        self.activation = activation

        self.regularization1 = regularization(num_groups=self.n_groups,
                                              num_channels=self.channels)
        self.conv1 = Conv(in_channels=self.channels,
                          out_channels=self.channels,
                          kernel_size=self.kernel,
                          padding=self.padding,
                          bias=True)

        self.regularization2 = regularization(num_groups=self.n_groups,
                                              num_channels=self.channels)
        self.conv2 = Conv(in_channels=self.channels,
                          out_channels=self.channels,
                          kernel_size=self.kernel,
                          padding=self.padding,
                          bias=True)

    def forward(self, input):
        xi = input
        layer = self.regularization1(xi)
        layer = self.activation(layer)
        layer = self.conv1(layer)
        layer = self.regularization2(layer)
        layer = self.activation(layer)
        Fxi = self.conv2(layer)
        out = xi + Fxi
        return out


class Encoder(nn.Module):

    def __init__(self, n_levels=4, input_channels=1, initial_channels=16,
                 blocks_per_level=None, kernel_size=3, activation=nn.ReLU,
                 regularization=nn.GroupNorm, n_groups=1, imdim=2):
        super().__init__()
        self.n_levels = n_levels
        self.activation = activation
        self.kernel_size = kernel_size
        self.regularization = regularization
        self.n_groups = 1
        if blocks_per_level is None:
            self.blocks_per_level = [1] + [2] * (n_levels - 2) + [4]
        if imdim == 2:
            Conv = nn.Conv2d
        elif imdim == 3:
            Conv = nn.Conv3d
        else:
            raise ValueError('imdim must be 2 or 3')

        self.sequence = []

        self.initial_conv = Conv(in_channels=input_channels,
                                 out_channels=initial_channels,
                                 kernel_size=self.kernel_size,
                                 padding=kernel_size - 2)

        n_channels = initial_channels
        for i in range(n_levels):
            for j in range(self.blocks_per_level[i]):
                resblock = ResBlock(block_channels=n_channels,
                                    kernel_size=3,
                                    activation=self.activation,
                                    regularization=self.regularization,
                                    n_groups=self.n_groups)
                self.sequence.append(resblock)
            if i == n_levels - 1:
                break
            self.sequence.append('skip_connect')
            strided_conv = Conv(in_channels=n_channels,
                                out_channels=2 * n_channels,
                                stride=2)
            n_channels = 2 * n_channels
            self.sequence.append(strided_conv)

    def forward(self, X):
        # must save final blocks of each subsampling layer
        skip_connections = []
        out = self.initial_conv(X)
        for layer in self.sequence:
            if layer == 'skip_connect':
                skip_connections.append(out)
                continue
            out = layer(out)
        return out, skip_connections


class VAEDecoder(nn.Module):

    def __init__(self, n_levels=4, input_volume=10, input_channels=256,
                 output_channels=1, initial_channels=16, latent_size = 128,
                 blocks_per_level=None, kernel_size=3, activation=nn.ReLU,
                 regularization=nn.GroupNorm, n_groups=1, imdim=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = activation
        self.regularization = regularization
        self.latent_size = latent_size
        self.n_groups = n_groups
        if blocks_per_level is None:
            blocks_per_level = [1] * n_levels
        if imdim == 2:
            Conv = nn.Conv2d
        elif imdim == 3:
            Conv = nn.Conv3d
        else:
            raise ValueError('imdim must be 2 or 3')

        self.sequence = []
        self.group_norm = nn.GroupNorm(num_groups=self.n_groups)
        self.initial_conv = Conv(in_channels=input_channels,
                                 out_channels=initial_channels,
                                 kernel_size=self.kernel_size,
                                 padding=kernel_size - 2)
        self.dense = nn.Linear(in_features=input_volume // 2,
                               out_features=2 * self.latent_size)

        self.upsample = partial(F.interpolate, mode='bilinear',
                                align_corners=True)
        n_channels = initial_channels
        for i in range(n_levels):
            conv1 = Conv(in_channels=n_channels,
                         out_channels=n_channels // 2,
                         kernel_size=1)
            self.sequence.append(conv1)
            self.sequence.append(self.upsample)
            n_channels = 2 * n_channels
            for j in range(blocks_per_level[i]):
                resblock = ResBlock(block_channels=n_channels,
                                    kernel_size=3,
                                    activation=self.activation,
                                    regularization=self.regularization,
                                    n_groups=self.n_groups)
                self.sequence.append(resblock)
            self.final_conv = Conv(in_channels=n_channels,
                                   out_channels=output_channels,
                                   kernel_size=self.kernel_size,
                                   padding=kernel_size - 2)
            self.sequence.append(self.final_conv)

    def sample(self, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std

    def forward(self, Z):
        out = self.group_norm(Z)
        out = self.activation(out)
        iconv = self.initial_conv(out)
        latent_dist = self.dense(iconv.view(-1))
        mu = latent_dist[:self.latent]
        logvar = latent_dist[self.latent_size:]
        out = mu + self.sample(logvar)
        for layer in self.sequence:
            out = layer(out)
        return out, mu, logvar


class SemanticDecoder(nn.Module):

    def __init__(self, n_levels=3, input_volume=10, input_channels=256,
                 output_channels=1,
                 blocks_per_level=None, kernel_size=3, activation=nn.ReLU,
                 regularization=nn.GroupNorm, imdim=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = activation
        self.regularization = regularization
        if blocks_per_level is None:
            blocks_per_level = [1] * n_levels
        if imdim == 2:
            Conv = nn.Conv2d
        elif imdim == 3:
            Conv = nn.Conv3d
        else:
            raise ValueError('imdim must be 2 or 3')

        self.sequence = []
        self.upsample = partial(F.interpolate, mode='bilinear',
                                align_corners=True)
        n_channels = input_channels
        for i in range(n_levels):
            conv1 = Conv(in_channels=n_channels,
                         out_channels=n_channels // 2,
                         kernel_size=1)
            self.sequence.append(conv1)
            self.sequence.append(self.upsample)
            self.sequence.append('skip_connect')
            n_channels = 2 * n_channels
            for j in range(blocks_per_level[i]):
                resblock = ResBlock(block_channels=n_channels,
                                    kernel_size=3,
                                    activation=self.activation,
                                    regularization=self.regularization,
                                    n_groups=self.n_groups)
                self.sequence.append(resblock)
            self.final_conv = Conv(in_channels=n_channels,
                                   out_channels=output_channels,
                                   kernel_size=self.kernel_size,
                                   padding=kernel_size - 2)

    def forward(self, Z, skip_connections):
        out = Z
        for layer in self.sequence:
            if layer == 'skip_connect':
                out = out + skip_connections.pop()
                continue
            out = layer(out)
        out = self.final_conv(out)
        out = F.softmax(out, dim=0)
        return out


class UVAENet(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.vae_decoder = VAEDecoder()
        self.semantic_decoder = SemanticDecoder()

    def forward(self, X):
        Z, skips = self.encoder(X)
        Y, latent = self.vae_decoder(Z)
        S = self.semantic_decoder(Z, skips)
        return S, (Y, latent)
