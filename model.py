"""pytorch Modules for UVAENet

Contains building blocks for UVAENet.

ResBlock: convolutional additive skip connection
Encoder: emits bottleneck layer for input images
VAEDecoder: samples from bottleneck to create latent distribution, reconstructs image
SemanticDecoder: upsamples bottleneck into softmax classes
UVAENet: constructor for creating full network
"""
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        return F.interpolate(x, **self.kwargs)


class GroupNorm3D(nn.Module):
    def __init__(self, num_features, num_groups=16, eps=1e-5):
        super(GroupNorm3D, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W, D = x.size()

        G = self.num_groups

        if C % G != 0:
            while C % G != 0 and G > 0:
                G -= 1
            print(
                "Warning: a GroupNorm3D operation was requested num_groups {} but had to use {} instead".format(
                    self.num_groups, G
                )
            )
            self.num_groups = G

        x = x.view(N, G, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W, D)
        return x * self.weight + self.bias


class ResBlock(nn.Module):
    def __init__(
        self,
        block_channels,
        kernel_size=3,
        activation=nn.LeakyReLU(0.2),
        regularization=nn.GroupNorm,
        n_groups=16,
        imdim=2,
    ):
        super().__init__()
        self.channels = block_channels
        self.kernel = kernel_size
        self.n_groups = n_groups
        if imdim == 2:
            Conv = nn.Conv2d
        elif imdim == 3:
            Conv = nn.Conv3d
            regularization = GroupNorm3D
        else:
            raise ValueError("imdim must be 2 or 3")
        self.padding = kernel_size - 2

        self.activation = activation

        self.regularization1 = regularization(
            num_features=self.channels, num_groups=self.n_groups,
        )
        self.conv1 = Conv(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel,
            padding=self.padding,
            bias=True,
        )

        self.regularization2 = regularization(
            num_features=self.channels, num_groups=self.n_groups,
        )
        self.conv2 = Conv(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel,
            padding=self.padding,
            bias=True,
        )

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
    def __init__(
        self,
        n_levels=4,
        input_channels=1,
        initial_channels=32,
        blocks_per_level=None,
        kernel_size=3,
        activation=nn.LeakyReLU(0.2),
        regularization=nn.GroupNorm,
        n_groups=1,
        imdim=2,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.activation = activation
        self.kernel_size = kernel_size
        self.regularization = regularization
        self.n_groups = n_groups
        if blocks_per_level is None:
            self.blocks_per_level = [1] + [2] * (n_levels - 2) + [4]
        if imdim == 2:
            Conv = nn.Conv2d
        elif imdim == 3:
            Conv = nn.Conv3d
        else:
            raise ValueError("imdim must be 2 or 3")

        self.sequence = nn.ModuleList()

        self.initial_conv = Conv(
            in_channels=input_channels,
            out_channels=initial_channels,
            kernel_size=self.kernel_size,
            padding=kernel_size - 2,
        )

        n_channels = initial_channels
        for i in range(n_levels):
            for j in range(self.blocks_per_level[i]):
                resblock = ResBlock(
                    block_channels=n_channels,
                    kernel_size=3,
                    activation=self.activation,
                    regularization=self.regularization,
                    n_groups=self.n_groups,
                    imdim=imdim,
                )
                self.sequence.append(resblock)
            if i == n_levels - 1:
                break
            self.sequence[-1].is_skip_connection = True
            strided_conv = Conv(
                in_channels=n_channels,
                out_channels=2 * n_channels,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.kernel_size - 2,
            )
            n_channels = 2 * n_channels
            self.sequence.append(strided_conv)

    def forward(self, X):
        # must save final blocks of each subsampling layer
        skip_connections = []
        out = self.initial_conv(X)
        for layer in self.sequence:
            out = layer(out)
            if hasattr(layer, "is_skip_connection"):
                skip_connections.append(out)
        return out, skip_connections


class VAEDecoder(nn.Module):
    def __init__(
        self,
        n_levels=3,
        input_shape=(10, 10),
        input_channels=256,
        output_channels=1,
        initial_channels=16,
        latent_size=128,
        blocks_per_level=None,
        kernel_size=3,
        activation=nn.LeakyReLU(0.2),
        regularization=nn.GroupNorm,
        n_groups=1,
        imdim=2,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.activation = activation
        self.regularization = regularization
        self.latent_size = latent_size
        self.input_shape = input_shape
        self.n_groups = n_groups
        if blocks_per_level is None:
            blocks_per_level = [1] * n_levels
        if imdim == 2:
            Conv = nn.Conv2d
            interp_mode = "bilinear"
        elif imdim == 3:
            Conv = nn.Conv3d
            interp_mode = "trilinear"
        else:
            raise ValueError("imdim must be 2 or 3")

        self.sequence = nn.ModuleList()
        self.group_norm = nn.GroupNorm(
            num_groups=self.n_groups, num_channels=input_channels
        )
        self.initial_conv = Conv(
            in_channels=input_channels,
            out_channels=initial_channels,
            kernel_size=self.kernel_size,
            padding=kernel_size - 2,
            stride=2,
        )
        shape = self.initial_conv(
            torch.zeros(1, input_channels, *input_shape, requires_grad=False)
        ).shape

        self.dense_to_latent = nn.Linear(
            in_features=np.product(shape), out_features=2 * self.latent_size,
        )
        self.dense = nn.Linear(
            in_features=self.latent_size,
            out_features=initial_channels * np.product(input_shape) // (2 ** imdim),
        )
        self.second_conv = Conv(
            in_channels=initial_channels,
            out_channels=input_channels,
            kernel_size=self.kernel_size,
            padding=kernel_size - 2,
        )
        self.upsample = Interpolate(
            mode=interp_mode, align_corners=True, scale_factor=2
        )
        n_channels = 2 * self.latent_size
        for i in range(n_levels):
            conv1 = Conv(
                in_channels=n_channels, out_channels=n_channels // 2, kernel_size=1
            )
            self.sequence.append(conv1)
            self.sequence.append(self.upsample)
            n_channels = n_channels // 2
            for j in range(blocks_per_level[i]):
                resblock = ResBlock(
                    block_channels=n_channels,
                    kernel_size=3,
                    activation=self.activation,
                    regularization=self.regularization,
                    n_groups=self.n_groups,
                    imdim=imdim,
                )
                self.sequence.append(resblock)
        self.final_conv = Conv(
            in_channels=n_channels,
            out_channels=output_channels,
            kernel_size=self.kernel_size,
            padding=kernel_size - 2,
        )
        self.sequence.append(self.final_conv)

    def sample(self, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std

    def forward(self, Z):
        out = self.group_norm(Z)
        out = self.activation(out)
        iconv = self.initial_conv(out)
        latent_dist = self.dense_to_latent(torch.flatten(iconv, start_dim=1))
        mu = latent_dist[:, : self.latent_size]
        logvar = latent_dist[:, self.latent_size :]
        out = mu + self.sample(logvar)
        out = self.dense(out)
        out = self.activation(out)
        batch_size = out.shape[0]
        out = out.view(batch_size, -1, *(i // 2 for i in self.input_shape))
        out = self.second_conv(out)
        out = self.upsample(out)
        for layer in self.sequence:
            out = layer(out)
        return out, mu, logvar


class SemanticDecoder(nn.Module):
    def __init__(
        self,
        n_levels=3,
        input_channels=256,
        output_channels=1,
        blocks_per_level=None,
        kernel_size=3,
        activation=nn.LeakyReLU(0.2),
        regularization=nn.GroupNorm,
        n_groups=1,
        imdim=2,
        final_activation="sigmoid",
    ):
        super().__init__()
        if final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif final_activation == "softmax":
            self.final_activation = nn.Softmax(dim=1)
        else:
            raise ValueError("activation %s not supported" % final_activation)
        self.kernel_size = kernel_size
        self.activation = activation
        self.regularization = regularization
        self.n_groups = n_groups
        if blocks_per_level is None:
            blocks_per_level = [1] * n_levels
        if imdim == 2:
            Conv = nn.Conv2d
            interp_mode = "bilinear"
        elif imdim == 3:
            Conv = nn.Conv3d
            interp_mode = "trilinear"
        else:
            raise ValueError("imdim must be 2 or 3")

        self.sequence = nn.ModuleList()
        self.upsample = Interpolate(
            mode=interp_mode, align_corners=True, scale_factor=2
        )
        n_channels = input_channels
        for i in range(n_levels):
            conv1 = Conv(
                in_channels=n_channels, out_channels=n_channels // 2, kernel_size=1
            )
            self.sequence.append(conv1)
            self.sequence.append(self.upsample)
            self.sequence[-1].is_skip_connection = True
            n_channels = n_channels // 2
            for j in range(blocks_per_level[i]):
                resblock = ResBlock(
                    block_channels=n_channels,
                    kernel_size=3,
                    activation=self.activation,
                    regularization=self.regularization,
                    n_groups=self.n_groups,
                    imdim=imdim,
                )
                self.sequence.append(resblock)
            self.final_conv = Conv(
                in_channels=n_channels,
                out_channels=output_channels,
                kernel_size=self.kernel_size,
                padding=kernel_size - 2,
            )

    def forward(self, Z, skip_connections):
        out = Z
        for layer in self.sequence:
            out = layer(out)
            if hasattr(layer, "is_skip_connection"):
                out = out + skip_connections.pop()
        out = self.final_conv(out)
        out = self.final_activation(out)
        return out


class VVAENet(nn.Module):
    def __init__(
        self,
        input_shape,
        encoder_config=None,
        vae_config=None,
        semantic_config=None,
        global_config=None,
    ):
        super().__init__()
        if global_config:
            if encoder_config:
                encoder_config.update(global_config)
            if vae_config:
                vae_config.update(global_config)
            if semantic_config:
                semantic_config.update(global_config)
        input_shape = (1,) + tuple(input_shape)
        if encoder_config is None:
            encoder_config = {}
        if vae_config is None:
            vae_config = {}
        if semantic_config is None:
            semantic_config = {}
        self.encoder = Encoder(**encoder_config)
        self.dropout = nn.Dropout3d(p=0.2)
        bottleneck, *_ = self.encoder(
            torch.zeros(input_shape, device=next(self.parameters()).device)
        )
        vae_config["input_shape"] = bottleneck.shape[2:]
        vae_config["output_channels"] = input_shape[1]
        self.vae_decoder = VAEDecoder(**vae_config)
        self.semantic_decoder = SemanticDecoder(**semantic_config)

    def forward(self, X):
        Z, skips = self.encoder(X)
        Z = self.dropout(Z)
        Y, mu, logvar = self.vae_decoder(Z)
        S = self.semantic_decoder(Z, skips)
        # P = torch.argmax(S, dim=1)
        return S, Y, mu, logvar


class ParallelVVAENet(nn.Module):
    def __init__(
        self,
        input_shape,
        encoder_config=None,
        vae_config=None,
        semantic_config=None,
        global_config=None,
    ):
        super().__init__()
        if global_config:
            if encoder_config:
                encoder_config.update(global_config)
            if vae_config:
                vae_config.update(global_config)
            if semantic_config:
                semantic_config.update(global_config)
        input_shape = (1,) + tuple(input_shape)
        if encoder_config is None:
            encoder_config = {}
        if vae_config is None:
            vae_config = {}
        if semantic_config is None:
            semantic_config = {}
        self.encoder = Encoder(**encoder_config).to("cuda:0")
        self.dropout = nn.Dropout3d(p=0.2)
        bottleneck, *_ = self.encoder(
            torch.zeros(input_shape, device=next(self.parameters()).device)
        )
        vae_config["input_shape"] = bottleneck.shape[2:]
        vae_config["output_channels"] = input_shape[1]
        self.vae_decoder = VAEDecoder(**vae_config).to("cuda:1")
        self.semantic_decoder = SemanticDecoder(**semantic_config).to("cuda:2")

    def forward(self, X):
        Z, skips = self.encoder(X.to("cuda:0"))
        skips = [s.to("cuda:2") for s in skips]
        Z = self.dropout(Z)
        Y, mu, logvar = self.vae_decoder(Z.to("cuda:1"))
        S = self.semantic_decoder(Z.to("cuda:2"), skips)
        # P = torch.argmax(S, dim=1)
        return S.to("cuda:0"), Y.to("cuda:0"), mu.to("cuda:0"), logvar.to("cuda:0")


class VNet(nn.Module):
    """
    This is not VNet from Milletari et. al.  It is VVAENet minus the variational autoencoder.
    """

    def __init__(
        self,
        input_shape,
        encoder_config=None,
        semantic_config=None,
        global_config=None,
        **_,  # permit vae_config kwarg to be passed in.
    ):
        super().__init__()
        if global_config:
            if encoder_config:
                encoder_config.update(global_config)
            if semantic_config:
                semantic_config.update(global_config)
        input_shape = (1,) + tuple(input_shape)
        if encoder_config is None:
            encoder_config = {}
        if semantic_config is None:
            semantic_config = {}
        self.encoder = Encoder(**encoder_config)
        self.dropout = nn.Dropout3d(p=0.2)
        self.semantic_decoder = SemanticDecoder(**semantic_config)

    def forward(self, X):
        Z, skips = self.encoder(X)
        Z = self.dropout(Z)
        S = self.semantic_decoder(Z, skips)
        return S


class VNetParallel(VNet):
    """
    This is not VNet from Milletari et. al.  It is VVAENet minus the variational autoencoder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = self.encoder.to("cuda:0")
        self.semantic_decoder = self.semantic_decoder.to("cuda:1")

    def forward(self, X):
        X.to("cuda:0")
        Z, skips = self.encoder(X)
        Z = Z.to("cuda:1")
        skips = [s.to("cuda:1") for s in skips]
        Z = self.dropout(Z)
        S = self.semantic_decoder(Z, skips)
        return S.to("cuda:0")

