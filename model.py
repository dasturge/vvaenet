import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, name, block_channels, kernel_size=3,
                 activation=nn.ReLU, regularization=nn.GroupNorm, imdim=2):
        super().__init__()
        self.name = name
        self.channels = block_channels
        self.kernel = kernel_size
        if imdim == 2:
            Conv = nn.Conv2d
        elif imdim == 3:
            Conv = nn.Conv3d
        self.padding = kernel_size - 2

        self.activation = activation

        self.regularization1 = regularization(num_groups=2, num_channels=self.channels)
        self.conv1 = Conv(in_channels=self.channels,
                          out_channels=self.channels,
                          kernel_size=self.kernel,
                          padding=self.padding,
                          bias=True)

        self.regularization2 = regularization(num_groups=2, num_channels=self.channels)
        self.conv2 = Conv(in_channels=self.channels,
                          out_channels=self.channels,
                          kernel_size=self.kernel,
                          padding=self.padding,
                          bias=True)

    def forward(self, input):
        xi = input
        layer = self.activation(xi)
        layer = self.regularization1(layer)
        layer = self.conv1(layer)
        layer = self.activation(layer)
        layer = self.regularization2(layer)
        F = self.conv2(layer)
        out = xi + F

        return out





class ModelSpec():

    def __init__(self):
        pass

    def transpose(self):
        pass


class Encoder(nn.Module):

    def __init__(self, n_blocks, input_channels=1, initial_channels=16,
                 blocks_per_level=(1, 2, 2, 4), kernel_size=3, activation=nn.ReLU,
                 regularization=nn.GroupNorm, imdim=2, **spec):
        super().__init__()
        self.spec = spec
        self.n_blocks = n_blocks
        self.activation = activation
        self.kernel_size = kernel_size
        self.regularization = regularization
        if imdim == 2:
            Conv = nn.Conv2d
        elif imdim == 3:
            Conv = nn.Conv3d

        self.sequence = []

        self.initial_conv = Conv(in_channels=input_channels,
                                 out_channels=initial_channels,
                                 kernel_size=self.kernel_size,
                                 padding=kernel_size - 2)
        self.sequence.append(self.initial_conv)

        n_channels = initial_channels
        for i in range(n_blocks):
            for j in range(blocks_per_level[i]):
                attrstring = 'conv_lvl%s_blk%s_nch%s' % (i, j, n_channels)
                resblock = ResBlock(name=attrstring,
                                    block_channels=n_channels,
                                    kernel_size=3,
                                    activation=self.activation,
                                    regularization=self.regularization)
                self.sequence.append(resblock)
                if i == n_blocks - 1:
                    break
            next_channels = 2 * n_channels
            strided_conv = Conv(in_channels=n_channels,
                                out_channels=next_channels,
                                stride=2)
            self.sequence.append(strided_conv)

    def forward(self, X):
        out = X
        for layer in self.sequence:
            y = layer(y)
        return y


class Decoder(nn.Module):
    pass


class Segmenter(nn.Module):
    pass


class UVAENet(nn.Module):
    pass
