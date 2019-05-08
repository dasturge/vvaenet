import torch
from torch import optim

from functions import dice_loss, kl_divergence, l2_loss


def train(net, X, y, epochs=1, **cfg):

    opt = optim.Adam(net.parameters())

    for e in range(epochs):

        for xi, yi in zip(X, y):
            y_pred, (x_recon, mu, logvar) = net(xi)
            l_vae = kl_divergence(mu, logvar)
            l_frob = l2_loss(x_recon, xi)
            l_dice = dice_loss(y_pred, yi)
            objective = l_dice + 0.1 * l_frob + 0.1 * l_vae
            objective.backward()
            opt.step()
            opt.zero_grad()
            print('loss %s' % objective.item())


if __name__ == '__main__':
    import torchvision.datasets as data
    data
