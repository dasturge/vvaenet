import torch
from torch import optim

from functions import dice_loss, kl_divergence, l2_loss


def train(net, X, y, epochs=1, batch_size=1, **cfg):

    opt = optim.Adam(net.parameters(), lr=1e-5)

    for e in range(epochs):

        for i, (xi, yi) in enumerate(zip(X, y)):
            y_pred, (x_recon, mu, logvar) = net(xi)
            l_vae = kl_divergence(mu, logvar)
            l_frob = l2_loss(x_recon, xi)
            l_dice = dice_loss(y_pred, yi)
            objective = l_dice + 0.1 * l_frob + 0.1 * l_vae
            objective.backward()
            if ((i + 1) / 20) == 0:
                opt.step()
                opt.zero_grad()
            print('loss %s' % objective.item())
        opt.step()
        opt.zero_grad()
        print('epoch %s loss %s' % (e, objective.item()))


if __name__ == '__main__':
    import torchvision.datasets as data
    data
