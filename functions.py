import torch
import torch.nn.functional as F


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.contiguous().view(-1).float()
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def kl_divergence(mu, logvar):
    kldiv = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kldiv


def l2_loss(x_recon, x):
    l2_norm = torch.mean((x_recon - x).pow(2))
    return l2_norm
