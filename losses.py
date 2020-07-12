import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(input, target):
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.contiguous().view(-1).float()
    intersection = (iflat * tflat).sum()

    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class KLDivergence(nn.Module):
    def forward(self, mu, logvar):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0
        )
        return kld_loss


def l2_loss(x_recon, x):
    l2_norm = torch.mean((x_recon - x).pow(2))
    return l2_norm
