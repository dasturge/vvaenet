import torch
import torch.nn as nn

from eisen.ops.losses import DiceLoss
from eisen.ops.metrics import DiceMetric


class SingleDiceMetric(DiceMetric):
    def __init__(self, class_index, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_index = class_index

    def forward(self, predictions, labels):
        predictions = predictions[:, self.class_index, ...]
        labels = labels[:, self.class_index, ...]
        return super().forward(predictions, labels)


class SingleDiceLoss(DiceLoss):
    def __init__(self, class_index, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_index = class_index

    def forward(self, predictions, labels):
        predictions = predictions[:, self.class_index, ...].unsqueeze(1)
        labels = labels[:, self.class_index, ...].unsqueeze(1)
        return super().forward(predictions, labels)


class KLDivergence(nn.Module):
    def __init__(self, N) -> None:
        super().__init__()
        self.N = N

    def forward(self, mu, logvar):
        kld_loss = (
            torch.sum(
                -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0
            )
            / self.N
        )
        return kld_loss

