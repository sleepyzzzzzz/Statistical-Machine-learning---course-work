import torch
import torch.nn as nn
from catalyst.contrib.nn import DiceLoss, LovaszLossMultiLabel, FocalLossBinary

Loss = {
    "bce": nn.BCEWithLogitsLoss,
    "dice": DiceLoss,
    "focal": FocalLossBinary
}

class MixedLoss(nn.Module):
    def __init__(self, names):
        super().__init__()

        self.loss_funcs = []

        for name in names:
            self.loss_funcs.append(Loss[name]())

        self.loss_funcs = nn.ModuleList(self.loss_funcs)

    def forward(self, input, target):
        loss = torch.stack([loss_func(input, target) for loss_func in self.loss_funcs])

        sum_loss = loss.sum()

        return sum_loss, loss