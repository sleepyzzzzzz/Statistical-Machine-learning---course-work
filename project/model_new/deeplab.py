import torch
import torch.nn as nn
import torch.nn.functional as F
from .aspp import build_aspp
from .decoder import build_decoder
from .resnet import build_backbone
import torchvision


class DeepLab(nn.Module):
    def __init__(self, output_stride=16, num_classes=7, num_low_level_feat=1):
        super(DeepLab, self).__init__()

        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(output_stride)
        self.aspp = build_aspp(output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, BatchNorm, num_low_level_feat)

    def forward(self, input):
        x, low_level_feats = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x)
        return x