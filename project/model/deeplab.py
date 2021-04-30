import torch
import torch.nn as nn
import torch.nn.functional as F
from .aspp import build_aspp
from .decoder import build_decoder
from .backbone import build_backbone
import torchvision


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 ibn_mode='none', num_low_level_feat=1, interpolate_before_lastconv=False):
        super(DeepLab, self).__init__()

        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, ibn_mode, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm, num_low_level_feat, interpolate_before_lastconv)
        self.interpolate_before_lastconv = interpolate_before_lastconv

    def forward(self, input):
        x, low_level_feats = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feats)

        if not self.interpolate_before_lastconv:
            x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x