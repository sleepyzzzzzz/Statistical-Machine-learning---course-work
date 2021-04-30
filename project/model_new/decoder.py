import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, num_classes, BatchNorm, num_low_level_feat):
        super(Decoder, self).__init__()
        low_level_inplanes = 256

        self.num_low_level_feat = num_low_level_feat

        self.bottlenecks = nn.ModuleList()

        for i in range(self.num_low_level_feat):
            self.bottlenecks.append(nn.Sequential(
                nn.Conv2d(low_level_inplanes * 2**i, 48, 1, bias=False),
                BatchNorm(48),
                nn.ReLU()))

        self.last_conv = nn.Sequential(nn.Conv2d(256 + 48 * self.num_low_level_feat, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, low_level_feats):
        low_level_feats = low_level_feats[:self.num_low_level_feat]

        for i, bottleneck in enumerate(self.bottlenecks):
            low_level_feats[i] = bottleneck(low_level_feats[i])
            low_level_feats[i] = F.interpolate(low_level_feats[i], size=low_level_feats[0].size()[2:], mode='bilinear', align_corners=True)

        x = F.interpolate(x, size=low_level_feats[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, *low_level_feats), dim=1)

        x = self.last_conv(x)

        return x

def build_decoder(num_classes, BatchNorm, num_low_level_feat):
    return Decoder(num_classes, BatchNorm, num_low_level_feat)