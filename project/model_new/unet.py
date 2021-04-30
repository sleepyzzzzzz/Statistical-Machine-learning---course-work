import torch
import torch.nn as nn
import torch.nn.functional as F

class unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.conv_down1 = nn.Sequential(nn.Conv2d(n_channels, 64, kernel_size = 3, stride = 1, padding = 1, bias=False),
                                     nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.Dropout2d(0.2),
                                     nn.ReLU(inplace=True))
        
        self.conv_down2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.Dropout2d(0.2),
                                     nn.ReLU(inplace=True))
        
        self.conv_down3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.Dropout2d(0.2),
                                     nn.ReLU(inplace=True))
        
        self.conv_down4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.Dropout2d(0.2),
                                     nn.ReLU(inplace=True))
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_mid = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),   
                                     nn.Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True))
        
        self.conv_up4 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True))
        
        self.conv_up3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True))
        
        self.conv_up2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True))
        
        self.conv_up1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                                     nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True))
        
        self.upconv4 = nn.ConvTranspose2d(1024, 1024, kernel_size=(2, 2), stride=2)
        
        self.upconv3 = nn.ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))
        
        self.upconv2 = nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
        
        self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2))
        
        self.conv_last = nn.Conv2d(64, n_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        out = self.conv_down1(x)
        out = self.pool(out)
        out = self.conv_down2(out)
        out = self.pool(out)
        out = self.conv_down3(out)
        out = self.pool(out)
        out = self.conv_down4(out)
        out = self.pool(out)
        out = self.conv_mid(out)
        out = self.upconv4(out)
        out = self.conv_up4(out)
        out = self.upconv3(out)
        out = self.conv_up3(out)
        out = self.upconv2(out)
        out = self.conv_up2(out)
        out = self.upconv1(out)
        out = self.conv_up1(out)
        out = self.conv_last(out)
        return out