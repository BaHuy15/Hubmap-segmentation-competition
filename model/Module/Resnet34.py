import torchvision
import torch.nn as nn
from model.Module.FPA import *
from model.Module.SCse import *
from model.Module.Decoderv2 import *
from model.Module.Decoder import *


# stage2 model
class Res34Unetv3(nn.Module):
    def __init__(self):
        super(Res34Unetv3, self).__init__()
        self.resnet = torchvision.models.resnet34(True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu)

        self.encode2 = nn.Sequential(self.resnet.layer1,
                                     SCse(64))
        self.encode3 = nn.Sequential(self.resnet.layer2,
                                     SCse(128))
        self.encode4 = nn.Sequential(self.resnet.layer3,
                                     SCse(256))
        self.encode5 = nn.Sequential(self.resnet.layer4,
                                     SCse(512))

        self.center = nn.Sequential(FPAv2(512, 256),
                                    nn.MaxPool2d(2, 2))

        self.decode5 = Decoderv2(256, 512, 64)
        self.decode4 = Decoderv2(64, 256, 64)
        self.decode3 = Decoderv2(64, 128, 64)
        self.decode2 = Decoderv2(64, 64, 64)
        self.decode1 = Decoder(64, 32, 64)

        self.dropout2d = nn.Dropout2d(0.4)
        self.dropout = nn.Dropout(0.4)

        self.fuse_pixel = conv3x3(320, 64)
        self.logit_pixel = nn.Conv2d(64, 1, kernel_size=1, bias=False)

        self.fuse_image = nn.Sequential(nn.Linear(512, 64),
                                        nn.ELU(True))
        self.logit_image = nn.Sequential(nn.Linear(64, 1),
                                         nn.Sigmoid())
        self.logit = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                                   nn.ELU(True),
                                   nn.Conv2d(64, 1, kernel_size=1, bias=False))

    def forward(self, x):
        # x: (batch_size, 3, 256, 256)
        batch_size, c, h, w = x.shape

        x = self.conv1(x)  # 64, 128, 128
        e2 = self.encode2(x)  # 64, 128, 128
        e3 = self.encode3(e2)  # 128, 64, 64
        e4 = self.encode4(e3)  # 256, 32, 32
        e5 = self.encode5(e4)  # 512, 16, 16

        e = F.adaptive_avg_pool2d(e5, output_size=1).view(batch_size, -1)  # 512
        e = self.dropout(e)

        f = self.center(e5)  # 256, 8, 8

        d5 = self.decode5(f, e5)  # 64, 16, 16
        d4 = self.decode4(d5, e4)  # 64, 32, 32
        d3 = self.decode3(d4, e3)  # 64, 64, 64
        d2 = self.decode2(d3, e2)  # 64, 128, 128
        d1 = self.decode1(d2)  # 64, 256, 256

        f = torch.cat((d1,
                       F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True),
                       F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=True),
                       F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=True),
                       F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=True)), 1)  # 320, 256, 256
        f = self.dropout2d(f)

        # segmentation process
        fuse_pixel = self.fuse_pixel(f)  # 64, 256, 256
        logit_pixel = self.logit_pixel(fuse_pixel)  # 1, 256, 256

        # classification process
        fuse_image = self.fuse_image(e)  # 64
        logit_image = self.logit_image(fuse_image)  # 1

        # combine segmentation and classification
        fuse = torch.cat([fuse_pixel,
                          F.upsample(fuse_image.view(batch_size, -1, 1, 1), scale_factor=256, mode='bilinear',
                                     align_corners=True)], 1)  # 128, 256, 256
        logit = self.logit(fuse)  # 1, 256, 256

        return logit#, logit_pixel , logit_image.view(-1)


