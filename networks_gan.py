# coding: UTF-8
"""
    @author: samuel ko
"""
import torch.nn.functional as F
import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, z_dims=512, d=64):
        super().__init__()
        self.deconv1 = nn.utils.spectral_norm(nn.ConvTranspose2d(z_dims, d * 8, 4, 1, 0))
        self.deconv2 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1))
        self.deconv3 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1))
        self.deconv4 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1))
        self.deconv5 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 2, d, 4, 2, 1))
        self.deconv6 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)  # 1 x 1
        x = F.relu(self.deconv1(input))  # 4 x 4
        x = F.relu(self.deconv2(x))  # 8 x 8
        x = F.relu(self.deconv3(x))  # 16 x 16
        x = F.relu(self.deconv4(x))  # 32 x 32
        x = F.relu(self.deconv5(x))  # 64 x 64
        x = F.tanh(self.deconv6(x))  # 128 x 128
        return x


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.layer1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.layer2 = nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        self.layer3 = nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        self.layer4 = nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))
        self.layer5 = nn.utils.spectral_norm(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False))
        self.layer6 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

    def forward(self, input):
        out = F.leaky_relu(self.layer1(input), 0.2, inplace=True)  # 64 x 64
        out = F.leaky_relu(self.layer2(out), 0.2, inplace=True)  # 32 x 32
        out = F.leaky_relu(self.layer3(out), 0.2, inplace=True)  # 16 x 16
        out = F.leaky_relu(self.layer4(out), 0.2, inplace=True)  # 8 x 8
        out = F.leaky_relu(self.layer5(out), 0.2, inplace=True)  # 4 x 4
        out = F.leaky_relu(self.layer6(out), 0.2, inplace=True)  # 1 x 1
        return out.view(-1, 1)