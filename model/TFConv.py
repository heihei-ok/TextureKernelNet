import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def get_data(basesigma, basethetas, channl):
    sigmas = []
    thetas = []
    for theta in basethetas:
        for sigma in basesigma:
            sigmas.append(sigma)
            thetas.append(theta)
    s, d = divmod(channl, len(basesigma) * len(basethetas))
    if s == 0:
        out_sigma = sigmas[:channl]
        out_theta = thetas[:channl]
    else:
        out_sigma = sigmas * (s + 1)
        out_sigma = out_sigma[:channl]
        out_theta = thetas * (s + 1)
        out_theta = out_theta[:channl]
    out_sigma = torch.tensor(out_sigma, dtype=torch.float32)
    out_theta = torch.tensor(out_theta, dtype=torch.float32)
    return out_sigma, out_theta


class GS(nn.Module):
    def __init__(self, out_channl, kernal_size, stride=2, padding=5, in_channl=1):
        super(GS, self).__init__()
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding
        if in_channl != 1:
            msg = " Only support one input channel ( here, in_channels = {%i} )" % in_channl
            raise ValueError(msg)
        self.in_channl = in_channl
        self.out_channl = out_channl
        self.basesigma = torch.tensor([0.6, 0.8, 1.0, 1.2], dtype=torch.float32)
        self.s = self.out_channl // len(self.basesigma)
        self.x = self.basesigma * (self.s + 1)
        self.sigmas = nn.Parameter(self.x[:self.out_channl].unsqueeze(1),
                                   requires_grad=True)

    def forward(self, fea):
        x0 = torch.ceil(torch.Tensor([self.kernal_size / 2])[0])
        y0 = torch.ceil(torch.Tensor([self.kernal_size / 2])[0])
        fea_y, fea_x = torch.meshgrid(
            [
                torch.linspace(-x0 + 1, x0 - 1, self.kernal_size),
                torch.linspace(-y0 + 1, y0 - 1, self.kernal_size),
            ], indexing='xy'
        )
        fea_x, fea_y = fea_x.repeat(self.out_channl, 1, 1).cuda(), fea_y.repeat(self.out_channl, 1, 1).cuda()

        g = torch.exp(
            -(fea_x ** 2 + fea_y ** 2) /
            (2 * (self.sigmas.unsqueeze(1)) ** 2)
        )
        g = g / abs(g).sum(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        self.filters = g.unsqueeze(1).cuda()
        return F.conv2d(fea, self.filters, stride=self.stride, padding=self.padding, bias=None)


class GS_X(nn.Module):
    def __init__(self, out_channl, kernal_size, stride=2, padding=5, in_channl=1):
        super(GS_X, self).__init__()
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding
        if in_channl != 1:
            msg = " Only support one input channel ( here, in_channels = {%i} )" % in_channl
            raise ValueError(msg)
        self.in_channl = in_channl
        self.out_channl = out_channl
        self.basesigma = torch.tensor([0.6, 0.8, 1.0, 1.2], dtype=torch.float32)
        self.basetheta = torch.arange(0, math.pi, math.pi / 6)
        self.sigmas = nn.Parameter(get_data(self.basesigma[:3], self.basetheta, self.out_channl)[0].unsqueeze(1),
                                   requires_grad=True)
        self.thetas = nn.Parameter(get_data(self.basesigma[:3], self.basetheta, self.out_channl)[1].unsqueeze(1),
                                   requires_grad=True)
        self.scales = nn.Parameter(
            1 + torch.ones([self.out_channl, self.in_channl], dtype=torch.float32).type(torch.Tensor) * 2,
            requires_grad=True)

    def forward(self, fea):
        x0 = torch.ceil(torch.Tensor([self.kernal_size / 2])[0])
        y0 = torch.ceil(torch.Tensor([self.kernal_size / 2])[0])
        fea_y, fea_x = torch.meshgrid(
            [
                torch.linspace(-x0 + 1, x0 - 1, self.kernal_size),
                torch.linspace(-y0 + 1, y0 - 1, self.kernal_size),
            ], indexing='xy'
        )
        fea_x, fea_y = fea_x.repeat(self.out_channl, 1, 1).cuda(), fea_y.repeat(self.out_channl, 1, 1).cuda()
        rotx = fea_x * torch.cos(self.thetas.unsqueeze(1)) + fea_y * torch.sin(self.thetas.unsqueeze(1))
        roty = -fea_x * torch.sin(self.thetas.unsqueeze(1)) + fea_y * torch.cos(self.thetas.unsqueeze(1))
        g_x = - (rotx / self.sigmas.unsqueeze(1)) * torch.exp(
            -(rotx ** 2 / (2 * self.sigmas.unsqueeze(1) ** 2) + roty ** 2 / (
                    2 * (self.sigmas.unsqueeze(1) * self.scales.unsqueeze(1)) ** 2))
        )
        g_x = g_x / abs(g_x).sum(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        self.filters = g_x.unsqueeze(1).cuda()
        return F.conv2d(fea, self.filters, stride=self.stride, padding=self.padding, bias=None)


class GS_X2(nn.Module):
    def __init__(self, out_channl, kernal_size, stride=2, padding=5, in_channl=1):
        super(GS_X2, self).__init__()
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding
        if in_channl != 1:
            msg = " Only support one input channel ( here, in_channels = {%i} )" % in_channl
            raise ValueError(msg)
        self.in_channl = in_channl
        self.out_channl = out_channl
        self.basesigma = torch.tensor([0.6, 0.8, 1.0, 1.2], dtype=torch.float32)
        self.basetheta = torch.arange(0, math.pi, math.pi / 6)
        self.sigmas = nn.Parameter(get_data(self.basesigma[:3], self.basetheta, self.out_channl)[0].unsqueeze(1),
                                   requires_grad=True)
        self.thetas = nn.Parameter(get_data(self.basesigma[:3], self.basetheta, self.out_channl)[1].unsqueeze(1),
                                   requires_grad=True)
        self.scales = nn.Parameter(
            1 + torch.ones([self.out_channl, self.in_channl], dtype=torch.float32).type(torch.Tensor) * 2,
            requires_grad=True)

    def forward(self, fea):
        x0 = torch.ceil(torch.Tensor([self.kernal_size / 2])[0])
        y0 = torch.ceil(torch.Tensor([self.kernal_size / 2])[0])
        fea_y, fea_x = torch.meshgrid(
            [
                torch.linspace(-x0 + 1, x0 - 1, self.kernal_size),
                torch.linspace(-y0 + 1, y0 - 1, self.kernal_size),
            ], indexing='xy'
        )
        fea_x, fea_y = fea_x.repeat(self.out_channl, 1, 1).cuda(), fea_y.repeat(self.out_channl, 1, 1).cuda()
        rotx = fea_x * torch.cos(self.thetas.unsqueeze(1)) + fea_y * torch.sin(self.thetas.unsqueeze(1))
        roty = -fea_x * torch.sin(self.thetas.unsqueeze(1)) + fea_y * torch.cos(self.thetas.unsqueeze(1))
        g_x = - (1 / self.sigmas.unsqueeze(1) ** 2) * (1 - (rotx ** 2 / self.sigmas.unsqueeze(1) ** 2)) \
              * torch.exp(
            -(rotx ** 2 / (2 * self.sigmas.unsqueeze(1) ** 2) + roty ** 2 / (
                    2 * (self.scales.unsqueeze(1) * self.sigmas.unsqueeze(1)) ** 2)))
        self.filters = g_x.unsqueeze(1).cuda()
        return F.conv2d(fea, self.filters, stride=self.stride, padding=self.padding, bias=None)


class LOG(nn.Module):
    def __init__(self, out_channl, kernal_size, stride=2, padding=5, in_channl=1):
        super(LOG, self).__init__()
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding
        if in_channl != 1:
            msg = " Only support one input channel ( here, in_channels = {%i} )" % in_channl
            raise ValueError(msg)
        self.in_channl = in_channl
        self.out_channl = out_channl
        self.basesigma = torch.tensor([0.6, 0.8, 1.0, 1.2], dtype=torch.float32)
        self.basesigma = torch.cat((self.basesigma, self.basesigma * 3), dim=0)

        self.s = self.out_channl // len(self.basesigma)
        self.x = self.basesigma * (self.s + 1)
        self.sigmas = nn.Parameter(self.x[:self.out_channl].unsqueeze(1),
                                   requires_grad=True)

    def forward(self, fea):
        x0 = torch.ceil(torch.Tensor([self.kernal_size / 2])[0])
        y0 = torch.ceil(torch.Tensor([self.kernal_size / 2])[0])
        fea_y, fea_x = torch.meshgrid(
            [
                torch.linspace(-x0 + 1, x0 - 1, self.kernal_size),
                torch.linspace(-y0 + 1, y0 - 1, self.kernal_size),
            ], indexing='xy'
        )
        fea_x, fea_y = fea_x.repeat(self.out_channl, 1, 1).cuda(), fea_y.repeat(self.out_channl, 1, 1).cuda()

        g = (-1 / (math.pi * self.sigmas.unsqueeze(1) ** 4)) * \
            (1 - ((fea_x ** 2 + fea_y ** 2) / (2 * self.sigmas.unsqueeze(1) ** 2))) * \
            torch.exp(-(fea_x ** 2 + fea_y ** 2) / (2 * self.sigmas.unsqueeze(1) ** 2))
        self.filters = g.unsqueeze(1).cuda()
        return F.conv2d(fea, self.filters, stride=self.stride, padding=self.padding, bias=None)


class Gab(nn.Module):
    def __init__(self, out_channl, kernal_size, stride=2, padding=5, in_channl=1):
        super(Gab, self).__init__()
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding
        if in_channl != 1:
            msg = " Only support one input channel ( here, in_channels = {%i} )" % in_channl
            raise ValueError(msg)
        self.in_channl = in_channl
        self.out_channl = out_channl
        self.basesigma = torch.tensor([0.6, 0.8, 1.0, 1.2], dtype=torch.float32)
        self.basetheta = torch.arange(0, math.pi, math.pi / 6)
        self.sigmas = nn.Parameter(get_data(self.basesigma, self.basetheta, self.out_channl)[0].unsqueeze(1),
                                   requires_grad=True)
        self.thetas = nn.Parameter(get_data(self.basesigma, self.basetheta, self.out_channl)[1].unsqueeze(1),
                                   requires_grad=True)
        self.scales = nn.Parameter(
            1 + torch.ones([self.out_channl, self.in_channl], dtype=torch.float32).type(torch.Tensor) * 2,
            requires_grad=True)
        self.freq = nn.Parameter(math.pi / self.sigmas, requires_grad=True)
        self.psi = nn.Parameter(
            math.pi * torch.rand(self.out_channl, 1), requires_grad=True
        )

    def forward(self, fea):
        x0 = torch.ceil(torch.Tensor([self.kernal_size / 2])[0])
        y0 = torch.ceil(torch.Tensor([self.kernal_size / 2])[0])
        fea_y, fea_x = torch.meshgrid(
            [
                torch.linspace(-x0 + 1, x0 - 1, self.kernal_size),
                torch.linspace(-y0 + 1, y0 - 1, self.kernal_size),
            ], indexing='xy'
        )
        fea_x, fea_y = fea_x.repeat(self.out_channl, 1, 1).cuda(), fea_y.repeat(self.out_channl, 1, 1).cuda()
        rotx = fea_x * torch.cos(self.thetas.unsqueeze(1)) + fea_y * torch.sin(self.thetas.unsqueeze(1))
        roty = -fea_x * torch.sin(self.thetas.unsqueeze(1)) + fea_y * torch.cos(self.thetas.unsqueeze(1))

        # g = torch.exp(
        #     -0.5 * ((rotx ** 2 + roty ** 2) / (self.sigmas.unsqueeze(1) + 1e-3) ** 2)
        # )
        # g = g * torch.cos(self.freq.unsqueeze(1) * rotx + self.psi.unsqueeze(1))
        # g = g / (2 * math.pi * self.sigmas.unsqueeze(1) ** 2)
        gb = torch.exp(
            -0.5 * (rotx ** 2 / self.sigmas.unsqueeze(1) ** 2 + roty ** 2 / (
                    self.sigmas.unsqueeze(1) * self.scales.unsqueeze(1)) ** 2)
        ) * torch.cos(self.freq.unsqueeze(1) * rotx + self.psi.unsqueeze(1))
        self.filters = gb.unsqueeze(1).cuda()
        return F.conv2d(fea, self.filters, stride=self.stride, padding=self.padding, bias=None)


class SCHMID(nn.Module):
    def __init__(self, out_channl, kernal_size, stride=2, padding=5, in_channl=1):
        super(SCHMID, self).__init__()
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding
        if in_channl != 1:
            msg = " Only support one input channel ( here, in_channels = {%i} )" % in_channl
            raise ValueError(msg)
        self.in_channl = in_channl
        self.out_channl = out_channl
        self.sigmas = nn.Parameter(
            (torch.rand([self.out_channl, self.in_channl], dtype=torch.float32).type(torch.Tensor) + 0.5),
            requires_grad=True)

        self.tao = nn.Parameter(
            (math.pi / 8)
            * torch.randint(0, 8, (self.out_channl, self.in_channl)).type(torch.Tensor),
            requires_grad=True)

    def forward(self, fea):
        x0 = torch.ceil(torch.Tensor([self.kernal_size / 2])[0])
        y0 = torch.ceil(torch.Tensor([self.kernal_size / 2])[0])
        fea_y, fea_x = torch.meshgrid(
            [
                torch.linspace(-x0 + 1, x0 - 1, self.kernal_size),
                torch.linspace(-y0 + 1, y0 - 1, self.kernal_size),
            ], indexing='xy'
        )
        fea_x, fea_y = fea_x.repeat(self.out_channl, 1, 1).cuda(), fea_y.repeat(self.out_channl, 1, 1).cuda()
        r = torch.sqrt(fea_x ** 2 + fea_y ** 2).cuda()
        g = torch.cos((r * math.pi * self.tao.unsqueeze(1)) / self.sigmas.unsqueeze(1)) * \
            torch.exp(-r ** 2 / (2 * self.sigmas.unsqueeze(1) ** 2))
        g = g - g.mean(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        g = g / torch.abs(g).sum(dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        self.filters = g.unsqueeze(1).cuda()
        return F.conv2d(fea, self.filters, stride=self.stride, padding=self.padding, bias=None)


if __name__ == "__main__":
    x = GS(18, 11)
