from torch import nn
# from torchvision.models.utils import load_state_dict_from_url
# 0.4 以及 0.7 的torchvision 的版本用此语句 高版本不再使用
from torch.hub import load_state_dict_from_url
import torch
from TFConv import GS, GS_X, LOG, Gab, GS_X2, SCHMID

__all__ = ['MobileNetV2', 'mobilenet_v2']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class lm(nn.Module):
    def __init__(self):
        super(lm, self).__init__()
        self.conv1_1 = GS_X(out_channl=18, in_channl=1, kernal_size=11, padding=5)
        self.conv1_2 = GS_X2(out_channl=18, in_channl=1, kernal_size=11, padding=5)
        self.conv1_3 = LOG(out_channl=8, in_channl=1, kernal_size=11, padding=5)
        self.conv1_4 = GS(out_channl=4, in_channl=1, kernal_size=11, padding=5)

    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x3 = self.conv1_3(x)
        x4 = self.conv1_4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, firstlayer, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 48
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                # 112, 112, 32 -> 112, 112, 16
                [1, 16, 1, 1],
                # 112, 112, 16 -> 56, 56, 24
                [6, 24, 2, 2],
                # 56, 56, 24 -> 28, 28, 32
                [6, 32, 3, 2],
                # 28, 28, 32 -> 14, 14, 64
                [6, 64, 4, 2],
                # 14, 14, 64 -> 14, 14, 96
                [6, 96, 3, 1],
                # 14, 14, 96 -> 7, 7, 160
                [6, 160, 3, 2],
                # 7, 7, 160 -> 7, 7, 320
                [6, 320, 1, 1],
            ]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.conv1 = firstlayer
        # 224, 224, 3 -> 112, 112, 32
        features = [self.conv1]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # 7, 7, 320 -> 7,7,1280
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        # 1280
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True


def mobilenet_v2(num_classes=5):
    firstlayer = ConvBNReLU(in_planes=1, out_planes=48, stride=2)
    return MobileNetV2(firstlayer, num_classes=num_classes)


def mobilenet_v2_Garbor(num_classes=1000):
    firstlayer = nn.Sequential(Gab(out_channl=48, in_channl=1, kernal_size=11),
                               nn.BatchNorm2d(48),
                               nn.ReLU6(inplace=True))
    return MobileNetV2(firstlayer, num_classes=num_classes)


def mobilenet_v2_SCHMID(num_classes=1000):
    firstlayer = nn.Sequential(SCHMID(out_channl=48, in_channl=1, kernal_size=11),
                               nn.BatchNorm2d(48),
                               nn.ReLU6(inplace=True))
    return MobileNetV2(firstlayer, num_classes=num_classes)


def mobilenet_v2_LM(num_classes=1000):
    firstlayer = nn.Sequential(lm(),
                               nn.BatchNorm2d(48),
                               nn.ReLU6(inplace=True))
    return MobileNetV2(firstlayer, num_classes=num_classes)
