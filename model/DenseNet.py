import re
from typing import Any, List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from TFConv import GS, GS_X, LOG, Gab, GS_X2, SCHMID


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


class _DenseLayer(nn.Module):
    def __init__(self,
                 input_c: int,
                 growth_rate: int,
                 bn_size: int,
                 drop_rate: float,
                 memory_efficient: bool = False):
        super(_DenseLayer, self).__init__()

        self.add_module("norm1", nn.BatchNorm2d(input_c))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(in_channels=input_c,
                                           out_channels=bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1,
                                           bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate,
                                           growth_rate,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           bias=False))
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concat_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concat_features)))
        return bottleneck_output

    @staticmethod
    def any_requires_grad(inputs: List[Tensor]) -> bool:
        for tensor in inputs:
            if tensor.requires_grad:
                return True

        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, inputs: List[Tensor]) -> Tensor:
        def closure(*inp):
            return self.bn_function(inp)

        return cp.checkpoint(closure, *inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        if isinstance(inputs, Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("memory efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)

        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self,
                 num_layers: int,
                 input_c: int,
                 bn_size: int,
                 growth_rate: int,
                 drop_rate: float,
                 memory_efficient: bool = False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_c + i * growth_rate,
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self,
                 input_c: int,
                 output_c: int):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(input_c))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(input_c,
                                          output_c,
                                          kernel_size=1,
                                          stride=1,
                                          bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    Densenet-BC model class for imagenet

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient
    """

    def __init__(self,
                 firstlayer,
                 growth_rate: int = 32,
                 block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
                 num_init_features: int = 48,
                 bn_size: int = 4,
                 drop_rate: float = 0,
                 num_classes: int = 1000,
                 memory_efficient: bool = False):
        super(DenseNet, self).__init__()

        # first conv+bn+relu+pool
        self.first = firstlayer
        self.features = nn.Sequential(OrderedDict([
            ("conv0", firstlayer),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # each dense block
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                input_c=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                memory_efficient=memory_efficient)
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(input_c=num_features,
                                    output_c=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # finnal batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # fc layer
        self.classifier = nn.Linear(num_features, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def Densenet121(input_channel=1, num_classes=5):
    firstlayer = nn.Conv2d(input_channel, 48, kernel_size=7, stride=2, padding=3, bias=False)
    return DenseNet(firstlayer, growth_rate=32,
                    block_config=(6, 12, 24, 16),
                    num_init_features=48,
                    num_classes=num_classes)


def Densenet121_Garbor(num_classes=1000):
    firstlayer = Gab(out_channl=48, in_channl=1, kernal_size=11)
    return DenseNet(firstlayer, growth_rate=32,
                    block_config=(6, 12, 24, 16),
                    num_init_features=48,
                    num_classes=num_classes)


def Densenet121_SCHMID(num_classes=1000):
    firstlayer = SCHMID(out_channl=48, in_channl=1, kernal_size=11)
    return DenseNet(firstlayer, growth_rate=32,
                    block_config=(6, 12, 24, 16),
                    num_init_features=48,
                    num_classes=num_classes)


def Densenet121_LM(num_classes=1000):
    firstlayer = lm()
    return DenseNet(firstlayer, growth_rate=32,
                    block_config=(6, 12, 24, 16),
                    num_init_features=48,
                    num_classes=num_classes)
