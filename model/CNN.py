import torch
from torch import nn
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


class CNN(nn.Module):
    def __init__(self, firstlayer, num_classes=1000):
        super(CNN, self).__init__()
        self.conv1 = firstlayer
        self.features = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 64, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=7, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 120), nn.ReLU(),
            nn.Linear(120, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def CNN_base(num_classes=1000):
    firstlayer = nn.Conv2d(1, 48, kernel_size=11, padding=5, stride=2)
    return CNN(firstlayer, num_classes=num_classes)


def CNN_Garbor(num_classes=1000):
    firstlayer = Gab(out_channl=48, in_channl=1, kernal_size=11)
    return CNN(firstlayer, num_classes=num_classes)


def CNN_SCHMID(num_classes=1000):
    firstlayer = SCHMID(out_channl=48, in_channl=1, kernal_size=11)
    return CNN(firstlayer, num_classes=num_classes)


def CNN_LM(num_classes=1000):
    firstlayer = lm()
    return CNN(firstlayer, num_classes=num_classes)


if __name__ == "__main__":
    ipt = torch.randn([1, 1, 224, 224]).cuda()
    model = CNN_LM(num_classes=5).cuda()
    out = model(ipt)
