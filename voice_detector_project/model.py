"""CNN classifiers (PyTorch). Input: (batch, 1, 128, 401). Output: logit per segment."""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = x if self.shortcut is None else self.shortcut(x)
        return self.relu(out + identity)


class SmallResNet(nn.Module):
    """Stem + 4 stages of 2 residual blocks (32, 64, 128, 256 channels)."""
    def __init__(self, channels=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, 1, 1, bias=False), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        layers, in_ch = [], channels[0]
        for i, ch in enumerate(channels):
            stride = 1 if i == 0 else 2
            layers += [ResBlock(in_ch, ch, stride), ResBlock(ch, ch, 1)]
            in_ch = ch
        self.stages = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels[-1], 1)

    def forward(self, x):
        x = self.pool(self.stages(self.stem(x))).flatten(1)
        return self.fc(self.dropout(x)).squeeze(1)


class BaselineCNN(nn.Module):
    """Small baseline: 3 conv blocks (16, 32, 64) with max pooling."""
    def __init__(self):
        super().__init__()
        chans, layers, in_ch = (16, 32, 64), [], 1
        for ch in chans:
            layers += [nn.Conv2d(in_ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            in_ch = ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(chans[-1], 1)

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1)).squeeze(1)


def build_model(name):
    return {"resnet": SmallResNet, "baseline": BaselineCNN}[name]()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
