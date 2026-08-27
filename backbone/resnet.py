"""
CIFAR-style ResNets (He et al. 2016), depth = 6n + 2.

The stem is a single 3x3 conv (no max-pool, no 7x7) and the three stages hold
n BasicBlocks each at widths 16/32/64, so depth 20 is n=3 and depth 110 is
n=18. Downsampling shortcuts use the option-B 1x1 projection.
"""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


def _make_layer(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
    blocks = [BasicBlock(in_ch, out_ch, stride)]
    for _ in range(1, n_blocks):
        blocks.append(BasicBlock(out_ch, out_ch, stride=1))
    return nn.Sequential(*blocks)


class ResNet(nn.Module):
    """CIFAR ResNet of depth 6n+2.

    zero_init_residual zero-initializes the last BN of every block, so each
    block starts as an identity map. It costs nothing and is what keeps deep
    stacks (110 layers = 54 blocks) trainable at lr=0.1 from scratch, which
    the original paper needed an lr=0.01 warmup phase to achieve.
    """

    def __init__(
        self,
        depth: int = 110,
        num_classes: int = 100,
        zero_init_residual: bool = True,
    ):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError(f"depth must be 6n+2 (20, 32, 44, 56, 110, ...), got {depth}")
        n = (depth - 2) // 6
        self.depth = depth

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = _make_layer(16, 16, n_blocks=n, stride=1)
        self.layer2 = _make_layer(16, 32, n_blocks=n, stride=2)
        self.layer3 = _make_layer(32, 64, n_blocks=n, stride=2)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.mean(dim=(2, 3))
        return self.fc(out)


def resnet20(**kwargs) -> ResNet:
    return ResNet(depth=20, **kwargs)


def resnet110(**kwargs) -> ResNet:
    return ResNet(depth=110, **kwargs)