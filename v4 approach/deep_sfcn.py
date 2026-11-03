# -*- coding: utf-8 -*-
"""Deeper transfer-learning SFCN for V4.

The UK Biobank-pretrained SFCN feature extractor is retained unchanged. Extra
residual 3D convolution blocks are inserted after the pretrained feature
extractor, at its final 5x6x5 feature-map resolution. The pretrained feature
extractor and copied age-bin classifier therefore remain loadable; the added
blocks are initialized close to identity so the pretrained representation is
not abruptly destroyed at initialization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConv3D(nn.Module):
    def __init__(self, channels: int = 64, activation: str = "relu"):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.act = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        # Start each block close to the identity mapping.
        nn.init.zeros_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x):
        residual = x
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(y + residual)


class DeeperTransferSFCN(nn.Module):
    """Wrap a pretrained core_v3 SFCN and add trainable residual blocks."""

    def __init__(self, pretrained_base: nn.Module, extra_blocks: int = 3):
        super().__init__()
        if extra_blocks < 1:
            raise ValueError("extra_blocks must be at least 1")
        # These modules retain their original names, allowing the pretrained
        # backbone and transferred age-bin head to be reused directly.
        self.feature_extractor = pretrained_base.feature_extractor
        self.extra_blocks = nn.Sequential(
            *[ResidualConv3D(channels=64) for _ in range(extra_blocks)]
        )
        self.classifier = pretrained_base.classifier

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.extra_blocks(x)
        x = self.classifier(x)
        return F.log_softmax(x.flatten(1), dim=1)
