# -*- coding: utf-8 -*-
"""
Ported (with attribution) from Peng et al. 2021, UKBiobank_deep_pretrain:
https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain (MIT license)
dp_model/model_files/resnet3d.py, unmodified except log_softmax(dim=1)
made explicit (was relying on a deprecated implicit-dim default).

IMPORTANT: this repo ships pretrained weights for SFCN only
(brain_age/run_20190719_00_epoch_best_mae.p and a sex-prediction SFCN
checkpoint) — there is no pretrained checkpoint for this architecture in
the zip you provided. resnet18() below is therefore trained FROM SCRATCH
as your third, structurally-diverse ensemble member (per brainage.pdf
Section III-C.4's mention of "complementary architectures such as
3D-ResNet"). It still contributes real ensemble diversity even without
pretraining, since its errors are far less correlated with the two
pretrained-SFCN members' errors than a third identically-pretrained SFCN
would be — but expect it to need more regularization/epochs than the
fine-tuned SFCNs to reach a comparable individual MAE, and don't be
surprised if the ensemble's MAE-per-model shows it as the weakest single
member; its job is to decorrelate the ensemble, not to win alone.

avgpool kernel [5,6,5] assumes a 160x192x160 input (matches your pipeline
exactly): conv1(stride2) + maxpool(stride2) + 3x stride-2 residual stages
= 32x downsampling -> (160,192,160)/32 = (5,6,5).
"""

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000,
                 channel_size=[64, 64, 128, 256, 512], dropout=False):
        c = channel_size
        self.inplanes = c[0]
        super(ResNet, self).__init__()
        net = nn.Sequential()
        net.add_module('conv1', nn.Conv3d(1, c[0], kernel_size=7, stride=2, padding=0, bias=False))
        net.add_module('bn1', nn.BatchNorm3d(c[0]))
        net.add_module('relu', nn.ReLU(inplace=True))
        net.add_module('maxpool', nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
        net.add_module('layer1', self._make_layer(block, c[1], layers[0]))
        net.add_module('layer2', self._make_layer(block, c[2], layers[1], stride=2))
        net.add_module('layer3', self._make_layer(block, c[3], layers[2], stride=2))
        net.add_module('layer4', self._make_layer(block, c[4], layers[3], stride=2))
        net.add_module('avgpool', nn.AvgPool3d([5, 6, 5], stride=1))
        if dropout is True:
            net.add_module('dropout', nn.Dropout(0.5))
        self.feature_extractor = net
        self.classifier = nn.Linear(c[4] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)  # explicit dim (was implicit/deprecated in the source repo)
        return x


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
