# -*- coding: utf-8 -*-
"""
SFCN architecture + transfer-learning helpers — Revision 2, Section III-C.1/C.2.

WHY THIS FILE REPLACES YOUR ORIGINAL Cell-6 SFCN CLASS:
Your original notebook defined its own SFCN (InstanceNorm3d, AdaptiveAvgPool3d,
a single-scalar nn.Linear head). That is architecturally incompatible with the
pretrained checkpoint's state_dict — different module names AND different
normalization layers (InstanceNorm3d vs. the checkpoint's BatchNorm3d, whose
running statistics matter for a frozen backbone). `load_state_dict` cannot
bridge that gap. The class below is the *exact* architecture from
dp_model/model_files/sfcn.py (module names: feature_extractor.conv_0 ..
conv_5, classifier.average_pool / dropout / conv_6) so the checkpoint loads
cleanly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True):
        super(SFCN, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            in_channel = 1 if i == 0 else channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(
                    'conv_%d' % i,
                    self.conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=1),
                )
            else:
                self.feature_extractor.add_module(
                    'conv_%d' % i,
                    self.conv_layer(in_channel, out_channel, maxpool=False, kernel_size=1, padding=0),
                )

        self.classifier = nn.Sequential()
        avg_shape = [5, 6, 5]  # matches 160x192x160 input after 5x stride-2 downsampling
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        in_channel = channel_number[-1]
        self.classifier.add_module(
            'conv_%d' % n_layer, nn.Conv3d(in_channel, output_dim, padding=0, kernel_size=1)
        )

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool:
            return nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        return nn.Sequential(
            nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x = F.log_softmax(x, dim=1)
        return x  # shape (B, output_dim, 1, 1, 1)


# =========================================================
# Transfer learning helpers (Section III-C.1)
# =========================================================

def load_pretrained_sfcn(checkpoint_path: str, n_bins: int, device):
    """
    Loads the UKBiobank-pretrained SFCN weights and re-heads the model for
    your own age-bin count (the UKB checkpoint used 40 bins over [42,82];
    your OASIS-3 cohort's age range is almost certainly different, so the
    final 1x1 conv (`classifier.conv_5` / the layer index n_layer) is always
    reinitialized regardless of n_bins — its shape depends on your data).

    Returns a model with ALL weights loaded except the final classifier
    conv, which is freshly initialized.
    """
    model = SFCN(output_dim=n_bins)

    state_dict = torch.load(checkpoint_path, map_location=device)
    # The released checkpoint was saved from inside nn.DataParallel, so keys
    # are prefixed with "module.". Strip that prefix if present.
    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                  for k, v in state_dict.items()}

    own_state = model.state_dict()
    loaded, skipped = [], []
    for name, param in state_dict.items():
        if name in own_state and own_state[name].shape == param.shape:
            own_state[name].copy_(param)
            loaded.append(name)
        else:
            skipped.append(name)

    model.load_state_dict(own_state)
    model.to(device)

    print(f"[load_pretrained_sfcn] loaded {len(loaded)} tensors, skipped {len(skipped)} "
          f"(expected: the final classifier conv, shape mismatch due to n_bins={n_bins})")
    if skipped:
        print(f"  skipped: {skipped}")

    return model


def freeze_sfcn_backbone(model: SFCN, unfreeze_from_block: int = 5):
    """
    Implements brainage.pdf Section III-C.2: "unfreeze the final
    convolutional block and the output layer" only.

    In this architecture, feature_extractor.conv_0..conv_4 are blocks 1-5
    (channels 32,64,128,256,256) and conv_5 is block 6 (256->64, the "final
    convolutional block"). classifier.* is the output layer. Default
    unfreeze_from_block=5 freezes conv_0..conv_4 and leaves conv_5 +
    classifier trainable, matching the paper exactly.
    """
    for name, param in model.feature_extractor.named_parameters():
        block_idx = int(name.split('.')[0].split('_')[1])  # "conv_3.0.weight" -> 3
        param.requires_grad = block_idx >= unfreeze_from_block

    for param in model.classifier.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[freeze_sfcn_backbone] trainable params: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.1f}%)")
    return model


def differential_lr_param_groups(model, backbone_lr=1e-5, head_lr=1e-4):
    """
    Section III-C.1: differential learning rates — small LR for the
    unfrozen backbone tail, larger LR for the freshly-initialized head.
    Works for both SFCN (feature_extractor / classifier) and, if you reuse
    it for the from-scratch ResNet, will just put everything in one group
    at backbone_lr since ResNet has no `classifier`-vs-`feature_extractor`
    split worth training differently when initialized from scratch.
    """
    backbone_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and n.startswith('feature_extractor')]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith('feature_extractor')]
    groups = []
    if backbone_params:
        groups.append({'params': backbone_params, 'lr': backbone_lr})
    if head_params:
        groups.append({'params': head_params, 'lr': head_lr})
    return groups


# =========================================================
# Unified prediction helper (works for SFCN AND resnet3d.ResNet — both
# now return log-softmax over the same age-bin scheme)
# =========================================================

def expected_age_from_logprobs(log_probs: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    """
    log_probs: (B, n_bins[, 1, 1, 1]) log-softmax output from either model.
    bin_centers: (n_bins,) tensor of bin center ages.
    Returns (B,) expected age = sum_i p_i * c_i  (paper Eq. 5).
    """
    probs = torch.exp(log_probs).reshape(log_probs.shape[0], -1)  # (B, n_bins)
    return probs @ bin_centers.to(probs.device, probs.dtype)
