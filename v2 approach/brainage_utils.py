# -*- coding: utf-8 -*-
"""
Ported (with attribution) from Peng et al. 2021, UKBiobank_deep_pretrain:
https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain (MIT license)
dp_model/dp_utils.py + dp_model/dp_loss.py, unmodified in logic.

These two functions are exactly what the pretrained SFCN checkpoint was
trained with, so reusing them verbatim (rather than reimplementing the
soft-label/KL scheme from the paper description) guarantees your training
target distribution and loss match what the frozen backbone already
"expects" from its pretraining.
"""

import numpy as np
from scipy.stats import norm
import torch.nn as nn


def num2vect(x, bin_range, bin_step, sigma):
    """
    v, bin_centers = num2vect(x, bin_range, bin_step, sigma)

    bin_range : (start, end) tuple
    bin_step  : divisor of (end - start)
    sigma     : = 0  -> hard label, v is bin index
                > 0  -> soft label, v is a probability vector (Gaussian
                        CDF mass integrated over each bin), sigma in years
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        raise ValueError("bin_range must be divisible by bin_step")
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step).astype(int)
        return i, bin_centers

    if np.isscalar(x):
        v = np.zeros((bin_number,))
        for i in range(bin_number):
            x1 = bin_centers[i] - float(bin_step) / 2
            x2 = bin_centers[i] + float(bin_step) / 2
            cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
            v[i] = cdfs[1] - cdfs[0]
        return v, bin_centers

    v = np.zeros((len(x), bin_number))
    for j in range(len(x)):
        for i in range(bin_number):
            x1 = bin_centers[i] - float(bin_step) / 2
            x2 = bin_centers[i] + float(bin_step) / 2
            cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
            v[j, i] = cdfs[1] - cdfs[0]
    return v, bin_centers


def crop_center(data, out_sp):
    """Returns the center crop of a volume. in_sp must be >= out_sp."""
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        return data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop] if x_crop or y_crop or z_crop else data
    elif nd == 4:
        return data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop] if x_crop or y_crop or z_crop else data
    raise ValueError(f"Wrong dimension! dim={nd}.")


def my_KLDivLoss(x, y):
    """
    KL-divergence loss for soft-label brain-age regression.
    x: model output, log-probabilities (already log_softmax'd), shape (B, n_bins)
    y: soft-label target, probabilities (from num2vect), shape (B, n_bins)
    Averaged over batch size (not summed), y is epsilon-stabilized to avoid log(0).
    """
    loss_func = nn.KLDivLoss(reduction="sum")
    y = y + 1e-16
    n = y.shape[0]
    return loss_func(x, y) / n
