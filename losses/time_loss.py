# Copyright 2018 Kaituo XU
#  MIT License (https://opensource.org/licenses/MIT)

import torch
import torch.nn as nn


EPS = 1e-6

def cal_SISNR(source, estimate_source):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """
    assert source.size() == estimate_source.size()
    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)
    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)
    return sisnr


def cal_SDR(target, est_target):
    assert target.size() == est_target.size()
    # Step 1. Zero-mean norm
    mean_source = torch.mean(target, dim=1, keepdim=True)
    mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
    target = target - mean_source
    est_target = est_target - mean_estimate
    # Step 2. SDR
    scaled_target = target
    e_noise = est_target - target
    sdr = torch.sum(scaled_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + EPS)
    sdr = 10 * torch.log10(sdr + EPS)
    return sdr  

    
