# Copyright 2018 Kaituo XU
#  MIT License (https://opensource.org/licenses/MIT)

import torch
import torch.nn as nn
from itertools import permutations

EPS = 1e-6

def cal_si_snr_with_pit(source, estimate_source, reorder_source = False):
    """Calculate SI-SNR with PIT training.
    Args:
        All in torch tensors
        source: [B, C, T], B: batch size, C: no. of speakers, T: sequence length
        estimate_source: [B, C, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # Step 1. Zero-mean norm
    zero_mean_target = source - torch.mean(source, dim=-1, keepdim=True)
    zero_mean_estimate = estimate_source - torch.mean(estimate_source, dim=-1, keepdim=True)
    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]
    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    # Step 3: Reorder the estimated source
    if reorder_source:
        reorder_estimate_source = _reorder_source(estimate_source, perms, max_snr_idx)
        return max_snr, reorder_estimate_source
    else:
        return max_snr

    
def cal_snr_with_pit(source, estimate_source, reorder_source = False):
    """Calculate SNR with PIT training.
    Args:
        All in torch tensors
        source: [B, C, T], B: batch size, C: no. of speakers, T: sequence length
        estimate_source: [B, C, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # Step 1. Zero-mean norm
    zero_mean_target = source - torch.mean(source, dim=-1, keepdim=True)
    zero_mean_estimate = estimate_source - torch.mean(estimate_source, dim=-1, keepdim=True)
    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    # s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    
    # pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    e_noise = s_estimate - s_target
    pair_wise_si_snr = torch.sum(s_target ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]
    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    # Step 3: Reorder the estimated source
    if reorder_source:
        reorder_estimate_source = _reorder_source(estimate_source, perms, max_snr_idx)
        return max_snr, reorder_estimate_source
    else:
        return max_snr

def _reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, *_ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source



