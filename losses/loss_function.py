import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

EPS = 1e-6

from .time_loss import cal_SDR, cal_SISNR

class loss_wrapper(_Loss):
    def __init__(self, loss_type):
        super(loss_wrapper, self).__init__()
        self.loss_type = loss_type
        if self.loss_type == 'hybrid':
            from .stft_loss import MultiResolutionSTFTLoss
            self.stft_loss = MultiResolutionSTFTLoss()
        if self.loss_type == 'SpEx-plus':
            from .class_loss import Loss_Softmax
            self.ae_loss = Loss_Softmax()

    def forward(self, clean, estimate):
        if self.loss_type == 'snr':
            loss = 0 - torch.mean(cal_SDR(clean, estimate))
        elif self.loss_type == 'sisdr':
            loss = 0 - torch.mean(cal_SISNR(clean, estimate))
        elif self.loss_type == 'hybrid':
            loss = 0 - torch.mean(cal_SISNR(clean, estimate)) + self.stft_loss(clean, estimate)
        elif self.loss_type == 'SpEx-plus':
            loss = self.spex_plus_loss(clean, estimate)
        else:
            raise NameError('Wrong loss selection')
        
        return loss
    
    def spex_plus_loss(self, clean, estimate):
        ests, ests2, ests3, spk_pred, speakers = estimate
        loss = 0 - torch.mean(cal_SISNR(clean, ests))
        if torch.sum(speakers) >=0:
            max_snr_2 = cal_SISNR(clean, ests2)
            max_snr_3 = cal_SISNR(clean, ests3)
            loss = 0.8*loss - 0.1*torch.mean(max_snr_2) - 0.1*torch.mean(max_snr_3)
            speaker_loss, spk_acc_0 = self.ae_loss(spk_pred, speakers)
            loss = loss + 0.5*speaker_loss
        return loss