# Copyright 2020 Kai Li
# Apache-2.0 license http://www.apache.org/licenses/LICENSE-2.0
# Modified from https://github.com/JusperLee/Dual-Path-RNN-Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
EPS = 1e-8
import math
import laion_clap

EPS = 1e-8

sector_mapping = {
    0: (-22.5, 22.5, "Front"),
    1: (-67.5, -22.5, "Right Front"),
    2: (-112.5, -67.5, "Right"),
    3: (-157.5, -112.5, "Right Rear"),
    4: (157.5, 202.5, "Rear"),
    5: (112.5, 157.5, "Left Rear"),
    6: (67.5, 112.5, "Left"),
    7: (22.5, 67.5, "Left Front"),
    8: (0.5, 2,"near"),
    9: (2,3.5,"middle"),
    10:(3.5,4.2,"far"),
}

spk_mapping = {
     1: "female",   
     2: "male",          
     3: "low",
     4: "loud",
}
mic_pairs = [(0, 1), (0, 2), (0,3),(1,2),(1,3),(2, 3)]  
class ss_Dprnn(nn.Module):
    def __init__(self, args):
        super(ss_Dprnn, self).__init__()
        N = args.network_audio.N
        L = args.network_audio.L
        B = args.network_audio.B
        H = args.network_audio.H
        K = args.network_audio.K
        R = args.network_audio.R
        
        self.encoder1 = Encoder1(L, N)
        self.encoder2 = Encoder2(L, N)
        self.ipd = TimeDomainIPD(kernel_size=L, num_filters=N)
        self.tpd =TPDLayer()
        self.textmodel = laion_clap.CLAP_Module(enable_fusion=True)
        self.textmodel.load_ckpt()
        for param in self.textmodel.parameters():
            param.requires_grad = False
        self.separator = rnn(args, N, B, H, K, R)
        self.decoder = Decoder(args, N, L)
        self.textmodel = laion_clap.CLAP_Module(enable_fusion=True)
        self.textmodel.load_ckpt()
        for param in self.textmodel.parameters():
            param.requires_grad = False
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def compute_V_time_domain(self, ipd, tpd):
        tpd =tpd.unsqueeze(-1).expand(-1, -1, -1, ipd.size(-1))
        cos_ipd = torch.cos(ipd)
        sin_ipd = torch.sin(ipd)
        cos_tpd = torch.cos(tpd)  
        sin_tpd = torch.sin(tpd)
        e_ipd = torch.stack([cos_ipd, sin_ipd], dim=-1)  # IPD 的方向向量
        e_tpd = torch.stack([cos_tpd, sin_tpd], dim=-1)  # TPD 的方向向量
        # 计算每对麦克风的内积
        inner_product = torch.sum(e_ipd * e_tpd, dim=-1)  # 内积：<e(TPD), e(IPD)>
        return inner_product
 

    def forward(self, mixture, ref=None):
        # mixture_w = self.encoder1(mixture[:,:,0])
        mixture_w = self.encoder2(mixture)
        ipd = self.ipd(mixture) 
        tpd = self.tpd(ref) 
        V=self.compute_V_time_domain(ipd,tpd)

        texts = [
                f"extract {spk_mapping[ref[i, 0].item()]} and {sector_mapping[ref[i, 1].item()][2]} speech"
                for i in range(ref.size(0))
                ]
        text_embed = self.textmodel.get_text_embedding(texts, use_tensor=True)
        est_mask = self.separator(mixture_w,text_embed,ipd,V)
        est_source = self.decoder(mixture_w, est_mask)
        T_origin = mixture.size(1)
        T_conv = est_source.size(1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source

class TimeDomainIPD(nn.Module):
    def __init__(self, kernel_size, num_filters):
        super(TimeDomainIPD, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        # 可学习窗函数（初始为 hamming 窗）
        window = torch.hamming_window(kernel_size)
        # self.window = nn.Parameter(window, requires_grad=True)
        self.register_buffer("window", window)
        # 构造余弦和正弦表
        n = torch.arange(kernel_size).unsqueeze(1)
        k = torch.arange(num_filters).unsqueeze(0)
        w_cos = torch.cos(2 * math.pi * n * k / kernel_size)  # [T, K]
        w_sin = torch.sin(2 * math.pi * n * k / kernel_size)  # [T, K]

        self.register_buffer("cos_table", w_cos.T)  # [K, T]
        self.register_buffer("sin_table", w_sin.T)  # [K, T]

    def forward(self, mixture):
        B, T, C = mixture.shape
        mixture = mixture.permute(0, 2, 1)  # => [B, C, T]

        kre = (self.window[None, :] * self.cos_table).unsqueeze(1)  # [K, 1, T]
        kim = (self.window[None, :] * self.sin_table).unsqueeze(1)

        ipd_features = []

        for (p1, p2) in mic_pairs:
            y1 = mixture[:, p1, :].unsqueeze(1)  # [B, 1, T]
            y2 = mixture[:, p2, :].unsqueeze(1)

            real_y1 = F.conv1d(y1, kre, stride=self.kernel_size // 2)
            imag_y1 = F.conv1d(y1, kim, stride=self.kernel_size // 2)
            real_y2 = F.conv1d(y2, kre, stride=self.kernel_size // 2)
            imag_y2 = F.conv1d(y2, kim, stride=self.kernel_size // 2)

            ipd1 = torch.atan2(imag_y1, real_y1)  # [B, K, T']
            ipd2 = torch.atan2(imag_y2, real_y2)
            ipd = ipd1 - ipd2
            ipd_features.append(ipd)

        ipd_features = torch.stack(ipd_features, dim=1)  # [B, num_pairs, K, T']
        return ipd_features


class TPDLayer(nn.Module):
    def __init__(self,
                 mic_pairs=[(0, 1), (0, 2), (0,3),(1,2),(1,3),(2, 3)]  ,
                 radius=0.05,
                 fs=16000,
                 v=343,
                 n_mics=4):
        super().__init__()
        self.fs = fs
        self.freqs = torch.linspace(0, fs // 2, 256)
        self.v = v
        self.mic_pairs = mic_pairs
        self.radius = radius
        self.n_mics = n_mics
        self.room_sz = [6, 6, 3]
        self.circle_radius=0.05
        # 预计算并缓存麦克风位置
        self.register_buffer("mic_positions", self._get_mic_positions())  # [4, 3]
        # 预计算并缓存每对麦克风间的相对向量 Δ(p)
        self.register_buffer("delta_vecs", self._get_delta_vectors())  # [P, 3]

    def _get_mic_positions(self):
        mic_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        mic_positions = np.array([
            [self.room_sz[0] / 2 + self.circle_radius * np.cos(a),
             self.room_sz[1] / 2 + self.circle_radius * np.sin(a),
             1.7] for a in mic_angles
        ])
        return torch.tensor(mic_positions, dtype=torch.float32)

    def _get_delta_vectors(self):
        """计算所有 mic 对之间的 Δ(p) 向量"""
        delta_list = []
        for (p1, p2) in self.mic_pairs:
            delta = self.mic_positions[p1] - self.mic_positions[p2]  # [3]
            delta_list.append(delta)
        return torch.stack(delta_list, dim=0)  # [P, 3]
   
    def forward(self, text):
        device = self.delta_vecs.device
        freqs = self.freqs.to(device)
        z1 = torch.tensor([
            float(sector_mapping[text[i, 1].item()][0])
            for i in range(text.size(0))
        ], dtype=torch.float32).unsqueeze(1)  # [B, 1]

        z2 = torch.tensor([
            float(sector_mapping[text[i, 1].item()][1])
            for i in range(text.size(0))
        ], dtype=torch.float32).unsqueeze(1)  # [B, 1]
        t1 = torch.cat([z1, z2], dim=1).to(device)

        phi = torch.zeros((t1.size(0), 2))
        phi[:, 0] = -90
        phi[:, 1] = 90
        t2=phi.to(device)
        N=8
        theta_samples = torch.linspace(0, 1, N, device=device).unsqueeze(0) * (t1[:, 1] - t1[:, 0]).unsqueeze(1) + t1[:, 0].unsqueeze(1)
        phi_samples   = torch.linspace(0, 1, N, device=device).unsqueeze(0) * (t2[:, 1] - t2[:, 0]).unsqueeze(1) + t2[:, 0].unsqueeze(1)

        theta_rad = torch.deg2rad(theta_samples)  # [B, N]
        phi_rad   = torch.deg2rad(phi_samples)    # [B, N]

        cos_theta = torch.cos(theta_rad)
        sin_theta = torch.sin(theta_rad)
        cos_phi   = torch.cos(phi_rad)
        sin_phi   = torch.sin(phi_rad)
        u = torch.stack([
            cos_theta * cos_phi,  # x
            sin_theta * cos_phi,  # y
            sin_phi               # z
        ], dim=-1)  # [B, N, 3]
        d_p = torch.einsum('pc,bnc->bnp', self.delta_vecs, u)  # [B, N, P]
        tau = d_p * self.fs / self.v  # [B, N, P]
        tpd = 2 * torch.pi * tau.unsqueeze(-1) * freqs.view(1, 1, 1, -1)  # [B, N, P, F]
        tpd =torch.mean(tpd,dim=1)
        return tpd
 
class Encoder1(nn.Module):
    def __init__(self, L, N):
        super(Encoder1, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
       
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w

class Encoder2(nn.Module):
    def __init__(self, L, N):
        super(Encoder2, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(4, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture = mixture.permute(0, 2,1) # [M, 1, T]
       
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w

class Decoder(nn.Module):
    def __init__(self, args, N, L):
        super(Decoder, self).__init__()
        self.N, self.L, self.args = N, L, args
        self.basis_signals = nn.Linear(N, L, bias=False)
        self.cfw =CFW_TCN(in_ch=256, out_ch=256, num_block=64,num_filters = 64)
    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]
        Returns:
            est_source: [M, C, T]
        """
        est_mask = self.cfw(mixture_w,est_mask,0.25)
        est_source = mixture_w * est_mask
        est_source = torch.transpose(est_source, 2, 1) # [M,  K, N]
        est_source = self.basis_signals(est_source)  # [M,  K, L]
        est_source = overlap_and_add(est_source, self.L//2) # M x C x T
        return est_source

class Dual_RNN_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, out_channels,
                 hidden_channels, rnn_type='LSTM',
                 dropout=0, bidirectional=False, num_spks=2):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # Norm
        self.intra_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.inter_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        # Linear
        self.intra_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        self.inter_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        

    def forward(self, x):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)
        
        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out

class rnn(nn.Module):
    def __init__(self,args, N, B, H, K, R):
        super(rnn, self).__init__()
        self.K , self.R , self.args = K, R, args
        # [M, N, K] -> [M, N, K]
        self.layer_norm = nn.GroupNorm(1, N, eps=1e-8)
        # [M, N, K] -> [M, B, K]
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        self.fuse_layer = FuseLayer(N, N, N)
        self.s_conv = nn.Conv1d(B+24, B, 1, bias=False)
        self.dual_rnn = nn.ModuleList([])
        for i in range(R):
            self.dual_rnn.append(Dual_RNN_Block(B, H,
                                     rnn_type='LSTM',  dropout=0,
                                     bidirectional=True))

        self.prelu = nn.PReLU()

        self.mask_conv1x1 = nn.Conv1d(B, N, 1, bias=False)

        # audio visual projection layer
        self.av_conv = nn.Conv1d(B+24, B, 1, bias=False)


    def forward(self, x, text,ipd,V):
        M, N, D = x.size()
        x = self.layer_norm(x) # [M, N, K]
        x = self.bottleneck_conv1x1(x) # [M, B, K]
        # spa
        V= V.real.float()
        V = torch.mean(V,dim=2).repeat(1, 2, 1)
        ipd = torch.mean(ipd,dim=2).repeat(1, 2, 1)
        x = torch.cat((x,ipd,V),1)
        x = self.s_conv(x)
        #sem
        x = self.fuse_layer(x, text)
        x, gap = self._Segmentation(x, self.K) # [M, B, k, S]

        for i in range(self.R):
            x = self.dual_rnn[i](x)

        x = self._over_add(x, gap)

        x = self.prelu(x)
        x = self.mask_conv1x1(x)
        
        x = F.relu(x)
        return x

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap


    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input



class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.

    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where

        output_size = (frames - 1) * frame_step + frame_length

    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length

    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long().cuda()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_layers=6, num_filters=64):
        super(TCN, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels, num_filters, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            in_channels = num_filters

        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(num_filters, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.network(x)
        return self.final_conv(x)


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.conv1 = nn.Conv1d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv1d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = torch.sigmoid(x)  # Using Swish or Sigmoid here, based on your original design
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)
        return x + x_in
class CFW_TCN(nn.Module):
    def __init__(self, in_ch, out_ch, num_block=1, num_filters=64):
        super(CFW_TCN, self).__init__()
        # Define ResBlock1D and TCN layers
        self.encode_enc_1 = ResBlock1D(2 * in_ch, in_ch)
        self.encode_enc_2 = TCN(in_ch, in_ch, num_filters=num_filters)
        self.encode_enc_3 = ResBlock1D(in_ch, out_ch)

    def forward(self, enc_feat, dec_feat, w=0.25):
        # Concatenate along the second dimension (channels dimension)
        enc_feat = self.encode_enc_1(torch.cat([enc_feat, dec_feat], dim=1))
        enc_feat = self.encode_enc_2(enc_feat)
        enc_feat = self.encode_enc_3(enc_feat)
        residual = w * enc_feat
        out = dec_feat + residual
        return out


class FiLM(nn.Module):
    def __init__(self, dim_in=512, hidden_dim=256):
        super(FiLM, self).__init__()
        self.beta = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1)
            )
        self.gamma = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1)
            )
        # self.gamma = nn.Linear(dim_in, hidden_dim)

    def forward(self, hidden_state, embed):
        return self.gamma(embed).unsqueeze(-1) * hidden_state + self.beta(embed).unsqueeze(-1)
    
class GlobalLayerNorm(nn.Module):
    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
           x = x.permute(0, 2, 3, 1).contiguous()
           # N x K x S x C == only channel norm
           x = super().forward(x)
           # N x C x K x S
           x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x

def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)        
class FuseLayer(nn.Module):
    def __init__(self,in_channels, out_channels, bottle_neck, norm='ln', fuse_type="BERT"):
        super(FuseLayer, self).__init__()
        self.fuse_type = fuse_type
        self.norm_audio = select_norm(norm, in_channels, 3)

        self.cue_down_sample2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.film1 = FiLM(512, in_channels) # 512为CLAP输出文本的维度
        self.layer_norm1 = nn.GroupNorm(1, in_channels, eps=1e-8)
        self.film2 = FiLM(in_channels, 128)
        self.layer_norm2 = nn.GroupNorm(1, 128, eps=1e-8)
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d1 = nn.Conv1d(in_channels, 128, 1, bias=False)
        self.conv1d2 = nn.Conv1d(128, out_channels, 1, bias=False)
            
        self.out_channels = out_channels
        
    def forward(self, x, cue):
        # [bs, dim, seq_len_a]
        x = self.norm_audio(x)
        # [bs, 256, seq_len]
        x = self.norm(x)
        # [bs, 256, seq_len]
        x = self.film1(x, cue) + x
        x = self.layer_norm1(x)
        # [bs, 128, sqe_len]
        x = self.conv1d1(x)
        # [bs, 128]
        cue = self.cue_down_sample2(cue)
        # [bs, 128, seq_len]
        x = self.film2(x, cue) + x
        x = self.layer_norm2(x)
        # [bs, 64, seq_len]
        x = self.conv1d2(x)
        return x