# -*- coding: utf-8 -*-
"""
RVC v2 推理网络（纯手写，复用 GPT-SoVITS 源码里的通用子模块）。

权重结构已逐张核对（斯卡蒂01 / 缄默德克萨斯 两个 pth 完全一致，457 张量）：
  - enc_p  : emb_phone=Linear(768->192) 内容投影（连续 HuBERT 768 维）
             emb_pitch=Embedding(256->192) 音高离散嵌入
             encoder = Conformer（attentions.Encoder, window=10, n_heads=2, 6 层）
             proj    = Conv1d(192->384)  -> 2*inter（m_p ‖ logs_p）
  - flow   : ResidualCouplingBlock(channels=192, hidden=192, kernel=5, WN n_layers=3, n_flows=4, gin=256)
             pre=Conv1d(96->192)  post=Conv1d(192->96)  enc=WN(hidden=192, n_layers=3, gin=256)
  - dec    : GeneratorNSF(HiFiGAN v2 + NSF 源)
             conv_pre/cond/conv_post/noise_convs 为普通 Conv1d
             ups / resblocks 为 weight_norm
             m_source = SourceModuleHnNSF（仅 l_linear 可学习）
  - emb_g  : Embedding(109->256)

坐标约定：全部复用 module.modules / module.attentions 的现成实现，
flow 的 pre/post 与 RVC 训练时的命名一致（已核对，无需 swap）。
"""

import math
import os
import sys

import numpy as np
import torch
from torch import nn

# ---- 把 GPT-SoVITS 源码挂进 sys.path（与 sovits_bridge 同一套约定）-----------
GSV_ROOT = os.environ.get("GSV_ROOT", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "gsv-src"))
for _p in (os.path.join(GSV_ROOT, "GPT_SoVITS"), GSV_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from module import commons  # weight_norm / Conv1d / sequence_mask / fused_add_tanh_sigmoid_multiply
from module.modules import (  # 与 RVC 同源的 VITS 组件
    Conv1d,
    ResBlock1,
    ResidualCouplingLayer,
    Flip,
)
from module.attentions import Encoder as ConformerEncoder  # RVC enc_p.encoder


# --------------------------------------------------------------------------
# 1) SineGen + SourceModuleHnNSF  ——  RVC v2 的谐波+噪声源
# --------------------------------------------------------------------------
class SineGen(nn.Module):
    """无参数的谐波正弦+噪声源生成器，按输出采样率生成。

    f0: (B, T) Hz，T 为模型内部帧率（100Hz）。输出长度 = T * (sr/100)。

    注意正弦幅度必须固定 0.1、噪声仅作用于清音帧：
    原版 RVC 的 noise_std=0.003，若把噪声提到与正弦同量级（0.1），
    源信号会变成「半白噪声」，解码器再放大后就是刺耳的高频噪声。
    """

    def __init__(self, sample_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    def forward(self, f0):
        # f0: (B, T) 帧率 -> 上采样到采样率后做相位积分
        b, t = f0.shape
        up = int(self.sample_rate // 100)  # 每帧样本数：320 / 400
        total = t * up
        if total <= 0:
            return torch.zeros((b, 1, 0), device=f0.device, dtype=f0.dtype)
        f0_up = torch.nn.functional.interpolate(
            f0.unsqueeze(1), size=total, mode="linear", align_corners=True
        ).squeeze(1)  # (B, total)
        rad_per_sample = 2.0 * math.pi * f0_up / self.sample_rate
        phase = torch.cumsum(rad_per_sample, dim=1)
        if self.harmonic_num <= 0:
            s = torch.sin(phase) * self.sine_amp
        else:
            s = torch.zeros_like(phase)
            for h in range(1, self.harmonic_num + 2):
                s = s + torch.sin(h * phase) * (self.sine_amp / h)
        s = s.unsqueeze(1)  # (B,1,total)
        # 只在清音帧叠加低幅噪声（原版 noise_std=0.003）
        if self.noise_std > 0:
            uv = (f0_up <= self.voiced_threshold).float().unsqueeze(1)
            s = s * (1.0 - uv) + torch.randn_like(s) * self.noise_std * uv
        return s


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sample_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0):
        super().__init__()
        self.sine_gen = SineGen(sample_rate, harmonic_num, sine_amp, noise_std, voiced_threshold)
        # 与 RVC 一致：Linear(harmonic_num+1, 1) 在「每帧的谐波维」上合并，再过 tanh
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        # x: (B, T) 帧率 f0（Hz）
        sine = self.sine_gen(x)                       # (B,1,total)
        sine = sine.transpose(1, 2)                   # (B,total,1)
        sine = self.l_tanh(self.l_linear(sine))       # (B,total,1)
        return sine.transpose(1, 2)                   # (B,1,total)


# --------------------------------------------------------------------------
# 2) enc_p  ——  ContentEncoder（RVC 版）
# --------------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(self, out_channels, hidden_channels, filter_channels, n_heads,
                 n_layers, kernel_size, p_dropout, content_dim=768, n_vocab_pitch=256, f0=1):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.f0 = f0
        # 连续 HuBERT 768 维 -> hidden
        self.emb_phone = nn.Linear(content_dim, hidden_channels)
        # 离散音高 0..255 -> hidden
        self.emb_pitch = nn.Embedding(n_vocab_pitch, hidden_channels)
        self.encoder = ConformerEncoder(
            hidden_channels, filter_channels, n_heads, n_layers,
            kernel_size, p_dropout, window_size=10,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        nn.init.zeros_(self.proj.bias)

    def forward(self, phone, pitch, x_mask):
        # phone: (B, T, content_dim)   pitch: (B, T) 离散
        # 严格对齐 RVC 原版 TextEncoder 的运算顺序：
        #   1) emb_phone + emb_pitch（先相加，未缩放）
        #   2) 整体 * sqrt(hidden)
        #   3) LeakyReLU(0.1)
        #   4) Conformer 编码 -> proj（proj 后不再乘 sqrt(out_channels)）
        x = self.emb_phone(phone)                        # (B, T, hidden)
        if self.f0 == 1 and pitch is not None:
            x = x + self.emb_pitch(pitch)                # (B, T, hidden) 先加音高嵌入
        x = x * math.sqrt(self.hidden_channels)          # (B, T, hidden)
        x = torch.nn.functional.leaky_relu(x, 0.1)       # RVC 在此有 LeakyReLU
        x = x.transpose(1, 2)                            # (B, hidden, T)
        x = x * x_mask
        x = self.encoder(x, x_mask, g=None)
        x = self.proj(x)                                 # 不做 sqrt(out_channels) 缩放（RVC 原版无）
        return x


# --------------------------------------------------------------------------
# 3) dec  ——  GeneratorNSF（RVC v2 HiFiGAN + NSF 源）
# --------------------------------------------------------------------------
class GeneratorNSF(nn.Module):
    def __init__(self, inter_channels, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                 upsample_kernel_sizes, gin_channels, noise_kernels, sample_rate):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.sample_rate = sample_rate

        self.conv_pre = nn.Conv1d(inter_channels, upsample_initial_channel, 7, 1, padding=3)
        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                torch.nn.utils.weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k, u, padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock(ch, k, tuple(d)))

        self.conv_post = nn.Conv1d(
            upsample_initial_channel // (2 ** self.num_upsamples), 1, 7, 1, padding=3, bias=False
        )

        self.noise_convs = nn.ModuleList()
        # 源信号是在「全采样率」下生成的，而每个上采样阶段的时间轴长度不同，
        # 所以每个阶段必须按「剩余上采样倍数」做 strided 卷积下采样才能对齐。
        # 原版 stride = prod(后续 upsample_rates)，权重里 kernel = stride*2 正是这个原因。
        # 这里如果漏掉 stride（按 stride=1 卷积再切片），源信号与内容会整体错位，
        # 输出就变成没有音高的噪声——这正是之前"效果很差"的主因。
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(self.ups):
                stride_f0 = int(np.prod(upsample_rates[i + 1:]))
                k = noise_kernels[i]
                self.noise_convs.append(
                    nn.Conv1d(1, ch, k, stride=stride_f0, padding=stride_f0 // 2)
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, ch, 1))

        self.m_source = SourceModuleHnNSF(sample_rate)

    def forward(self, x, g=None, f0=None):
        x = self.conv_pre(x)
        x = x + self.cond(g)
        # 源信号保持全采样率 (B,1,total)，逐阶段用 strided noise_conv 对齐后相加
        src = self.m_source(f0) if f0 is not None else None
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            if src is not None:
                s_i = self.noise_convs[i](src)
                n = x.size(-1)
                if s_i.size(-1) > n:
                    s_i = s_i[:, :, :n]
                elif s_i.size(-1) < n:
                    s_i = torch.nn.functional.pad(s_i, (0, n - s_i.size(-1)))
                x = x + s_i
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.conv_post(torch.nn.functional.leaky_relu(x, 0.1))
        return torch.tanh(x)


# --------------------------------------------------------------------------
# 4) flow  ——  ResidualCouplingBlock（直接复用 GSV 实现）
# --------------------------------------------------------------------------
class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels, hidden_channels, kernel_size, dilation_rate,
                    n_layers, gin_channels=gin_channels, mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


# --------------------------------------------------------------------------
# 5) 顶层 SynthesizerTrn
# --------------------------------------------------------------------------
class SynthesizerTrn(nn.Module):
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels,
                 split_bins, n_layers_flow, n_layers_text, n_heads, kernel_text,
                 resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                 upsample_initial_channel, upsample_kernel_sizes, n_speakers,
                 gin_channels, sample_rate, noise_kernels):
        super().__init__()
        self.inter_channels = inter_channels
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.sample_rate = sample_rate

        self.enc_p = TextEncoder(
            out_channels=2 * inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=split_bins,  # 768
            n_heads=n_heads,
            n_layers=n_layers_text,
            kernel_size=int(kernel_text) if str(kernel_text).isdigit() else 3,
            p_dropout=0.1,
            content_dim=split_bins,
        )
        self.dec = GeneratorNSF(
            inter_channels, ResBlock1, resblock_kernel_sizes,
            resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
            upsample_kernel_sizes, gin_channels, noise_kernels, sample_rate,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, kernel_size=5, dilation_rate=1,
            n_layers=3, n_flows=n_layers_flow, gin_channels=gin_channels,
        )
        self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def infer(self, phone, pitch, pitchf, sid, x_mask, noise_scale=0.6):
        g = self.emb_g(sid).unsqueeze(-1)  # (B, gin, 1)
        z_p_full = self.enc_p(phone, pitch, x_mask)  # (B, 2*inter, T)
        m_p, logs_p = torch.split(z_p_full, self.inter_channels, dim=1)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, x_mask, g=g, reverse=True)  # (B, inter, T)
        o = self.dec(z, g=g, f0=pitchf)  # (B, 1, T*sr/100)
        return o


# --------------------------------------------------------------------------
# 6) 加载器：从 pth 读取 config + 权重 shape，构建并载入
# --------------------------------------------------------------------------
def _read_cfg(cpt):
    cfg = cpt["config"]
    sr_raw = cpt.get("sr")
    if isinstance(sr_raw, str):
        sr = int(sr_raw.rstrip("k")) * 1000
    else:
        sr = int(sr_raw)
    return cfg, sr


def load_rvc_model(pth_path, device="cpu"):
    cpt = torch.load(pth_path, map_location="cpu", weights_only=True)
    cfg, sr = _read_cfg(cpt)
    sd = cpt["weight"]

    spec_channels = cfg[0]
    segment_size = cfg[1]
    inter_channels = cfg[2]
    hidden_channels = cfg[3]
    split_bins = cfg[4]          # 768
    # 耦合层数量以「权重里实际存在的 flow.flows 偶数索引」为准。
    # cfg[6] 在 RVC v2 里常写成 6，但实测 pth 只有 4 个 ResidualCouplingLayer
    # （flow.flows 索引 0,2,4,6），若按 6 构建会有 2 层随机未训练权重。
    _flow_idx = sorted(
        set(int(k.split(".")[2]) for k in sd if k.startswith("flow.flows."))
    )
    if _flow_idx:
        n_layers_flow = len([i for i in _flow_idx if i % 2 == 0])
    else:
        n_layers_flow = int(cfg[6])
    # cfg[7]/cfg[9] 这两个字段在 RVC 的 list-config 里经常是占位默认值，
    # 与真正训练出的结构对不上（实测是 6 层、kernel=3）。一律从权重反推：
    #   - enc_p.encoder.ffn_layers 的数量 = 编码器层数
    #   - ffn_layers.0.conv_1 的卷积核宽 = kernel_size
    _ffn_keys = [k for k in sd if k.startswith("enc_p.encoder.ffn_layers.")]
    n_layers_text = len(set(k.split(".")[3] for k in _ffn_keys))
    kernel_text = int(sd["enc_p.encoder.ffn_layers.0.conv_1.weight"].shape[-1])
    n_heads = 2                  # 由 emb_rel_k(1,21,96) 推出 (192/96)
    resblock_kernel_sizes = cfg[10]
    resblock_dilation_sizes = cfg[11]
    upsample_rates = cfg[12]
    upsample_initial_channel = cfg[13]
    upsample_kernel_sizes = cfg[14]
    n_speakers = cfg[15]
    gin_channels = cfg[16]

    # noise_convs 的 kernel 直接从权重读，避免猜测
    noise_kernels = [
        int(sd[f"dec.noise_convs.{i}.weight"].shape[-1])
        for i in range(len(upsample_rates))
    ]

    model = SynthesizerTrn(
        spec_channels=spec_channels, segment_size=segment_size,
        inter_channels=inter_channels, hidden_channels=hidden_channels,
        split_bins=split_bins, n_layers_flow=n_layers_flow,
        n_layers_text=n_layers_text, n_heads=n_heads, kernel_text=kernel_text,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        upsample_rates=upsample_rates, upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=upsample_kernel_sizes, n_speakers=n_speakers,
        gin_channels=gin_channels, sample_rate=sr, noise_kernels=noise_kernels,
    )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    return model, {"sr": sr, "gin_channels": gin_channels, "n_speakers": n_speakers,
                   "missing": missing, "unexpected": unexpected}
