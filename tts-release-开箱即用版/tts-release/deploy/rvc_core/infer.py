# -*- coding: utf-8 -*-
"""
RVC 推理封装：内容(HuBERT) + 音高(RMVPE onnx) 提取，以及级联转换。

级联定义（用户所选）：
   文本 -> GPT-SoVITS(基准音色) 产出干声 A  ->  RVC(目标音色) 把 A 的
   内容/音高重合成，得到目标音色音频 B。

本模块只负责「给定一段源音频 A，用某个 RVC 模型把它变声为目标音色」。
上游（sovits_bridge）负责先用 GSV 生成 A，再调用本模块的 convert()。
"""

import os
import sys

import numpy as np
import torch
import librosa
import onnxruntime as ort
from transformers import AutoFeatureExtractor, HubertModel

# ---- GSV 源码路径（models.py 已挂，这里再保险一次）----------------------
GSV_ROOT = os.environ.get("GSV_ROOT", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "gsv-src"))
for _p in (os.path.join(GSV_ROOT, "GPT_SoVITS"), GSV_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# 用显式文件路径加载同目录的 models.py，并注册成唯一模块名。
# 不能用 `import models`：在 sovits_bridge 里 GPT-SoVITS 会先加载，
# 顶层包名 `models` 已被 GSV 内部同名模块占进 sys.modules，`import models`
# 会拿到错的模块（表现为 module 'models' has no attribute 'load_rvc_model'）。
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location("rvc_core_models", os.path.join(_HERE, "models.py"))
rvc_models = _ilu.module_from_spec(_spec)
sys.modules["rvc_core_models"] = rvc_models
_spec.loader.exec_module(rvc_models)  # noqa: E402

_HUBERT_SR = 16000
_N_BINS = 360
# RMVPE 360 个输出是「音分(cents)」：每 20 cents 一档，基准 10 Hz，
# 起点 1997.379 cents -> 32.7Hz(C1)，终点 ~2005Hz(B6)。
_CENTS_BASE = 1997.3794084376191
# RVC emb_pitch(Embedding 256) 用 mel 刻度把 f0 离散到 0..255，训练范围 50~1100Hz。
_F0_BIN_MIN_HZ = 50.0
_F0_BIN_MAX_HZ = 1100.0


def _coarse_to_f0(coarse):
    """离散音高 0..255 -> 中心 f0(Hz)，仅在需要时反向使用。"""
    f0_mel_min = 1127.0 * np.log(1.0 + _F0_BIN_MIN_HZ / 700.0)
    f0_mel_max = 1127.0 * np.log(1.0 + _F0_BIN_MAX_HZ / 700.0)
    f0_mel = np.asarray(coarse, dtype=np.float64) / 255.0 * (f0_mel_max - f0_mel_min) + f0_mel_min
    return 700.0 * (np.exp(f0_mel / 1127.0) - 1.0)


def _f0_to_coarse(f0):
    """连续 f0(Hz) -> 离散 1..255（与 RVC emb_pitch 完全一致）。

    注意 RVC 的映射区间是 1..255 而不是 0..255：静音帧落到 1，bin 0 从未参与训练。
    若把静音映射到 0，会去查一个没训过的 embedding 行，清音段立刻变脏。
    """
    f0 = np.asarray(f0, dtype=np.float64)
    f0_mel = 1127.0 * np.log(1.0 + np.maximum(f0, 0.0) / 700.0)
    f0_mel_min = 1127.0 * np.log(1.0 + _F0_BIN_MIN_HZ / 700.0)
    f0_mel_max = 1127.0 * np.log(1.0 + _F0_BIN_MAX_HZ / 700.0)
    # RVC: (mel - min) * (256-2) / (max-min) + 1  ->  [1, 255]
    coarse = (f0_mel - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0
    coarse = np.clip(np.rint(coarse), 1, 255).astype(np.int64)
    return coarse


def _median_filter(f0, radius=3):
    """f0 中值滤波（对应 RVC filter_radius）。

    只对有声帧滤波，避免把 0（静音）卷进有声边界导致音高跳变。
    不做这一步时，逐帧 f0 的抖动会让源信号产生"沙哑/破音"。
    """
    if radius is None or radius <= 2:
        return np.asarray(f0, dtype=np.float32)
    from scipy.signal import medfilt
    arr = np.asarray(f0, dtype=np.float64)
    voiced = arr > 0
    if voiced.sum() < 3:
        return arr.astype(np.float32)
    sm = medfilt(arr, kernel_size=int(radius))
    return np.where(voiced, sm, arr).astype(np.float32)


def _change_rms(src, sr_src, opt, sr_opt, rate):
    """音量包络对齐（对应 RVC rms_mix_rate）。

    rate=0 完全用合成音频自己的包络，rate=1 完全跟随源音频。
    不做这一步，长句子的音量起伏会和参考干声脱节，听感"忽大忽小"。
    """
    if rate <= 0:
        return opt
    r1 = librosa.feature.rms(y=src, frame_length=max(2, sr_src // 2),
                             hop_length=max(1, sr_src // 2))[0]
    r2 = librosa.feature.rms(y=opt, frame_length=max(2, sr_opt // 2),
                             hop_length=max(1, sr_opt // 2))[0]
    n = len(opt)
    if n == 0 or len(r1) == 0 or len(r2) == 0:
        return opt
    g1 = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(r1)), r1)
    g2 = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(r2)), r2)
    g2 = np.maximum(g2, 1e-3)
    gain = np.power(np.maximum(g1, 1e-6), 1.0 - rate) * np.power(g2, rate - 1.0)
    return (opt * gain).astype(np.float32)


def _protect_unvoiced(opt, src, f0, hop, protect):
    """清音/气声保护（对应 RVC protect）。

    RVC 的 NSF 声码器对齿音、气口这类非周期段重建很差，容易发毛、
    发"电子味"。做法是把无音高帧按 protect 权重混回源音频。
    protect=0 不混（最"纯"但最毛），protect=0.5 半混，>0.5 关闭。
    """
    if protect <= 0:
        return opt
    n = min(len(opt), len(src))
    if n <= 0:
        return opt
    w = np.zeros(n, dtype=np.float32)
    for i, v in enumerate(f0):
        if v < 1.0:
            a, b = i * hop, min((i + 1) * hop, n)
            if a < b:
                w[a:b] = 1.0
    if not w.any():
        return opt
    # 掩码做 10ms 平滑，避免硬切换产生咔哒声
    k = max(1, int(0.010 * (n / max(len(opt), 1)) * hop) or 1)
    kern = np.ones(max(3, 2 * (hop // 20) + 1), dtype=np.float32)
    kern /= kern.sum()
    w = np.convolve(w, kern, mode="same")
    m = np.clip(w * protect, 0.0, 1.0)
    out = opt.copy()
    out[:n] = opt[:n] * (1.0 - m) + src[:n] * m
    return out


def load_index(index_path):
    """加载 RVC 的 faiss 检索库，返回 (index, big_npy)。

    index 里存的就是训练集的 HuBERT 特征；推理时用「最近邻加权平均」
    把它混进当前特征，是把音色拉向目标角色的最关键一步。
    没有这一步，输出只是"模型平均音色"，明显不像本人。
    """
    if not index_path or not os.path.exists(index_path):
        return None, None
    import faiss
    # Windows 版 faiss 用窄字符打开文件，路径含中文会直接报 "could not open"。
    # 因此非 ASCII 路径先复制到纯 ASCII 的缓存目录再读。
    read_path = index_path
    if any(ord(ch) > 127 for ch in index_path):
        import hashlib
        import shutil
        cache_dir = os.environ.get("RVC_INDEX_CACHE") or os.path.join(
            os.path.expanduser("~"), ".cache", "rvc_index")
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError:
            cache_dir = os.path.join(os.path.abspath(os.sep), "rvc_index_cache")
            os.makedirs(cache_dir, exist_ok=True)
        tag = hashlib.md5(index_path.encode("utf-8")).hexdigest()[:12]
        cached = os.path.join(cache_dir, tag + ".index")
        if not os.path.exists(cached) or os.path.getsize(cached) != os.path.getsize(index_path):
            shutil.copyfile(index_path, cached)
        read_path = cached
    idx = faiss.read_index(read_path)
    big = idx.reconstruct_n(0, idx.ntotal)
    return idx, big.astype(np.float32)


def _blend_index(feat_np, index, big_npy, index_rate):
    """feat_np: (T,768)。近邻加权平均后按 index_rate 混合。"""
    k = min(8, int(getattr(index, "ntotal", 8)) or 8)
    score, ix = index.search(feat_np, k)
    weight = np.square(1.0 / np.maximum(score, 1e-6))
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
    return (index_rate * npy + (1.0 - index_rate) * feat_np).astype(np.float32)


def _decode_cents(sal):
    """sal: (T,360) sigmoid 显著性 -> (f0_hz (T,), peak_conf (T,))。

    采用 RMVPE 论文的局部加权平均（峰值 ±4 档），比单纯 argmax 更准：
        cents = 20*bin + 1997.379 ;  f0 = 10 * 2^(cents/1200)
    """
    T = sal.shape[0]
    center = np.argmax(sal, axis=1)
    peak = sal[np.arange(T), center]
    cents = 20.0 * np.arange(_N_BINS) + _CENTS_BASE  # (360,)
    idx = center[:, None] + np.arange(-4, 5)[None, :]  # (T, 9)
    idx = np.clip(idx, 0, _N_BINS - 1)
    w = np.take_along_axis(sal, idx, axis=1)          # (T, 9)
    cw = cents[idx]                                  # (T, 9)
    s = w.sum(axis=1)
    avg_cents = (w * cw).sum(axis=1) / np.maximum(s, 1e-12)
    f0 = 10.0 * np.power(2.0, avg_cents / 1200.0)
    f0[s <= 0] = 0.0
    return f0, peak


class RVCInfer:
    def __init__(self, hubert_dir, rmvpe_path, device="cpu"):
        self.device = device
        # chinese-hubert-base 只有 feature_extractor 配置（无 tokenizer），
        # 用 AutoFeatureExtractor 直接产出归一化 input_values（与 RVC 一致）。
        self.feat = AutoFeatureExtractor.from_pretrained(hubert_dir)
        self.hubert = HubertModel.from_pretrained(hubert_dir).to(device).eval()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        self.sess = ort.InferenceSession(rmvpe_path, sess_options=opts,
                                          providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    # ---- 内容：HuBERT 768 维 -------------------------------------------
    def get_content(self, wav16k):
        """wav16k: float32 mono @16k -> (1, T, 768) @device。"""
        wav16k = np.asarray(wav16k, dtype=np.float32)
        inputs = self.feat(wav16k, sampling_rate=_HUBERT_SR,
                           return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            feat = self.hubert(inputs).last_hidden_state  # (1, T, 768)
        return feat

    # ---- 音高：RMVPE onnx --------------------------------------------
    _RMVPE_PAD = 32  # UNet 下采样总倍数，mel 时间轴需为其整数倍，否则解码端 concat 错位

    def _mel(self, wav16k):
        # 与 RMVPE 训练时一致的预处理：波形放大到 int16 幅度 -> 功率谱 -> mel -> 自然对数
        n_fft, win, hop, n_mels = 1024, 1024, 160, 128
        y = wav16k.astype(np.float32) * 32768.0
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win)
        power = np.abs(stft) ** 2
        mel = librosa.filters.mel(sr=_HUBERT_SR, n_fft=n_fft, n_mels=n_mels,
                                  fmin=30.0, fmax=8000.0, norm=None)
        mel_spec = mel @ power
        mel_log = np.log(mel_spec + 1e-6)
        return mel_log.astype(np.float32)  # (128, T)

    def get_pitch(self, wav16k, thred=0.03, method="rmvpe"):
        """返回 (f0_hz (1,T), coarse (1,T))。无音高帧 f0=0。

        method="rmvpe"：神经网络逐帧分类，抗噪好、对气声宽容；
        method="pyin" ：librosa 概率 YIN，干净人声上更稳，可用来交叉校验。
        """
        wav16k = np.asarray(wav16k, dtype=np.float32)
        if method == "pyin":
            f0 = librosa.pyin(wav16k, fmin=float(_F0_BIN_MIN_HZ),
                              fmax=float(_F0_BIN_MAX_HZ), sr=_HUBERT_SR,
                              frame_length=1024, hop_length=160,
                              center=True)[0]
            f0 = np.nan_to_num(np.asarray(f0, dtype=np.float32), nan=0.0)
            return f0[None].astype(np.float32), _f0_to_coarse(f0)[None]
        if method == "harvest":
            import pyworld as pw
            f0, _t = pw.harvest(np.asarray(wav16k, dtype=np.float64), _HUBERT_SR,
                                f0_floor=float(_F0_BIN_MIN_HZ),
                                f0_ceil=float(_F0_BIN_MAX_HZ),
                                frame_period=10.0)
            f0 = np.asarray(f0, dtype=np.float32)
            return f0[None], _f0_to_coarse(f0)[None]
        mel = self._mel(wav16k)  # (128, T)
        T = mel.shape[1]
        if T < 1:
            return np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.int64)
        # 补齐到 32 的整数倍，避免 RMVPE UNet 解码端 concat 维度错位
        pad = (self._RMVPE_PAD - T % self._RMVPE_PAD) % self._RMVPE_PAD
        if pad:
            mel = np.pad(mel, ((0, 0), (0, pad)), mode="constant")
        onnx_in = mel[None]  # (1,128,T')
        out = self.sess.run([self.out_name], {self.in_name: onnx_in})[0]  # (1,T',360)
        if out.shape[1] == _N_BINS:  # 兼容极少见布局 (1,360,T')
            out = out.transpose(0, 2, 1)
        sal = out[0]  # (T',360) sigmoid 显著性
        f0, conf = _decode_cents(sal)
        voiced = conf >= thred
        f0 = np.where(voiced, f0, 0.0).astype(np.float32)
        coarse = _f0_to_coarse(f0)
        # 裁回原始长度
        return f0[:T][None].astype(np.float32), coarse[:T][None].astype(np.int64)

    # ---- 核心：源音频 -> RVC 目标音色 --------------------------------
    def convert(self, wav, sr, model, sid, noise_scale=0.66666,
                index=None, big_npy=None, index_rate=0.75,
                filter_radius=3, protect=0.33, rms_mix_rate=0.25,
                f0_method="rmvpe"):
        """源音频 -> RVC 目标音色，完整复刻 RVC 正式推理流程。

        参数含义与 RVC WebUI 的滑块一一对应：
            index_rate    检索特征混入比例，越高越像目标角色（0~1，默认 0.75）
            filter_radius f0 中值滤波半径，抑制音高抖动（>2 生效）
            protect       清音/气声保护，0~0.5，越大越接近原声（默认 0.33）
            rms_mix_rate  音量包络跟随源音频的程度（0~1，默认 0.25）
        返回 (audio float32 @ model.sr, sr)。
        """
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
        if peak > 1e-6:
            wav = wav / peak * 0.95
        if sr != _HUBERT_SR:
            wav16k = librosa.resample(wav, orig_sr=sr, target_sr=_HUBERT_SR)
        else:
            wav16k = wav.astype(np.float32)

        with torch.no_grad():
            content = self.get_content(wav16k)                      # (1,T50,768)
            f0, _ = self.get_pitch(wav16k, method=f0_method)        # (1,T100)

        t = int(f0.shape[1])
        if t < 1:
            raise RuntimeError("音频过短，无法提取特征")

        # ① index 检索：在 50Hz 域把训练集近邻特征混进来（音色相似度关键）
        if index is not None and big_npy is not None and index_rate > 0:
            feat_np = content[0].cpu().numpy().astype(np.float32)
            blended = _blend_index(feat_np, index, big_npy, float(index_rate))
            content = torch.from_numpy(blended).unsqueeze(0).to(self.device)

        # ② 50Hz -> 100Hz。RVC 用 F.interpolate(scale_factor=2)（nearest），
        #    这里必须一致：linear 会把特征抹平，听感发糊、咬字不清。
        content = torch.nn.functional.interpolate(
            content.transpose(1, 2), scale_factor=2, mode="nearest"
        ).transpose(1, 2)

        # ③ 帧数按 RVC 的做法取两者下限，两边一起截断（不要补零，补零会多出静音尾巴）
        p_len = min(int(content.shape[1]), t)
        content = content[:, :p_len, :].to(self.device)

        # ④ f0 中值滤波：去掉逐帧抖动，否则源信号沙哑
        f0_f = _median_filter(f0[0], filter_radius)[None][:, :p_len]
        pitchf = torch.from_numpy(f0_f.astype(np.float32)).to(self.device)
        pitch = torch.from_numpy(_f0_to_coarse(f0_f)).to(self.device)
        x_mask = torch.ones(1, 1, p_len, device=self.device)
        t = p_len

        with torch.no_grad():
            audio = model.infer(content, pitch, pitchf,
                                torch.LongTensor([int(sid)]).to(self.device),
                                x_mask, noise_scale)
        audio = audio.squeeze().cpu().numpy().astype(np.float32)
        audio = audio - np.mean(audio)

        out_sr = int(model.sample_rate)
        hop = int(np.prod(getattr(model.dec, "upsample_rates", [out_sr // 100])))

        # ④ 清音/气声保护：无音高帧按 protect 混回源音频
        if protect and protect > 0:
            src_out = librosa.resample(wav, orig_sr=sr, target_sr=out_sr) if sr != out_sr \
                else wav.astype(np.float32)
            n = min(len(audio), len(src_out))
            if n > 0:
                head = _protect_unvoiced(audio[:n], src_out[:n], f0_f[0], hop, float(protect))
                tail = audio[n:] if len(audio) > n else np.zeros(0, dtype=np.float32)
                audio = np.concatenate([head, tail]).astype(np.float32)

        # ⑤ 音量包络对齐
        if rms_mix_rate and rms_mix_rate > 0:
            src_out = librosa.resample(wav, orig_sr=sr, target_sr=out_sr) if sr != out_sr \
                else wav.astype(np.float32)
            audio = _change_rms(src_out, out_sr, audio, out_sr, float(rms_mix_rate))

        # ⑥ 归一化防削波
        peak_out = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak_out > 0.99:
            audio = audio / peak_out * 0.95
        return audio.astype(np.float32), out_sr


def load_model(pth_path, device="cpu"):
    """加载 RVC 模型，返回 (model, meta)。"""
    return rvc_models.load_rvc_model(pth_path, device=device)


if __name__ == "__main__":
    import soundfile as sf

    here = os.path.dirname(os.path.abspath(__file__))
    assets = os.path.join(os.path.dirname(here), "assets")
    hubert_dir = os.path.join(GSV_ROOT, "GPT_SoVITS", "pretrained_models", "chinese-hubert-base")
    rmvpe = os.path.join(assets, "rmvpe.onnx")

    infer = RVCInfer(hubert_dir, rmvpe, device="cpu")

    # 自洽验证需要一段源音频和一个 RVC 模型，均由环境变量提供，仓库里不放具体路径
    #   RVC_PTH    必填，RVC 声学模型路径
    #   RVC_SRC    可选，源音频路径
    pth = os.environ.get("RVC_PTH", "")
    if not pth or not os.path.exists(pth):
        print("未指定模型。请设置 RVC_PTH（及可选 RVC_SRC）后重跑：")
        print('  set RVC_PTH=D:\\models\\your\\model.pth')
        sys.exit(0)
    model, meta = load_model(pth, device="cpu")
    print("meta:", meta)

    wav_path = os.environ.get("RVC_SRC") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_src_clip.wav")
    if not os.path.exists(wav_path):
        print("缺少源音频，请设置 RVC_SRC 指向一段人声 wav：", wav_path)
        sys.exit(0)
    wav, sr = sf.read(wav_path, dtype="float32")  # 参考音频原始采样率
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    audio, out_sr = infer.convert(wav, sr, model, sid=0, noise_scale=0.6)
    print("in sr", sr, "out sr", out_sr, "out dur", len(audio) / out_sr)
    # f0 诊断
    wav16k = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    f0, coarse = infer.get_pitch(wav16k)
    voiced = f0[f0 > 0]
    print("f0 voiced mean/median/min/max:",
          voiced.mean() if len(voiced) else 0,
          np.median(voiced) if len(voiced) else 0,
          voiced.min() if len(voiced) else 0,
          voiced.max() if len(voiced) else 0)
    sf.write(os.path.join(here, "test_dekesasi.wav"), audio, out_sr)
    print("saved test_dekesasi.wav")
