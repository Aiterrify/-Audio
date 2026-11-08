# -*- coding: utf-8 -*-
"""RVC 变声质量对照：同一段源音频，分别用「旧流程(index_rate=0)」与
「完整流程(index_rate=0.75 + protect + rms)」转换，再和参考音频比指标。

指标说明（都是越小/越接近越像）：
    centroid   频谱重心，偏亮说明齿音/高频被 vocoder 放大
    mcd        倒谱失真（dB），衡量音色包络差异，经验上 < 8 才算接近
    f0 中位    用自相关独立测基频，确认没有整体跑调
"""
import os
import sys

import numpy as np
import librosa
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from infer import RVCInfer, load_model, load_index  # noqa: E402

# GPT-SoVITS 源码根目录：优先读环境变量 GSV_ROOT，未设置时回退到同级的 gsv-src
GSV_ROOT = os.environ.get("GSV_ROOT") or os.path.join(os.path.dirname(_HERE), "gsv-src")
HUBERT = os.path.join(GSV_ROOT, "GPT_SoVITS", "pretrained_models", "chinese-hubert-base")
RMVPE = os.path.join(os.path.dirname(_HERE), "assets", "rmvpe.onnx")


def acf_f0(x, sr):
    x = x - x.mean()
    if np.max(np.abs(x)) < 1e-6:
        return 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= (ac[0] + 1e-9)
    lo, hi = int(sr / 600), int(sr / 60)
    seg = ac[lo:hi]
    if len(seg) == 0:
        return 0.0
    return sr / (lo + int(np.argmax(seg)))


def f0_stats(a, sr):
    a16 = librosa.resample(a, orig_sr=sr, target_sr=16000) if sr != 16000 else a
    win, hop, out = 640, 320, []
    for i in range(0, max(0, len(a16) - win), hop):
        seg = a16[i:i + win]
        if np.sqrt(np.mean(seg ** 2)) < 0.02:
            continue
        f = acf_f0(seg, 16000)
        if 60 <= f <= 600:
            out.append(f)
    return float(np.median(out)) if out else 0.0


def logmel(a, sr):
    a16 = librosa.resample(a, orig_sr=sr, target_sr=16000) if sr != 16000 else a
    m = librosa.feature.melspectrogram(y=a16, sr=16000, n_mels=64, fmin=40, fmax=7600)
    return librosa.power_to_db(m)


def metrics(ref, sr_ref, out, sr_out):
    m1, m2 = logmel(ref, sr_ref), logmel(out, sr_out)
    n = min(m1.shape[1], m2.shape[1])
    a, b = m1[:, :n].ravel(), m2[:, :n].ravel()
    cos = float((a - a.mean()) @ (b - b.mean()) /
                (np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean()) + 1e-9))
    mcd = float(np.mean(np.abs(a - b)))
    c1 = float(np.median(librosa.feature.spectral_centroid(
        y=librosa.resample(ref, orig_sr=sr_ref, target_sr=16000) if sr_ref != 16000 else ref,
        sr=16000)))
    o16 = librosa.resample(out, orig_sr=sr_out, target_sr=16000) if sr_out != 16000 else out
    c2 = float(np.median(librosa.feature.spectral_centroid(y=o16, sr=16000)))
    return dict(cos=cos, mcd=mcd, cent_ref=c1, cent_out=c2,
                f0_ref=f0_stats(ref, sr_ref), f0_out=f0_stats(out, sr_out),
                rms=float(np.sqrt(np.mean(out ** 2))),
                clip=float(np.mean(np.abs(out) > 0.999)))


def main():
    src_wav = os.path.join(_HERE, "bench_ref.wav")
    if not os.path.exists(src_wav):
        print("缺少 bench_ref.wav（先用 ffmpeg 从参考音频抽出 10s）")
        return

    # 待测模型由环境变量指定，避免把个人模型路径写进仓库
    #   RVC_PTH   必填，RVC 声学模型路径
    #   RVC_INDEX 可选，检索索引路径
    pth = os.environ.get("RVC_PTH", "")
    index_path = os.environ.get("RVC_INDEX", "")
    if not pth or not os.path.exists(pth):
        print("未指定待测模型。请设置 RVC_PTH（及可选 RVC_INDEX）后重跑：")
        print('  set RVC_PTH=D:\\models\\your\\model.pth')
        return

    model, meta = load_model(pth, device="cpu")
    infer = RVCInfer(HUBERT, RMVPE, device="cpu")
    index, big = load_index(index_path)
    print("index vectors:", 0 if index is None else index.ntotal)

    wav, sr = sf.read(src_wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)

    base = metrics(wav, sr, wav, sr)
    print("\n[参考自身] centroid=%.0f f0=%.1f" % (base["cent_ref"], base["f0_ref"]))

    variants = [
        ("旧流程 index=0", dict(index=None, big_npy=None, index_rate=0.0,
                            protect=0.0, rms_mix_rate=0.0)),
        ("+index 0.75", dict(index=index, big_npy=big, index_rate=0.75,
                             protect=0.0, rms_mix_rate=0.0)),
        ("+index+protect+rms", dict(index=index, big_npy=big, index_rate=0.75,
                                    protect=0.33, rms_mix_rate=0.25)),
    ]
    for name, kw in variants:
        out, osr = infer.convert(wav, sr, model, sid=0, **kw)
        m = metrics(wav, sr, out, osr)
        print("[%-18s] centroid %4.0f -> %4.0f | mcd %5.2f | cos %5.3f | "
              "f0 %5.1f -> %5.1f | rms %.3f | clip %.2f%%"
              % (name, m["cent_ref"], m["cent_out"], m["mcd"], m["cos"],
                 m["f0_ref"], m["f0_out"], m["rms"], m["clip"] * 100))
        sf.write(os.path.join(_HERE, "bench_%s.wav" % name.replace(" ", "_").replace("+", "p")),
                 out, osr)


if __name__ == "__main__":
    main()
