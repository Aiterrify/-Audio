# -*- coding: utf-8 -*-
"""
验证脚本：用独立的 pyin 方法分别提取「源音频 A」与「RVC 输出 B」的基频，
比较两者音高轮廓是否一致。

若 A 与 B 的基频高度相关（整体平移很小），说明 RVC 级联正确保留了旋律，
即便 RMVPE 内部 f0 的绝对标定与 pyin 有差异——因为模型训练/推理都使用 RMVPE，
内部一致即可。

若 B 相对 A 系统性平移（如整体高/低一个八度或几 semitone），才是真正的 bug。
"""
import os
import numpy as np
import librosa
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))

SR_TARGET = 16000  # pyin 在 16k 上最稳


def extract_f0_pyin(path, sr=SR_TARGET):
    wav, _ = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if _ != sr:
        wav = librosa.resample(wav, orig_sr=_, target_sr=sr)
    wav = librosa.util.normalize(wav)
    # pyin 返回 f0、voiced_flag、voiced_prob
    f0, voiced, _ = librosa.pyin(
        wav, fmin=librosa.note_to_hz("C1"), fmax=librosa.note_to_hz("C7"),
        sr=sr, frame_length=1024, hop_length=160,
    )
    f0 = np.where(voiced, f0, 0.0)
    return f0  # 100Hz 帧率（hop=160 @16k）


def stats(f0):
    v = f0[f0 > 0]
    if len(v) == 0:
        return dict(n=0, mean=0, median=0, min=0, max=0, voiced_frac=0.0)
    return dict(
        n=len(v), mean=float(v.mean()), median=float(np.median(v)),
        min=float(v.min()), max=float(v.max()),
        voiced_frac=float(len(v) / len(f0)),
    )


def log_cents_diff(a, b):
    """对同向有声帧，计算 b/a 的 cents 偏差（应接近 0 才说明音高被保留）。"""
    a = np.asarray(a)
    b = np.asarray(b)
    m = (a > 0) & (b > 0)
    if m.sum() < 10:
        return None
    cents = 1200.0 * np.log2(b[m] / a[m])
    return dict(
        n=int(m.sum()),
        mean_cents=float(cents.mean()),
        median_cents=float(np.median(cents)),
        std_cents=float(cents.std()),
        p05=float(np.percentile(cents, 5)),
        p95=float(np.percentile(cents, 95)),
    )


def main():
    src = os.path.join(os.path.dirname(HERE), "test_src_clip.wav")
    out = os.path.join(HERE, "test_dekesasi.wav")
    if not os.path.exists(out):
        print("ERROR: 找不到", out, "请先运行 infer.py")
        return

    f0_src = extract_f0_pyin(src)
    f0_out = extract_f0_pyin(out)

    # 对齐长度
    n = min(len(f0_src), len(f0_out))
    f0_src = f0_src[:n]
    f0_out = f0_out[:n]

    print("=== 源音频 A (test_src_clip.wav) pyin 基频 ===")
    for k, v in stats(f0_src).items():
        print(f"  {k}: {v}")
    print("=== RVC 输出 B (test_dekesasi.wav) pyin 基频 ===")
    for k, v in stats(f0_out).items():
        print(f"  {k}: {v}")
    print("=== A vs B 音高一致性（独立 pyin 比较，单位 cents）===")
    d = log_cents_diff(f0_src, f0_out)
    if d is None:
        print("  同向有声帧不足，无法比较")
    else:
        for k, v in d.items():
            print(f"  {k}: {v}")
        if abs(d["median_cents"]) < 100 and d["std_cents"] < 200:
            print("  结论：B 与 A 音高基本一致（中位偏差 <100¢，标准差 <200¢）"
                  "-> RVC 级联正确保留了旋律。")
        else:
            print("  结论：B 相对 A 存在系统性音高漂移，需排查 RMVPE/coarse 映射。")


if __name__ == "__main__":
    main()
