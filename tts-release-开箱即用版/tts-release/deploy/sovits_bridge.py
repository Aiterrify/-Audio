# -*- coding: utf-8 -*-
"""
GPT-SoVITS 推理桥接服务（sidecar）

职责：把 GPT-SoVITS 的 TTS pipeline 包装成一份稳定的 HTTP 接口，
供 Java 主程序调用。上层（Java / Web UI）不感知模型版本、权重路径、
参考音频这些细节，只按 voiceId 说话。

设计取舍：
1. 单 pipeline + 热切换权重，而不是每个音色一个 pipeline。
   一个 v2 模型常驻内存约 1.2-1.8 GB（GPT + SoVITS + HuBERT + BERT），
   两个音色各建一套会翻倍。热切换代价约 2-3 秒，只发生在首次切换时，
   之后缓存命中的请求零开销。
2. 全程一把锁。GPT-SoVITS 的 pipeline 内部有状态（当前 ref audio、
   当前权重），并发调用会互相污染，必须串行化。真并发由 Java 侧的
   任务队列负责排队。
3. 输出 16bit 单声道 WAV。不在这里转 mp3，因为 Java 侧已有 Mp3Joiner，
   格式转换只应发生一处。
"""

import argparse
import io
import json
import logging
import os
import sys
import threading
import time
import wave

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("sovits-bridge")

GSV_ROOT = os.environ.get("GSV_ROOT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gsv-src")

# RVC 级联所需的本地目录（绝对路径，因为 bootstrap() 之后会 chdir 到 GSV_ROOT）
_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
_RVC_CORE_DIR = os.path.join(_DEPLOY_DIR, "rvc_core")
_RVC_ASSETS_DIR = os.path.join(_DEPLOY_DIR, "assets")


def bootstrap(root: str):
    """把 GPT-SoVITS 源码目录挂进 sys.path，并切换工作目录。

    GPT-SoVITS 内部大量使用相对路径（GPT_SoVITS/pretrained_models/xxx），
    且模块之间用 `from AR.models...` 这种顶层导入，必须同时满足两点：
    sys.path 包含仓库根，且 cwd 在仓库根。
    """
    if not os.path.isdir(root):
        raise RuntimeError("GPT-SoVITS 源码目录不存在: %s" % root)
    os.chdir(root)
    # 两个路径缺一不可：仓库根提供 tools/、GPT_SoVITS.tools 这类包，
    # GPT_SoVITS 子目录提供 AR、module、feature_extractor 这些顶层包。
    # 官方 api_v2.py 里也同样是两次 append。
    for p in (os.path.join(root, "GPT_SoVITS"), root):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------- 音色配置

class VoiceRegistry:
    def __init__(self, path: str):
        self.path = self._resolve(path)
        self.voices = {}
        self.default_id = ""
        self._load()

    @staticmethod
    def _resolve(path: str) -> str:
        """把配置路径解析成绝对路径。

        相对路径可能相对当前工作目录，也可能相对脚本目录或项目根目录
        （例如传 deploy/voices.local.json 时基准应是项目根）。
        逐个尝试，返回第一个真实存在的；都不存在则返回基于脚本目录的规范化结果，
        这样报错信息里的路径是有意义的。
        """
        if os.path.isabs(path):
            return path
        here = os.path.dirname(os.path.abspath(__file__))
        project = os.path.dirname(here)
        for base in (os.getcwd(), project, here):
            cand = os.path.normpath(os.path.join(base, path))
            if os.path.exists(cand):
                return cand
        return os.path.normpath(os.path.join(project, path))

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        voices = {}
        for v in data.get("voices", []):
            if v.get("type") == "rvc":
                # RVC 级联音色：本身不持有 GSV 权重，而是一个 RVC 模型 +
                # 一个用于生成干声的基准 GSV 音色(baseVoiceId)。
                for key in ("id", "baseVoiceId", "rvc"):
                    if not v.get(key):
                        raise RuntimeError("RVC 音色缺少字段 %s: %s" % (key, v.get("id")))
                if not (v["rvc"] or {}).get("pth"):
                    raise RuntimeError("RVC 音色缺少 rvc.pth: %s" % v.get("id"))
            else:
                for key in ("id", "gpt", "sovits", "ref", "refText"):
                    if not v.get(key):
                        raise RuntimeError("音色配置缺少字段 %s: %s" % (key, v.get("id")))
            voices[v["id"]] = v
        # 二次校验：RVC 音色的基准音色必须存在且本身是 GSV 音色（不能级联套级联）。
        for v in voices.values():
            if v.get("type") == "rvc":
                b = v.get("baseVoiceId")
                if b not in voices:
                    raise RuntimeError("RVC 音色 %s 的 baseVoiceId=%s 不存在" % (v["id"], b))
                if voices[b].get("type") == "rvc":
                    raise RuntimeError("RVC 音色的基准音色不能还是 RVC: %s" % v["id"])
        if not voices:
            raise RuntimeError("音色池为空")
        default_id = data.get("default") or next(iter(voices))
        if default_id not in voices:
            raise RuntimeError("默认音色 %s 不在音色池中" % default_id)
        # 整体替换而不是原地改，保证重载失败时旧配置仍然可用。
        self.voices = voices
        self.default_id = default_id

    def reload(self):
        """热重载。上传新音色后无需重启进程，代价只是一次文件读取。"""
        self._load()
        log.info("音色配置已重载，共 %d 个音色", len(self.voices))

    def get(self, voice_id: str) -> dict:
        return self.voices.get(voice_id) or self.voices[self.default_id]

    def public_list(self):
        return [
            {
                "id": v["id"],
                "name": v.get("name", v["id"]),
                "desc": v.get("desc", ""),
                "avatar": v.get("avatar", v.get("name", v["id"])[:1]),
                "gender": v.get("gender", ""),
                "lang": v.get("lang", "zh"),
                "type": v.get("type", "gsv"),
                "refDuration": v.get("refDuration"),
                # 前端据此决定情绪下拉是否可用：没有对应参考音频就不给选项，
                # 免得用户选了却没效果。
                "emotions": sorted((v.get("emotions") or {}).keys()),
            }
            for v in self.voices.values()
        ]


# ---------------------------------------------------------------- 推理引擎

class SovitsEngine:
    def __init__(self, registry: VoiceRegistry, device: str = "cpu"):
        # 必须在导入 torch 之前设好线程数，OpenMP 只读一次环境变量。
        os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
        # fast_langdetect 首次会下载模型，走国内镜像否则会卡住。
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        import torch
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        torch.set_num_threads(os.cpu_count() or 4)
        log.info("推理线程数 %d", torch.get_num_threads())

        self.registry = registry
        self.lock = threading.Lock()
        self.current_voice = None  # 当前已加载进 pipeline 的 GSV 音色 id
        self.load_seconds = 0.0
        # RVC 级联：共享一套 HuBERT + RMVPE 特征提取器；按 pth 缓存 RVC 生成器。
        self._rvc_infer = None
        self._rvc_models = {}

        first = registry.get(registry.default_id)
        # TTS_Config 只从 configs["custom"] 取实际配置，缺这个键会静默回落
        # 到内置的 v2 底模（路径不存在再抛 FileNotFoundError），
        # 表现为「明明传了权重却说找不到文件」。
        cfg = {
            "custom": {
                "device": device,
                "is_half": False,
                "version": "v2",
                "t2s_weights_path": first["gpt"],
                "vits_weights_path": first["sovits"],
                "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
                "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
            }
        }
        log.info("加载初始音色: %s", first["id"])
        t0 = time.time()
        self.pipeline = TTS(TTS_Config(cfg))
        self.current_voice = first["id"]
        self.load_seconds = time.time() - t0
        log.info("初始加载完成，耗时 %.1fs", self.load_seconds)

    def ensure_voice(self, voice_id: str):
        """调用方必须已持有 self.lock。

        对 RVC 级联音色，pipeline 实际需要加载的是它的基准 GSV 音色；
        因此 self.current_voice 记录的是「已加载的 GSV 音色」，而非请求的音色 id。
        """
        v = self.registry.get(voice_id)
        gsv_id = v.get("baseVoiceId") if v.get("type") == "rvc" else v["id"]
        if gsv_id == self.current_voice:
            return
        gv = self.registry.get(gsv_id)
        t0 = time.time()
        log.info("切换 GSV 基准音色 -> %s (%s)", gv["id"], gv.get("name"))
        self.pipeline.init_t2s_weights(gv["gpt"])
        self.pipeline.init_vits_weights(gv["sovits"])
        self.current_voice = gv["id"]
        log.info("切换完成，耗时 %.1fs，模型版本 %s", time.time() - t0,
                 getattr(self.pipeline.configs, "version", "?"))

    def _build_gsv_req(self, text, v, speed, temperature, top_k, top_p,
                       fragment_interval, seed, sample_steps, emotion, text_lang):
        """构造 GPT-SoVITS 请求体（普通音色与 RVC 级联的基准音色共用）。

        返回 (req, emo, lang)。情绪只能靠切换参考音频实现，GSV 没有情感标签通道。
        """
        ref_audio = v["ref"]
        prompt_text = v["refText"]
        emo = (emotion or "").strip()
        if emo:
            e = (v.get("emotions") or {}).get(emo)
            if isinstance(e, dict):
                ref_audio = e.get("ref") or ref_audio
                prompt_text = e.get("refText") or prompt_text
            elif isinstance(e, str):
                ref_audio = e
        # 语言同样可以从请求覆盖：翻译成英文后必须让模型按英文音素读，
        # 否则会拿中文声韵去硬念拉丁字母。
        lang = (text_lang or "").strip() or v.get("lang", "zh")
        req = {
            "text": text,
            "text_lang": lang,
            "ref_audio_path": ref_audio,
            "prompt_text": prompt_text,
            "prompt_lang": v.get("refLang", v.get("lang", "zh")),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "temperature": float(temperature),
            "text_split_method": v.get("splitMethod", "cut5"),
            "batch_size": 1,
            "speed_factor": float(speed),
            "fragment_interval": float(fragment_interval),
            "seed": int(seed),
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "streaming_mode": False,
            "media_type": "wav",
            # v2ProPlus 走采样式解码，步数直接影响耗时：32 步 RTF 2.94、
            # 16 步 2.37、8 步 2.10。默认 16 是听感与等待的折中，音色配置可覆盖。
            "sample_steps": int(v.get("sampleSteps", sample_steps)),
        }
        return req, emo, lang

    def _run_pipeline(self, req):
        """调用方必须已持有 self.lock。跑一次 GSV 推理，返回 (int16 pcm, sr)。"""
        chunks = []
        sr = 32000
        for sample_rate, audio in self.pipeline.run(req):
            sr = sample_rate
            chunks.append(audio)
        if not chunks:
            raise RuntimeError("推理未产出音频")
        import numpy as np
        return np.concatenate(chunks).astype("int16"), sr

    # ---- RVC 级联辅助 ----
    def _device(self) -> str:
        return str(getattr(self.pipeline.configs, "device", "cpu"))

    def _ensure_rvc_infer(self):
        if self._rvc_infer is not None:
            return self._rvc_infer
        if _RVC_CORE_DIR not in sys.path:
            sys.path.insert(0, _RVC_CORE_DIR)
        from infer import RVCInfer  # 本地手写 RVC 推理封装（HuBERT + RMVPE）
        hubert = os.path.join(GSV_ROOT, "GPT_SoVITS", "pretrained_models",
                              "chinese-hubert-base")
        rmvpe = os.path.join(_RVC_ASSETS_DIR, "rmvpe.onnx")
        if not os.path.exists(rmvpe):
            raise RuntimeError("缺少 RVC 音高模型 rmvpe.onnx: %s" % rmvpe)
        log.info("加载 RVC 特征提取器 (HuBERT + RMVPE)，device=%s", self._device())
        self._rvc_infer = RVCInfer(hubert, rmvpe, device=self._device())
        return self._rvc_infer

    def _get_rvc_model(self, rvc_cfg):
        pth = rvc_cfg["pth"]
        model = self._rvc_models.get(pth)
        if model is not None:
            return model
        if _RVC_CORE_DIR not in sys.path:
            sys.path.insert(0, _RVC_CORE_DIR)
        from infer import load_model  # infer.load_model -> models.load_rvc_model
        model, meta = load_model(pth, device=self._device())
        log.info("加载 RVC 模型 %s missing=%d unexpected=%d",
                 os.path.basename(pth), len(meta.get("missing", [])),
                 len(meta.get("unexpected", [])))
        self._rvc_models[pth] = model
        return model

    def _get_rvc_index(self, rvc_cfg):
        """加载（并缓存）RVC 的 faiss 检索库。缺失就直接返回 (None, None)。"""
        path = (rvc_cfg or {}).get("index")
        if not path:
            return None, None
        cache = getattr(self, "_rvc_indexes", None)
        if cache is None:
            cache = self._rvc_indexes = {}
        if path in cache:
            return cache[path]
        if _RVC_CORE_DIR not in sys.path:
            sys.path.insert(0, _RVC_CORE_DIR)
        from infer import load_index
        idx, big = load_index(path)
        if idx is None:
            log.warning("RVC index 未找到，音色相似度会明显下降: %s", path)
        else:
            log.info("加载 RVC index %s 向量数=%d", os.path.basename(path), idx.ntotal)
        cache[path] = (idx, big)
        return idx, big

    def _convert_rvc(self, pcm, sr, v, pcm16=True):
        """把源音频换成 RVC 目标音色，返回 (float32 audio, out_sr)。

        pcm16=True 表示入参是 int16 PCM（GSV 干声），False 表示已经是
        [-1,1] 的 float 波形（用户在浏览器里录的声音）。
        """
        import numpy as np
        rvc_cfg = v["rvc"]
        infer = self._ensure_rvc_infer()
        model = self._get_rvc_model(rvc_cfg)
        index, big_npy = self._get_rvc_index(rvc_cfg)
        wav = np.asarray(pcm, dtype=np.float32)
        if pcm16:
            wav = wav / 32768.0
        # 这几个参数就是 RVC WebUI 上的四个滑块，缺一个音色都会明显走样
        return infer.convert(
            wav, sr, model,
            sid=int(rvc_cfg.get("sid", 0)),
            noise_scale=float(rvc_cfg.get("noiseScale", 0.66666)),
            index=index, big_npy=big_npy,
            index_rate=float(rvc_cfg.get("indexRate", 0.75)),
            filter_radius=int(rvc_cfg.get("filterRadius", 3)),
            protect=float(rvc_cfg.get("protect", 0.33)),
            rms_mix_rate=float(rvc_cfg.get("rmsMixRate", 0.25)),
            f0_method=str(rvc_cfg.get("f0Method", "rmvpe")),
        )

    def synthesize(self, text: str, voice_id: str, speed: float = 1.0,
                   temperature: float = 1.0, top_k: int = 15, top_p: float = 1.0,
                   fragment_interval: float = 0.3, seed: int = -1,
                   sample_steps: int = 16, emotion: str = "",
                   text_lang: str = "") -> tuple:
        v = self.registry.get(voice_id)
        t0 = time.time()
        import numpy as np

        if v.get("type") == "rvc":
            # 级联：基准 GSV 音色出干声 A -> RVC 把 A 的内容/音高换成目标音色 B
            base_v = self.registry.get(v["baseVoiceId"])
            req, emo, lang = self._build_gsv_req(
                text, base_v, speed, temperature, top_k, top_p,
                fragment_interval, seed, sample_steps, emotion, text_lang)
            with self.lock:
                self.ensure_voice(voice_id)  # 内部解析到 baseVoiceId 的 GSV 权重
                dry_pcm, dry_sr = self._run_pipeline(req)
                audio, out_sr = self._convert_rvc(dry_pcm, dry_sr, v)
            # 统一到 32000Hz，与主链路（GSV v2 输出）一致，避免下游 Mp3Joiner 采样率假设不符
            if out_sr != 32000:
                import librosa
                audio = librosa.resample(audio, orig_sr=out_sr, target_sr=32000)
                out_sr = 32000
            out = np.clip(audio * 32767.0, -32768.0, 32767.0).astype("int16")
            wav = encode_wav(out.tobytes(), out_sr)
            dur = len(out) / out_sr
            cost = time.time() - t0
            log.info("RVC 合成完成 voice=%s 干声=%.2fs 输出=%.2fs 耗时=%.2fs",
                     v["id"], len(dry_pcm) / dry_sr, dur, cost)
            return wav, dur

        # 普通 GSV 音色
        req, emo, lang = self._build_gsv_req(
            text, v, speed, temperature, top_k, top_p,
            fragment_interval, seed, sample_steps, emotion, text_lang)
        with self.lock:
            self.ensure_voice(voice_id)
            pcm, sr = self._run_pipeline(req)
        wav = encode_wav(pcm.tobytes(), sr)
        cost = time.time() - t0
        dur = len(pcm) / sr
        log.info("合成完成 voice=%s 情绪=%s 语种=%s 字数=%d 音频=%.2fs 耗时=%.2fs RTF=%.2f",
                 v["id"], emo or "-", lang, len(text), dur, cost, cost / max(dur, 0.01))
        return wav, dur

    # ---- 变声：音频 -> 音频（不经 GSV，只有 RVC 音色支持）----
    def convert_audio(self, raw: bytes, voice_id: str, volume: float = 1.0) -> tuple:
        """把用户上传/录制的音频换成 RVC 目标音色，返回 (wav 字节, 时长秒)。

        raw 是完整的 wav 文件字节（浏览器端用 WebAudio 采 PCM 后自己封装）。
        与 synthesize 的区别：这里没有文本、没有 GSV，源音频本身就是内容，
        RVC 只替换音色与音高特征。
        """
        import soundfile as sf
        import librosa
        import numpy as np

        # 不能用 registry.get()：它对未知 id 会静默回落到默认音色，
        # 用户选了 A 却听到 B 的变声结果，排查起来极难。
        v = self.registry.voices.get(voice_id)
        if v is None:
            raise ValueError("音色不存在: %s" % voice_id)
        if v.get("type") != "rvc":
            raise ValueError(
                "「%s」是 GPT-SoVITS 音色，只能从文字生成；变声需要 RVC 模型"
                % v.get("name", voice_id))

        try:
            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception as e:
            raise ValueError("音频解码失败（需要 16bit PCM WAV）：%s" % e)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.size == 0:
            raise ValueError("音频为空")
        if sr <= 0:
            raise ValueError("采样率非法: %s" % sr)

        dur_in = data.size / float(sr)
        if dur_in < 0.3:
            raise ValueError("录音太短（%.2fs），至少 0.3 秒" % dur_in)
        if dur_in > 60:
            raise ValueError("音频过长（%.1fs），上限 60 秒" % dur_in)

        t0 = time.time()
        with self.lock:
            audio, out_sr = self._convert_rvc(data, sr, v, pcm16=False)

        if volume and abs(float(volume) - 1.0) > 1e-3:
            audio = np.clip(audio * float(volume), -1.0, 1.0)
        # 与主链路统一到 32000Hz，前端播放/下载不必关心模型采样率（40k/48k）
        if out_sr != 32000:
            audio = librosa.resample(audio, orig_sr=out_sr, target_sr=32000)
            out_sr = 32000
        out = np.clip(audio * 32767.0, -32768.0, 32767.0).astype("int16")
        wav = encode_wav(out.tobytes(), out_sr)
        dur = len(out) / float(out_sr)
        cost = time.time() - t0
        log.info("变声完成 voice=%s 输入=%.2fs@%d 输出=%.2fs 耗时=%.2fs RTF=%.2f",
                 v["id"], dur_in, sr, dur, cost, cost / max(dur_in, 0.01))
        return wav, dur


def encode_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


# ---------------------------------------------------------------- HTTP 层

def build_app(engine: SovitsEngine, registry: VoiceRegistry):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel

    app = FastAPI(title="Sovits Bridge", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    class TtsReq(BaseModel):
        text: str
        voiceId: str = ""
        # Java 侧 LocalHttpEngine 沿用它自己的字段名 voice，
        # 这里同时接受两种写法，避免上层为一个字段名改动整条调用链。
        voice: str = ""
        speed: float = 1.0
        temperature: float = 1.0
        topK: int = 15
        topP: float = 1.0
        fragmentInterval: float = 0.3
        seed: int = -1
        # 情绪键（angry / sad / breath）与目标语种（zh / en / ja）。
        # 二者都是可选的，缺省即沿用音色自带配置。
        emotion: str = ""
        textLang: str = ""

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "voice": engine.current_voice,
            "voices": len(registry.voices),
            "loadSeconds": round(engine.load_seconds, 1),
            # 让前端能显示真实模型版本（v2 / v2Pro / v2ProPlus 差异明显，
            # 写死一个 v2 会误导）
            "version": getattr(engine.pipeline.configs, "version", ""),
            "device": str(getattr(engine.pipeline.configs, "device", "cpu")),
            "threads": os.cpu_count(),
        }

    @app.get("/voices")
    def voices():
        return {"default": registry.default_id, "voices": registry.public_list()}

    @app.post("/tts")
    def tts(req: TtsReq):
        text = (req.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="文本为空")
        if len(text) > 2000:
            raise HTTPException(status_code=400, detail="文本超长（上限 2000 字）")
        try:
            audio, duration = engine.synthesize(
                text, req.voiceId or req.voice, req.speed, req.temperature,
                req.topK, req.topP, req.fragmentInterval, req.seed,
                emotion=req.emotion, text_lang=req.textLang,
            )
        except Exception as e:  # 模型层异常统一转 500，保留原始信息便于排查
            log.exception("合成失败")
            return JSONResponse(status_code=500, content={"error": str(e)})
        return Response(
            content=audio,
            media_type="audio/wav",
            headers={"X-Audio-Duration": "%.2f" % duration},
        )

    @app.post("/warmup")
    def warmup(body: dict = None):
        vid = (body or {}).get("voiceId") or registry.default_id
        with engine.lock:
            engine.ensure_voice(vid)
        return {"status": "ok", "voice": engine.current_voice}

    @app.post("/convert")
    async def convert(request: Request, voiceId: str = "", volume: float = 1.0):
        """音频 -> 音频变声。请求体是裸 wav 字节，不走 multipart。

        用 async 端点只为了 await 取 body；真正的推理丢到线程池，
        否则会阻塞事件循环，连 /health 都探不到。
        """
        import asyncio

        raw = await request.body()
        if not raw:
            raise HTTPException(status_code=400, detail="音频为空")
        if len(raw) > 30 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="音频过大（上限 30MB）")
        loop = asyncio.get_running_loop()
        try:
            audio, duration = await loop.run_in_executor(
                None, engine.convert_audio, raw, voiceId, volume)
        except ValueError as e:
            # 音色类型不对、音频解码失败这类属于用户输入问题，回 400 而不是 500
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            log.exception("变声失败")
            return JSONResponse(status_code=500, content={"error": str(e)})
        return Response(
            content=audio,
            media_type="audio/wav",
            headers={"X-Audio-Duration": "%.2f" % duration},
        )

    @app.post("/reload")
    def reload_voices():
        """上传新音色后调用，免去重启桥接（重启要重新加载 1.8GB 权重）。"""
        try:
            registry.reload()
        except Exception as e:
            log.exception("重载音色失败")
            raise HTTPException(status_code=500, detail=str(e))
        # 当前音色在新配置里可能已不存在，交给下次请求的 ensure_voice 处理。
        if engine.current_voice not in registry.voices:
            engine.current_voice = None
        return {"status": "ok", "voices": len(registry.voices)}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9880)
    ap.add_argument("--voices", required=True, help="voices.local.json 路径")
    ap.add_argument("--gsv-root", default=GSV_ROOT)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # --gsv-root 同样支持相对路径，依次尝试工作目录、项目根、脚本目录
    gsv_root = args.gsv_root
    if not os.path.isabs(gsv_root):
        here = os.path.dirname(os.path.abspath(__file__))
        project = os.path.dirname(here)
        for base in (os.getcwd(), project, here):
            cand = os.path.normpath(os.path.join(base, gsv_root))
            if os.path.isdir(cand):
                gsv_root = cand
                break

    bootstrap(gsv_root)
    registry = VoiceRegistry(args.voices)
    engine = SovitsEngine(registry, args.device)
    app = build_app(engine, registry)

    import uvicorn
    log.info("桥接服务启动 http://%s:%d  音色=%s", args.host, args.port,
             ",".join(registry.voices))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
