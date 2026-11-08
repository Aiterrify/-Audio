[README.md](https://github.com/user-attachments/files/34539021/README.md)
# -Audio# 文音 Audio · 本地 AI 配音工作台

![Java](https://img.shields.io/badge/Java-17-orange)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![CPU](https://img.shields.io/badge/Runs%20on-CPU-lightgrey)

一个自己用的本地配音工作台：浏览器里输入文字，用自己训练的 GPT-SoVITS / RVC 音色生成音频。**全程离线，不出内网，不按字符数计费。**

## 为什么写这个

市面上的 TTS 服务基本卡在三个地方：

| 痛点 | 具体表现 |
|---|---|
| **按量计费** | 有声书动辄十万字，云端一个字一分钱，一千块起步 |
| **音色固定** | 官方音色库就那么几个，想要特定角色的声音只能自己训，但训完了没处放 |
| **文本上云** | 未发布的小说稿、内部资料，扔进第三方接口总归不放心 |

GPT-SoVITS 这类开源方案解决了模型问题，但生态里大多是命令行脚本、Gradio 页面，一次只能合成一段，声音配置散落在各种 yaml 里。真正写长稿的时候，缺的是一个能管音色、能连续出稿、能随时切角色的工作台。

这个项目就是补这一层。

## 特性

| | |
|---|---|
| **多音色池** | 统一管理 GPT-SoVITS 与 RVC 两类模型，热切换约 1.7s，无需重启 |
| **情绪控制** | 切参考音频实现（生气 / 开心 / 难过 / 喘息），不是估测参数 |
| **跨语种合成** | 中文模型读英文、日文模型读中文，`text_lang` 与 `prompt_lang` 解耦 |
| **批量配音** | `---` 分隔逐段生成，适合长文与多角色稿 |
| **变声器** | 麦克风录音或上传音频，走 RVC 单独通道 |
| **翻译** | 中 → 英 / 日，免费通道或自建 OpenAI 兼容接口 |
| **内网穿透** | 内置 Cloudflare Tunnel 配置，一个域名指向本机工作台 |

纯 CPU 即可运行（实测 RTF 2.4~2.9），有显卡会快一个数量级。

## 技术栈与规模

Java 17 负责 HTTP 服务、任务调度、引擎抽象与二进制协议；Python 负责推理桥接，把 GPT-SoVITS 的散装接口收敛成一套统一的音色注册表；前端零框架、零构建、零外部依赖，直接手写。

| 部分 | 文件 | 行数 | 说明 |
|---|---|---|---|
| Java | 30 | 4696 | HTTP/API、TCP 协议、任务队列、四种引擎实现 |
| Python | 5 | 1652 | 推理桥接、音色池、RVC 推理 |
| 前端 | 6 | 3753 | 三栏工作台 + 旧版单页 |
| **合计** | **52** | **~10100** | 含文档与脚本 |

依赖只有 jackson-databind、slf4j、logback 三个，没有 Spring，没有前端框架。

## 关于开发过程

这个项目是我（一个计算机专业学生）独立完成的需求定义、架构设计、接口划分和全部验收测试，**编码环节大量使用了 AI 辅助**——包括代码生成、报错排查和文档整理。

具体来说：GSV 的版本头识别规则（`00`=v1 … `08`=v5turbo）、底模不参与推理这个关键结论、双管线架构的取舍、9 条「踩过的坑」全部是我在实际调试中撞出来并验证的；AI 承担的是把结论写成代码，以及在我卡住时提供排查方向。**没有 AI 这个项目大概完成不了，但每一行能跑通的代码我都知道它为什么那样写。**

如果你在意「是否纯手写」这件事，这里如实说明，不做包装。

## 快速开始

### 1. 准备 GPT-SoVITS

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS && pip install -r requirements.txt
# 下载预训练底模（至少需要 chinese-hubert-base、chinese-roberta-wwm-ext-large）
python download.py
```

### 2. 准备模型文件

每个音色需要三样东西：

| 文件 | 说明 |
|---|---|
| `*.ckpt` | GPT 语义模型（Text2Semantic） |
| `*.pth` | SoVITS 声学模型 |
| `*.wav` | 参考音频，3~10 秒单人干净人声，附逐字文本 |

### 3. 配置

```bash
cp .env.example .env       # 填入你的 GSV_ROOT / GSV_MODELS
cp deploy/voices.example.json deploy/voices.local.json   # 按真实路径改写
```

### 4. 编译与启动

```bash
mvn clean package -DskipTests    # 产出 target/tts-batch.jar

start.bat                        # 一键拉起引擎 + 工作台
```

手动启动：

```bat
deploy\run_engine.bat            rem 推理引擎，端口 9880
run_ui.bat                       rem 工作台，端口 8090
```

打开 `http://127.0.0.1:8090`。

## 架构

```
浏览器
  │  HTTP (JSON / audio/wav)
  ▼
Java 工作台  HttpApiServer  :8090     ← 静态页面 + REST + 转发
  │  HTTP/1.1 (JSON)
  ▼
Python 桥接  sovits_bridge  :9880     ← 音色池、权重热切换、推理调度
  │
  ▼
GPT-SoVITS (CPU / CUDA)   ─┬─ GSV 通道：文本 → 音频
                            └─ RVC 通道：音频 → 音频（级联时接在 GSV 后面）
```

另有自研 TCP 协议服务（`serve` 子命令，端口 8089），演示二进制分帧、粘包处理与 CRC 校验，与 Web 通道互不影响。

**为什么不是一层？** GPT-SoVITS 的模型状态是进程级全局的——换音色必须改全局权重，加载耗时且不能并发。这段逻辑只能用 Python 写。但如果整个服务都在 Python 里，前端、任务队列、引擎抽象、协议编解码全都得挤进 FastAPI，那就成了一个大脚本站。

所以按**职责**切：Java 管「有多少任务、谁先来、用哪个引擎」，Python 管「加载哪个权重、怎么推理」。两者之间只有一条 JSON 契约，Py 那一侧可以独立替换成别的推理后端，Java 那侧不用改。

## 仓库结构

```
tts-batch/
├─ start.bat                      一键启动
├─ run_ui.bat                     工作台启动脚本
├─ .env.example                   环境变量模板
├─ deploy/
│  ├─ sovits_bridge.py            推理桥接（FastAPI）
│  ├─ run_engine.bat              引擎启动脚本
│  ├─ voices.example.json         音色池模板（含字段说明）
│  ├─ voices.local.json           你自己的音色池（已 gitignore）
│  ├─ rvc_core/                   RVC 推理实现
│  └─ assets/                     模型资产（已 gitignore，需自行下载）
├─ web/                           前端（无框架，零外部依赖）
│  ├─ studio.html / .css / .js    新版三栏工作台
│  └─ index.html / style.css / app.js   旧版单页
└─ src/main/java/...              Java 侧
   ├─ engine/                     各 TTS 引擎实现（Edge / Fish / Custom / Local）
   ├─ net/                        二进制协议编解码
   ├─ server/                     TCP 服务与任务队列
   └─ web/                        HTTP 接口与静态资源
```

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 引擎状态、设备、模型版本 |
| GET | `/api/voices` | 音色列表（来自 `voices.local.json`） |
| POST | `/api/voices/scan` | 扫描模型目录，自动识别 ckpt / pth 配对 |
| POST | `/api/voices/register` | 注册新音色 |
| POST | `/api/voices/delete` | 移除音色配置（不删权重文件） |
| GET | `/api/settings` | 读取工作台设置 |
| POST | `/api/settings` | 保存设置 |
| POST | `/api/warmup` | 预加载指定音色，切换权重 |
| POST | `/api/tts` | 合成，返回 `audio/wav`，头 `X-Cost-Ms` 为服务端耗时 |
| POST | `/api/translate` | 中 → 英 / 日 |
| POST | `/api/convert` | RVC 变声，请求体为 wav 二进制 |

```json
POST /api/tts
{
  "text": "要合成的文字",
  "voiceId": "voice-a",
  "speed": 1.0,
  "temperature": 1.0,
  "textLang": "zh",
  "emotion": "开心"
}
```

支持分享链接：`/?text=...&voice=voice-a&speed=0.92&autoplay=1`。

## 音色池配置

完整字段说明见 `deploy/voices.example.json` 的 `_fields` 段。最小可用配置：

```json
{
  "id": "voice-a",
  "name": "示例音色",
  "lang": "zh",
  "refLang": "zh",
  "gpt": "D:/models/voice-a/voice-a-e15.ckpt",
  "sovits": "D:/models/voice-a/voice-a_e8_s1248.pth",
  "ref": "D:/models/voice-a/ref/ref-a.wav",
  "refText": "该片段对应的文字"
}
```

`ref` 必须是 3~10 秒、单人、无背景音乐的片段；`refText` 必须与音频内容**逐字一致**，否则音色相似度和断句都会明显变差。

### 加情绪

`emotions` 的每个键都替换 `ref` / `refText` / `refDuration`：

```json
"emotions": {
  "开心": { "ref": "D:/models/voice-a/ref/happy.wav", "refText": "开心时的原话", "refDuration": 4.2 }
}
```

这是 GPT-SoVITS 唯一可靠的情绪控制方式——模型没有情感标签通道。

### RVC 级联

给一个已有的 GSV 音色叠 RVC 变声，用 `type: "rvc"` + `baseVoiceId` 指向基准音色。适合只有变声模型的角色。

## 实测性能

测试机：Ryzen 7 8845H，纯 CPU，16 线程。

| 模型版本 | 23 字 / 6s 音频 | RTF |
|---|---|---|
| v2ProPlus | 12.9 ~ 17.8s | 2.4 ~ 2.9 |
| v4 LoRA（需 vocoder） | 70.8 ~ 117.8s | 27.3 |

音色热切换约 1.7s。**没有 NVIDIA 显卡时这是正常速度**，CPU 上 GPT-SoVITS 的 RTF 约 2~3；v4 模型在 CPU 上基本不可用，建议配 GPU。

## 部署到公网（可选）

用 Cloudflare Tunnel 把本机 8090 映射到自己的域名，不用开端口、不需要公网 IP：

```bash
cloudflared tunnel login
cloudflared tunnel create tts
cloudflared tunnel route dns tts tts.example.com
cloudflared tunnel run tts
```

`~/.cloudflared/config.yml`：

```yaml
tunnel: <tunnel-id>
credentials-file: <path>/<tunnel-id>.json
ingress:
  - hostname: tts.example.com
    service: http://127.0.0.1:8090
  - service: http_status:404
```

注意三点：

1. **这个服务没有鉴权**，拿到链接谁都能用。对自己以外的人开放前，务必加 [Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/applications/) 或自建登录。
2. **免费版代理超时 100 秒**，v4 模型的长句会被切断。
3. 本机关机即断线，需要开机自启才能长期在线。

## 踩过的坑

| 现象 | 原因 | 处理 |
|---|---|---|
| 传了权重却说找不到文件 | `TTS_Config` 只读 `configs["custom"]`，缺键会静默回落到内置 v2 底模 | 配置必须套一层 `custom` |
| FastAPI 报 422 且 body 为 null | JDK 客户端默认带 `Upgrade: h2c`，uvicorn 不认明文升级，丢弃请求体 | 请求钉死 `HTTP_1_1` |
| `ModuleNotFoundError: jieba_fast` | 需要 MSVC 编译 | 用纯 Python `jieba` 写兼容层 |
| `require torch >= 2.6` | transformers 4.53+ 为 CVE-2025-32434 加的检查 | `.bin` 转 `safetensors` |
| 控制台中文乱码 | `chcp 65001` 与 Java 17 的 GBK 输出冲突 | 启动脚本改用 `chcp 936` |
| 本机服务连不上 | 系统代理劫持了 127.0.0.1 | 网关只连回环，不设代理 |
| v4 模型加载报「底模缺失」 | v4 权重是 LoRA 增量，需配 `s2Gv4.pth` + `vocoder.pth` | 下载到 `pretrained_models/gsv-v4-pretrained/` |
| faiss 读索引报 could not open | Windows 版 faiss 用窄字符打开路径 | 含中文的路径先复制到 ASCII 缓存目录 |
| 录音时长比实际长 | 用 `performance.now()` 计时含 AudioContext 启动开销 | 按样本数 / 采样率算 |

## 环境依赖

- JDK 17+、Maven 3.6+
- Python 3.10（建议 conda 独立环境），依赖见 GPT-SoVITS 的 `requirements.txt`
- GPT-SoVITS 源码与预训练底模：`chinese-hubert-base`、`chinese-roberta-wwm-ext-large`、`sv/pretrained_eres2netv2w24s4ep4.ckpt`
- 若用 RVC 通道，还需 `rmvpe.onnx`（音高提取）与 `hubert_base.pt`，放在 `deploy/assets/`

## 已知限制

- 纯 CPU 推理，长文本（>200 字）等待时间明显变长，建议单次不超过 100 字。
- 情感控制完全依赖参考音频本身的情绪，没有独立的 style 参数。
- 全局单锁串行推理：多个请求会排队，不适合多人共用。
- 浏览器自动播放策略可能拦截首次播放，点一下播放键即可。

## 模型与版权

**本仓库不包含任何模型权重、参考音频或合成结果。** 你需要自己准备合法的音色素材。

需要注意：

- GPT-SoVITS 本体遵循 MIT 协议。
- 社区训练的第三方音色模型，多数标注为 **CC-BY-NC**（非商用）或类似协议，且角色形象版权通常归原作品方所有。用于商业用途前请自行确认授权。
- 模拟真人声音时，需取得本人同意；不得用于伪造他人言论。
- 本项目仅作为技术工具，使用者需自行承担素材来源合规责任。

## License

代码部分采用 MIT。第三方模型与素材的授权见各自来源。
