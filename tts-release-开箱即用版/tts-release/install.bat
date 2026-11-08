@echo off
chcp 936 >nul
title Wenyan Audio - One-click Install
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo   文音 Audio  一键安装
echo   会做四件事：下载 GPT-SoVITS 源码 / 下载底模 /
echo               建立 Python 环境 / 生成本地配置
echo   首次安装约 10-30 分钟，取决于网速
echo ============================================================
echo.

rem ---------- 0. 前置检查 ----------
where git >nul 2>nul
if errorlevel 1 (
  echo [ERROR] 未找到 git，请先安装 Git for Windows：
  echo         https://git-scm.com/download/win
  pause & exit /b 1
)

rem Python：优先环境变量 PY，其次 PATH 里的 python
if not defined PY set "PY=python"
"%PY%" -c "import sys; sys.exit(0 if sys.version_info[:2]>=(3,9) else 1)" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] 未找到可用的 Python 3.9+。
  echo         请安装 Python 3.10 并勾选 "Add to PATH"，或设置环境变量 PY 指向解释器。
  echo         建议用 Miniconda：https://docs.conda.io/en/latest/miniconda.html
  pause & exit /b 1
)
for /f "delims=" %%v in ('"%PY%" -c "import sys;print(sys.version.split()[0])"') do set "PYVER=%%v"
echo [OK] Python %PYVER%
echo [OK] git 已就绪

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "GSV_ROOT=%ROOT%\gsv-src"
set "GSV_MODELS=%ROOT%\models"

rem ---------- 1. GPT-SoVITS 源码 ----------
echo.
echo [1/4] 检查 GPT-SoVITS 源码 ...
if exist "%GSV_ROOT%\GPT_SoVITS\inference_webui.py" (
  echo       已存在，跳过
) else (
  echo       正在克隆（约 200MB，可能需要几分钟）...
  git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git "%GSV_ROOT%"
  if errorlevel 1 (
    echo [ERROR] 克隆失败。常见原因：
    echo         1) 网络无法访问 GitHub，试试先开代理
    echo         2) 目标目录已存在但不是完整仓库
    echo         可手动下载后解压到：%GSV_ROOT%
    pause & exit /b 1
  )
)

rem ---------- 2. 预训练底模 ----------
echo.
echo [2/4] 下载预训练底模（约 1.7GB）...
set "PM=%GSV_ROOT%\GPT_SoVITS\pretrained_models"

call :fetch "chinese-hubert-base/pytorch_model.bin" "%PM%\chinese-hubert-base\pytorch_model.bin"
call :fetch "chinese-hubert-base/config.json"     "%PM%\chinese-hubert-base\config.json"
call :fetch "chinese-hubert-base/preprocessor_config.json" "%PM%\chinese-hubert-base\preprocessor_config.json"
call :fetch "chinese-roberta-wwm-ext-large/pytorch_model.bin" "%PM%\chinese-roberta-wwm-ext-large\pytorch_model.bin"
call :fetch "chinese-roberta-wwm-ext-large/config.json"       "%PM%\chinese-roberta-wwm-ext-large\config.json"
call :fetch "chinese-roberta-wwm-ext-large/tokenizer.json"    "%PM%\chinese-roberta-wwm-ext-large\tokenizer.json"
call :fetch "s2G488k.pth" "%PM%\s2G488k.pth"
call :fetch "s2D488k.pth" "%PM%\s2D488k.pth"

rem ---------- 3. Python 依赖 ----------
echo.
echo [3/4] 安装 Python 依赖 ...
"%PY%" -c "import torch, fastapi, librosa, soundfile, transformers" >nul 2>nul
if errorlevel 1 (
  echo       未检测到完整依赖，开始安装（约 2GB，耐心等）...
  echo       使用清华镜像加速
  "%PY%" -m pip install -r "%GSV_ROOT%\requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
  if errorlevel 1 (
    echo.
    echo [WARN] requirements.txt 安装失败，改用精简依赖集重试 ...
    "%PY%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
    "%PY%" -m pip install fastapi uvicorn librosa soundfile numpy transformers ^
        jieba cn2an g2p_en LangSegment wordsegment nltk pyyaml scipy numba ^
        -i https://pypi.tuna.tsinghua.edu.cn/simple
  )
) else (
  echo       依赖已满足，跳过
)

rem ---------- 4. 生成本地配置 ----------
echo.
echo [4/4] 生成本地配置 ...
"%PY%" "%ROOT%\deploy\gen_config.py"
if errorlevel 2 (
  echo.
  echo [WARN] 配置已生成，但有模型文件缺失，详见上面的提示
)
if not exist "%ROOT%\deploy\voices.local.json" (
  echo [ERROR] 配置生成失败，请检查 deploy\voices.runtime.json 是否存在
  pause & exit /b 1
)

echo.
echo ============================================================
echo   安装完成
echo.
echo   现在可以双击 start.bat 启动
echo   首次启动加载模型需要 20-60 秒
echo ============================================================
echo.
pause
exit /b 0

rem ========== 子程序：下载单个文件，支持断点续传 ==========
:fetch
set "REL=%~1"
set "DST=%~2"
if exist "%DST%" (
  for %%A in ("%DST%") do if %%~zA GTR 1024 (
    echo       [skip] %REL%
    exit /b 0
  )
)
for %%D in ("%DST%") do if not exist "%%~dpD" mkdir "%%~dpD"
echo       [get ] %REL%
curl -L --fail --retry 3 --retry-delay 2 --speed-limit 10240 --speed-time 30 ^
     -C - -o "%DST%" ^
     "https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/main/%REL%"
if errorlevel 1 (
  echo       [WARN] 下载失败：%REL%
  echo              可从 https://huggingface.co/lj1995/GPT-SoVITS 手动下载，
  echo              放到：%DST%
)
exit /b 0
