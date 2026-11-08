@echo off
rem 与 Java 侧保持一致用 GBK 代码页，避免中文日志乱码
chcp 936 >nul
title Wenyan Audio - Engine
setlocal enabledelayedexpansion
cd /d "%~dp0.."

set "ROOT=%~dp0.."

rem ---- GPT-SoVITS 源码根目录 ----
if not defined GSV_ROOT set "GSV_ROOT=%ROOT%\gsv-src"
if not exist "%GSV_ROOT%\GPT_SoVITS" (
  echo ============================================================
  echo  [ERROR] 找不到 GPT-SoVITS 源码: %GSV_ROOT%
  echo.
  echo  请先双击 install.bat 完成安装。
  echo ============================================================
  pause
  exit /b 1
)

rem ---- Python ----
if not defined PY (
  if exist "%ROOT%\.venv\Scripts\python.exe" set "PY=%ROOT%\.venv\Scripts\python.exe"
)
if not defined PY set "PY=python"
"%PY%" -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] 找不到 Python，请先运行 install.bat，或设置环境变量 PY
  pause
  exit /b 1
)

rem ---- 音色配置 ----
set "VOICES=%~dp0voices.local.json"
if not exist "%VOICES%" (
  echo [WARN] 缺少 voices.local.json，尝试自动生成 ...
  "%PY%" "%~dp0gen_config.py"
  if not exist "%VOICES%" (
    echo [ERROR] 生成失败，请运行 install.bat
    pause
    exit /b 1
  )
)

echo ============================================
echo   GPT-SoVITS Bridge
echo   python : %PY%
echo   gsv    : %GSV_ROOT%
echo   voices : %VOICES%
echo   首次加载模型需要 20-60 秒，看到 "bridge started" 即就绪
echo ============================================
echo.

"%PY%" "%~dp0sovits_bridge.py" --port 9880 --voices "%VOICES%" --gsv-root "%GSV_ROOT%"

echo.
echo engine exited with code %errorlevel%
pause
