@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
title ScholarAgent 本地工作台

rem One-click launcher: ensure venv, start local workspace, open browser.
cd /d "%~dp0"
if errorlevel 1 (
  echo [错误] 无法进入项目目录。
  pause
  exit /b 1
)

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "REQ=%CD%\requirements.txt"
set "PORT=8765"

if not exist "%REQ%" (
  echo [错误] 找不到 requirements.txt，请确认脚本位于项目根目录。
  pause
  exit /b 1
)

if not exist "%VENV_PY%" (
  echo [准备] 未找到 .venv，正在创建虚拟环境...
  where python >nul 2>&1
  if errorlevel 1 (
    echo [错误] 系统里找不到 python。请先安装 Python 3.10+ 并勾选 Add to PATH。
    pause
    exit /b 1
  )
  python -m venv .venv
  if errorlevel 1 (
    echo [错误] 创建虚拟环境失败。
    pause
    exit /b 1
  )
  set "NEED_INSTALL=1"
) else (
  rem 虚拟环境在,但核心依赖可能缺失(上次安装中断)
  "%VENV_PY%" -c "import openai,dotenv,httpx,pypdf" >nul 2>&1
  if errorlevel 1 set "NEED_INSTALL=1"
)

if defined NEED_INSTALL (
  echo [准备] 安装依赖 requirements.txt ...
  "%VENV_PY%" -m pip install -U pip
  if errorlevel 1 (
    echo [错误] 升级 pip 失败。
    pause
    exit /b 1
  )
  "%VENV_PY%" -m pip install -r "%REQ%"
  if errorlevel 1 (
    echo [错误] 依赖安装失败。请检查网络后重试。
    pause
    exit /b 1
  )
)

if not exist "%CD%\.env" (
  if exist "%CD%\.env.example" (
    echo [提示] 未找到 .env，已从 .env.example 复制一份。
    echo         云端模型请编辑 .env 填入 LLM_API_KEY；本机有 Ollama 时可跳过。
    copy /Y "%CD%\.env.example" "%CD%\.env" >nul
  ) else (
    echo [提示] 未找到 .env。可稍后自行配置 API Key，或使用本机 Ollama。
  )
)

echo.
echo ========================================
echo   ScholarAgent 本地工作台
echo   地址: http://127.0.0.1:%PORT%
echo   按 Ctrl+C 可停止服务
echo ========================================
echo.

"%VENV_PY%" webapp.py --port %PORT% --open
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo.
  echo [错误] 工作台异常退出，代码 %EC%。
  pause
)
endlocal & exit /b %EC%
