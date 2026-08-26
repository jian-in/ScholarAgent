# ScholarAgent one-click launcher (PowerShell)
# Double-click start.bat if script policy blocks this file.
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$Req = Join-Path $Root "requirements.txt"
$Port = 8765

function Write-Step([string]$Message) {
    Write-Host $Message
}

if (-not (Test-Path $Req)) {
    throw "找不到 requirements.txt，请确认脚本位于项目根目录。"
}

$needInstall = $false
if (-not (Test-Path $VenvPy)) {
    Write-Step "[准备] 未找到 .venv，正在创建虚拟环境..."
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) {
        throw "系统里找不到 python。请先安装 Python 3.10+ 并勾选 Add to PATH。"
    }
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "创建虚拟环境失败。" }
    $needInstall = $true
} else {
    & $VenvPy -c "import openai,dotenv,httpx,pypdf" 2>$null
    if ($LASTEXITCODE -ne 0) { $needInstall = $true }
}

if ($needInstall) {
    Write-Step "[准备] 安装依赖 requirements.txt ..."
    & $VenvPy -m pip install -U pip
    if ($LASTEXITCODE -ne 0) { throw "升级 pip 失败。" }
    & $VenvPy -m pip install -r $Req
    if ($LASTEXITCODE -ne 0) { throw "依赖安装失败。请检查网络后重试。" }
}

$EnvFile = Join-Path $Root ".env"
$EnvExample = Join-Path $Root ".env.example"
if (-not (Test-Path $EnvFile) -and (Test-Path $EnvExample)) {
    Write-Step "[提示] 未找到 .env，已从 .env.example 复制一份。"
    Write-Step "        云端模型请编辑 .env 填入 LLM_API_KEY；本机有 Ollama 时可跳过。"
    Copy-Item $EnvExample $EnvFile
}

Write-Host ""
Write-Host "========================================"
Write-Host "  ScholarAgent 本地工作台"
Write-Host "  地址: http://127.0.0.1:$Port"
Write-Host "  按 Ctrl+C 可停止服务"
Write-Host "========================================"
Write-Host ""

& $VenvPy (Join-Path $Root "webapp.py") --port $Port --open
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host ""
    Write-Host "[错误] 工作台异常退出，代码 $exitCode。"
    exit $exitCode
}
