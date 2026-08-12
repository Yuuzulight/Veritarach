<#
.SYNOPSIS
  Sets up a uv-managed Python environment with the Veritarach stack.

  Checks the GPU FIRST and refuses to guess the PyTorch install command --
  the CUDA build and the CPU build are different commands, and installing the
  wrong one silently gives you CPU-only.
#>

[CmdletBinding()]
param(
  [string]$EnvPath = "C:\Users\Yuuzu\Desktop\Github Projects\Veritarach\.venv",
  [string]$ProjectPath = "C:\Users\Yuuzu\Desktop\Github Projects\Veritarach"
)

$ErrorActionPreference = "Stop"

Write-Host "==> Checking GPU..." -ForegroundColor Cyan
$gpuInfo = $null
try {
  $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv 2>$null
} catch {}

if (-not $gpuInfo) {
  Write-Warning "nvidia-smi not found or no NVIDIA GPU detected."
  Write-Warning "DeBERTa fine-tuning will run on CPU -- slow enough to change what's realistic in a week."
  Write-Warning "Get the CPU-only PyTorch command from https://pytorch.org/get-started/locally/ before continuing."
} else {
  Write-Host $gpuInfo
  Write-Host "==> GPU found. Get the matching CUDA build command from https://pytorch.org/get-started/locally/" -ForegroundColor Green
  Write-Host "    (pip's default `pip install torch` gives CPU-only -- don't just run that)." -ForegroundColor Green
}

Write-Host "`n==> This script will NOT install torch for you -- copy the exact command pytorch.org gives you," -ForegroundColor Yellow
Write-Host "    based on the GPU result above, and run it inside the venv this creates." -ForegroundColor Yellow

Push-Location $ProjectPath
try {
  Write-Host "==> Syncing project dependencies with uv at $EnvPath..." -ForegroundColor Cyan
  uv sync

  Write-Host "`n==> Now install torch by hand:" -ForegroundColor Green
  Write-Host "    & `"$EnvPath\Scripts\python.exe`" -m pip install <command from pytorch.org>"
}
finally {
  Pop-Location
}

Write-Host "`n==> Still missing, not an install problem -- from Discord:" -ForegroundColor Yellow
Write-Host "    - The node secret and testnet MACHINA source"
Write-Host "    - Confirmation of the curated Intent list (AI_DETECTION vs. TEXT_AUTHENTICITY_CHECK/CONTENT_VERIFICATION fallback)"
