$ErrorActionPreference = "Stop"
$VenvScripts = Join-Path $PWD "venv\Scripts\Activate.ps1"
. $VenvScripts

Write-Host "Re-installing torch cu121 to override any incompatible versions..." -ForegroundColor Cyan
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --upgrade

Write-Host "Installing streamdiffusion[tensorrt]..." -ForegroundColor Cyan
cd StreamDiffusion
python -m pip install -e .[tensorrt]
cd ..

Write-Host "Installing TensorRT extensions..." -ForegroundColor Cyan
pip install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post12.dev4
