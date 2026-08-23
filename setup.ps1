# setup.ps1
# This script sets up a Python virtual environment and installs StreamDiffusion with TensorRT.

$ErrorActionPreference = "Stop"

Write-Host "Setting up StreamDiffusion Environment..." -ForegroundColor Green

# 1. Create and activate virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment 'venv'..." -ForegroundColor Cyan
    python -m venv venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
$VenvScripts = Join-Path $PWD "venv\Scripts\Activate.ps1"
. $VenvScripts

# 2. Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# 3. Install other base requirements
Write-Host "Installing OpenCV..." -ForegroundColor Cyan
pip install "opencv-python==4.9.0.80"

# 4. Install PyTorch with CUDA 12.1 (recommended for recent TensorRT) - doing this after xformers to prevent downgrade
Write-Host "Installing PyTorch..." -ForegroundColor Cyan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --upgrade

# 4.5. Install compatible huggingface_hub and transformers for diffusers
Write-Host "Installing compatible huggingface_hub..." -ForegroundColor Cyan
pip install huggingface_hub==0.25.2 transformers==4.35.2 diffusers==0.24.0

# 5. Clone StreamDiffusion if it doesn't exist
if (-not (Test-Path "StreamDiffusion")) {
    Write-Host "Cloning StreamDiffusion repository..." -ForegroundColor Cyan
    git clone https://github.com/cumulo-autumn/StreamDiffusion.git
}

# 6. Install StreamDiffusion with TensorRT support
Write-Host "Installing StreamDiffusion with TensorRT support..." -ForegroundColor Cyan
cd StreamDiffusion
python -m pip install -e .[tensorrt]
cd ..

# 7. Install TensorRT extensions manually to avoid broken dependencies
Write-Host "Installing TensorRT extensions..." -ForegroundColor Cyan
pip install nvidia-cudnn-cu12==8.9.4.25
pip install --pre --extra-index-url https://pypi.nvidia.com tensorrt==9.0.1.post12.dev4
pip install "numpy<2" "cuda-python>=12.0,<13.0" polygraphy onnx-graphsurgeon

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "To run the script, make sure to activate the environment: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
