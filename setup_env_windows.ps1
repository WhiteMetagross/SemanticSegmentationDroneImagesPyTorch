# Conda Environment Setup Script for Windows 11
# Semantic Segmentation Project - VGG16-UNet

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Semantic Segmentation Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if conda is available
if (!(Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Conda not found. Please install Miniconda or Anaconda first." -ForegroundColor Red
    Write-Host "Download from: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Yellow
    exit 1
}

# Environment name
$ENV_NAME = "segmentation"

Write-Host "Creating conda environment: $ENV_NAME" -ForegroundColor Green
Write-Host ""

# Create conda environment with Python 3.11
conda create -n $ENV_NAME python=3.11 -y

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to create conda environment." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Activating environment..." -ForegroundColor Green
Write-Host ""

# Activate environment
conda activate $ENV_NAME

Write-Host "Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Green
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install PyTorch." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Installing project dependencies..." -ForegroundColor Green
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install dependencies." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Verifying installation..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verify installation
python -c @"
import sys
print(f'Python: {sys.version}')
print()

try:
    import numpy
    print(f'NumPy: {numpy.__version__}')
except Exception as e:
    print(f'NumPy: FAILED - {e}')

try:
    import scipy
    print(f'SciPy: {scipy.__version__}')
except Exception as e:
    print(f'SciPy: FAILED - {e}')

try:
    import cv2
    print(f'OpenCV: {cv2.__version__}')
except Exception as e:
    print(f'OpenCV: FAILED - {e}')

try:
    import matplotlib
    print(f'Matplotlib: {matplotlib.__version__}')
except Exception as e:
    print(f'Matplotlib: FAILED - {e}')

try:
    import albumentations
    print(f'Albumentations: {albumentations.__version__}')
except Exception as e:
    print(f'Albumentations: FAILED - {e}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA Device: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'PyTorch: FAILED - {e}')
"@

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Setup Complete" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment, run:" -ForegroundColor Yellow
Write-Host "  conda activate $ENV_NAME" -ForegroundColor White
Write-Host ""
Write-Host "To run the training script:" -ForegroundColor Yellow
Write-Host "  python semantic-drone-dataset-vgg16-unet.py" -ForegroundColor White
Write-Host ""
