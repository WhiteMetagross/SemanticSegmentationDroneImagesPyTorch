#!/bin/bash
# Conda Environment Setup Script for Linux
# Semantic Segmentation Project - VGG16-UNet

set -e  # Exit on error

echo "========================================"
echo "Semantic Segmentation Environment Setup"
echo "========================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda not found. Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Environment name
ENV_NAME="segmentation"

echo "Creating conda environment: $ENV_NAME"
echo ""

# Create conda environment with Python 3.11
conda create -n $ENV_NAME python=3.11 -y

echo ""
echo "Activating environment..."
echo ""

# Activate environment (for current shell)
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo "Verifying installation..."
echo "========================================"
echo ""

# Verify installation
python << 'EOF'
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
EOF

echo ""
echo "========================================"
echo "Setup Complete"
echo "========================================"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run the training script:"
echo "  python semantic-drone-dataset-vgg16-unet.py"
echo ""
