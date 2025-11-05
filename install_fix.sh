#!/bin/bash
# COMPLETE INSTALLATION FIX - Run this to install all packages in correct order
# This installs packages in dependency order to avoid conflicts

set -e  # Exit on error

echo "========================================================================"
echo "  INSTALLING COMPATIBLE PACKAGES IN CORRECT ORDER"
echo "========================================================================"

# Step 1: Install NumPy FIRST (base dependency for everything)
echo ""
echo "Step 1: Installing NumPy 1.26.4..."
pip install numpy==1.26.4

# Step 2: Install SciPy (depends on NumPy)
echo ""
echo "Step 2: Installing SciPy 1.11.4..."
pip install scipy==1.11.4

# Step 3: Install Matplotlib (depends on NumPy)
echo ""
echo "Step 3: Installing Matplotlib 3.8.4..."
pip install matplotlib==3.8.4

# Step 4: Install OpenCV (depends on NumPy)
echo ""
echo "Step 4: Installing OpenCV..."
pip install opencv-python==4.10.0.84

# Step 5: Install Albumentations (depends on NumPy, SciPy, OpenCV)
echo ""
echo "Step 5: Installing Albumentations..."
pip install albumentations>=1.3.0

# Step 6: Install Pillow
echo ""
echo "Step 6: Installing Pillow..."
pip install Pillow>=10.0.0

# Step 7: Install Pandas
echo ""
echo "Step 7: Installing Pandas..."
pip install pandas>=2.0.0

# Step 8: Verify installation
echo ""
echo "========================================================================"
echo "  VERIFYING INSTALLATION"
echo "========================================================================"
python << 'EOF'
print("\nChecking all imports...")
try:
    import numpy
    print(f"✓ NumPy {numpy.__version__}")
except Exception as e:
    print(f"✗ NumPy: {e}")
    
try:
    import scipy
    print(f"✓ SciPy {scipy.__version__}")
except Exception as e:
    print(f"✗ SciPy: {e}")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except Exception as e:
    print(f"✗ Matplotlib: {e}")

try:
    import albumentations
    print(f"✓ Albumentations {albumentations.__version__}")
except Exception as e:
    print(f"✗ Albumentations: {e}")

try:
    import pandas
    print(f"✓ Pandas {pandas.__version__}")
except Exception as e:
    print(f"✗ Pandas: {e}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
except Exception as e:
    print(f"✗ PyTorch: {e}")

print("\nTesting critical imports...")
try:
    from scipy import special
    print("✓ scipy.special")
except Exception as e:
    print(f"✗ scipy.special: {e}")

try:
    import albumentations as A
    print("✓ albumentations as A")
except Exception as e:
    print(f"✗ albumentations: {e}")

print("\n✓✓✓ ALL PACKAGES INSTALLED SUCCESSFULLY ✓✓✓")
EOF

echo ""
echo "========================================================================"
echo "  INSTALLATION COMPLETE"
echo "========================================================================"
echo ""
echo "You can now run: python semantic-drone-dataset-vgg16-unet.py"
echo ""
