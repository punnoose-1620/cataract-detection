#!/bin/bash

# Eye/Cataract Detection System Setup Script
echo "=============================================="
echo "Setting up Eye/Cataract Detection Environment"
echo "=============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Found Python version: $python_version"

# Create virtual environment
ENV_NAME="eye_detection_env"
echo "🔧 Creating virtual environment: $ENV_NAME"

if [ -d "$ENV_NAME" ]; then
    echo "⚠️  Virtual environment already exists. Removing old environment..."
    rm -rf "$ENV_NAME"
fi

python3 -m venv "$ENV_NAME"
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source "$ENV_NAME/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip to latest version..."
pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to upgrade pip"
    exit 1
fi

# Install requirements
echo "📦 Installing required packages from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to install requirements"
        exit 1
    fi
else
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"
python3 -c "import torchvision; print(f'✅ TorchVision: {torchvision.__version__}')"
python3 -c "import cv2; print(f'✅ OpenCV: {cv2.__version__}')"
python3 -c "import ultralytics; print(f'✅ Ultralytics: {ultralytics.__version__}')"

# Check for CUDA availability
python3 -c "import torch; print(f'✅ CUDA Available: {torch.cuda.is_available()}')"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python3 -c "import torch; print(f'✅ CUDA Device: {torch.cuda.get_device_name(0)}')"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To activate the environment in the future, run:"
echo "    source $ENV_NAME/bin/activate"
echo ""
echo "To run the main model script:"
echo "    python eye_detection_model.py"
echo ""
echo "To run tests:"
echo "    python test.py"
echo ""
echo "==============================================" 