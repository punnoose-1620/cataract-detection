#!/usr/bin/env python3
"""
Eye/Cataract Detection System Demo
=================================

Quick demo script to showcase the system capabilities.

Author: AI Assistant  
Date: 2024
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import config

def main():
    """Run system demo"""
    print("🔍 EYE/CATARACT DETECTION SYSTEM DEMO")
    print("=" * 50)
    
    print("\n📋 System Overview:")
    print("- State-of-the-art YOLO-based eye detection")
    print("- Comprehensive dataset analysis")
    print("- Model training and evaluation")
    print("- Feature visualization and explainability")
    print("- Production-ready deployment")
    
    print(f"\n📊 Dataset Information:")
    print(f"- Training images: {len(list(config.TRAIN_IMAGES_PATH.glob('*.jpg')))}")
    print(f"- Validation images: {len(list(config.VAL_IMAGES_PATH.glob('*.jpg')))}")
    print(f"- Test images: {len(list(config.TEST_IMAGES_PATH.glob('*.jpg')))}")
    print(f"- Classes: {config.CLASS_NAMES}")
    
    print(f"\n🏗️ Model Configuration:")
    print(f"- Architecture: {config.MODEL_NAME}")
    print(f"- Input size: {config.INPUT_SIZE}")
    print(f"- Batch size: {config.BATCH_SIZE}")
    print(f"- Epochs: {config.EPOCHS}")
    
    print(f"\n📁 Output Directories:")
    print(f"- Models: {config.MODELS_DIR}")
    print(f"- Reports: {config.REPORTS_DIR}")
    print(f"- Visualizations: {config.VISUALIZATIONS_DIR}")
    print(f"- Logs: {config.LOGS_DIR}")
    
    print("\n🚀 Quick Start Commands:")
    print("1. Setup environment:")
    print("   chmod +x setup.sh && ./setup.sh")
    
    print("\n2. Analyze dataset:")
    print("   python eye_detection_model.py --analyze-only")
    
    print("\n3. Train model:")
    print("   python eye_detection_model.py --train")
    
    print("\n4. Test model:")
    print("   python test.py")
    
    print("\n5. Run inference:")
    print("   python eye_detection_model.py --predict --source Dataset/test/images")
    
    print("\n6. Export model:")
    print("   python eye_detection_model.py --export --format onnx")
    
    print("\n" + "=" * 50)
    print("🎯 Ready to detect eye conditions with AI!")
    print("📚 Check README.md for detailed documentation")

if __name__ == "__main__":
    main() 