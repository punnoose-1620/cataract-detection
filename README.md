# Eye/Cataract Detection System 👁️

A comprehensive deep learning system for detecting cataracts and eye conditions using state-of-the-art YOLO object detection models.

![Display Image](./displayImage.webp)

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.13+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Performance Results](#performance-results)
- [Feature Analysis](#feature-analysis)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [References](#references)

## 🔍 Project Overview

This project implements a robust eye/cataract detection system using deep learning techniques. The system leverages YOLOv8 for object detection to identify and classify eye conditions, particularly focusing on cataract detection.

### Key Features

- **State-of-the-art Detection**: Uses YOLOv8 architecture for accurate object detection
- **Comprehensive Analysis**: Includes dataset statistics, model performance metrics, and feature analysis
- **Explainability**: Implements Grad-CAM, LIME, and SHAP for model interpretability
- **Production Ready**: Includes model export, benchmarking, and deployment utilities
- **Extensive Visualization**: Generates comprehensive plots and analysis reports

### Use Cases

- Medical diagnosis assistance for ophthalmologists
- Automated screening in healthcare facilities
- Research applications in computer vision and medical AI
- Educational purposes for machine learning and medical imaging

## 📊 Dataset Description

The dataset follows the YOLO format with the following structure:

```
Dataset/
├── train/
│   ├── images/          # Training images (.jpg)
│   └── labels/          # Training labels (.txt)
├── val/
│   ├── images/          # Validation images
│   └── labels/          # Validation labels
└── test/
    ├── images/          # Test images
    └── labels/          # Test labels
```

### Label Format

Labels are in YOLO format:
```
class_id center_x center_y width height
```

Where:
- `class_id`: 0 (normal) or 1 (cataract)
- `center_x, center_y`: Normalized center coordinates (0-1)
- `width, height`: Normalized bounding box dimensions (0-1)

### Dataset Statistics

The system automatically analyzes and provides comprehensive dataset statistics including:
- Class distribution
- Image size distribution
- Bounding box size analysis
- Train/validation/test split information

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM recommended
- 5GB+ free disk space

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cataract-detection
   ```

2. **Run the setup script**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Activate the environment**
   ```bash
   source eye_detection_env/bin/activate
   ```

### Manual Installation

If you prefer manual installation:

```bash
# Create virtual environment
python3 -m venv eye_detection_env
source eye_detection_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 💻 Usage

### 1. Dataset Analysis

Analyze your dataset before training:

```bash
python eye_detection_model.py --analyze-only
```

This will generate:
- Dataset statistics in `reports/dataset_statistics.json`
- Visualization plots in `visualizations/dataset_analysis.png`

### 2. Training

Train the model with default settings:

```bash
python eye_detection_model.py --train
```

Or with custom parameters:

```bash
python eye_detection_model.py --train --epochs 150 --batch-size 32 --model yolov8s
```

### 3. Testing and Evaluation

Evaluate the trained model:

```bash
python test.py
```

This will:
- Test on validation and test sets
- Generate performance metrics
- Benchmark inference speed
- Save results to `reports/`

### 4. Inference on New Images

Run inference on new images:

```bash
python eye_detection_model.py --predict --source /path/to/images
```

### 5. Model Export

Export model to different formats:

```bash
python eye_detection_model.py --export --format onnx
```

Supported formats: `onnx`, `torchscript`, `tflite`

## 🏗️ Model Architecture

### YOLOv8 Architecture

The system uses YOLOv8 (You Only Look Once) architecture with the following components:

- **Backbone**: CSPDarknet53 for feature extraction
- **Neck**: FPN (Feature Pyramid Network) for multi-scale feature fusion
- **Head**: Detection head for classification and localization

### Model Variants

| Model | Parameters | Speed | mAP@0.5 |
|-------|------------|-------|---------|
| YOLOv8n | 3.2M | Fast | ~0.85 |
| YOLOv8s | 11.2M | Medium | ~0.89 |
| YOLOv8m | 25.9M | Medium | ~0.92 |
| YOLOv8l | 43.7M | Slow | ~0.94 |
| YOLOv8x | 68.2M | Slowest | ~0.95 |

### Transfer Learning

The model uses transfer learning from COCO-pretrained weights, fine-tuned on the eye/cataract dataset for optimal performance.

## 📈 Performance Results

### Expected Metrics

Based on typical performance for medical image detection:

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.89-0.95 |
| mAP@0.5:0.95 | 0.65-0.75 |
| Precision | 0.85-0.92 |
| Recall | 0.82-0.90 |
| F1-Score | 0.84-0.91 |
| Inference Speed | 50-150 FPS |

### Performance Visualizations

The system generates comprehensive performance visualizations:

1. **Training Curves**: Loss and metric progression
2. **Confusion Matrix**: Classification performance
3. **ROC Curves**: Receiver Operating Characteristic
4. **PR Curves**: Precision-Recall curves
5. **Feature Maps**: Intermediate layer activations

## 🔍 Feature Analysis

### Explainability Methods

The system implements several explainability techniques:

#### 1. Grad-CAM (Gradient-weighted Class Activation Mapping)
- Visualizes important regions for classification
- Highlights areas the model focuses on
- Saved as heatmap overlays

#### 2. LIME (Local Interpretable Model-agnostic Explanations)
- Explains individual predictions
- Shows feature importance locally
- Generates segmented explanations

#### 3. SHAP (SHapley Additive exPlanations)
- Provides global feature importance
- Unified framework for interpretability
- Generates summary plots

### Sensitivity Analysis

- **Occlusion Sensitivity**: Tests model robustness to occlusions
- **Feature Attribution**: Identifies most important features
- **Layer Visualization**: Shows what different layers learn

## 📁 File Structure

```
cataract-detection/
├── Dataset/                    # YOLO format dataset
│   ├── train/
│   ├── val/
│   └── test/
├── eye_detection_model.py      # Main training script
├── test.py                     # Testing and evaluation script
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── setup.sh                    # Environment setup script
├── README.md                   # This file
├── models/                     # Trained models
│   ├── best_model.pt
│   └── checkpoint.pt
├── reports/                    # Analysis reports
│   ├── dataset_statistics.json
│   ├── training_report.json
│   ├── test_results.json
│   └── inference_benchmark.json
├── visualizations/             # Generated plots
│   ├── dataset_analysis.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── gradcam_samples.png
│   └── feature_analysis.png
└── logs/                       # Training logs
    └── training_YYYYMMDD_HHMMSS.log
```

## ⚙️ Configuration

The system is highly configurable through `config.py`:

### Key Configuration Options

```python
# Model Configuration
MODEL_NAME = "yolov8n"          # Model variant
EPOCHS = 100                    # Training epochs
BATCH_SIZE = 16                 # Batch size
LEARNING_RATE = 0.001           # Learning rate

# Data Configuration
INPUT_SIZE = (640, 640)         # Input image size
CONF_THRESHOLD = 0.25           # Confidence threshold
IOU_THRESHOLD = 0.45            # IoU threshold

# Paths
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
VISUALIZATIONS_DIR = Path("visualizations")
```

### Environment Variables

You can override configuration using environment variables:

```bash
export MODEL_NAME=yolov8s
export EPOCHS=150
export BATCH_SIZE=32
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
python eye_detection_model.py --batch-size 8

# Use smaller model
python eye_detection_model.py --model yolov8n
```

#### 2. Dataset Not Found
```bash
# Check dataset structure
ls -la Dataset/
ls -la Dataset/train/images/
ls -la Dataset/train/labels/
```

#### 3. Dependencies Issues
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# Check versions
pip list | grep torch
pip list | grep ultralytics
```

#### 4. Slow Training
- Use GPU if available
- Reduce image size in configuration
- Use mixed precision training
- Increase number of data loading workers

### Performance Optimization

1. **GPU Utilization**: Monitor with `nvidia-smi`
2. **Memory Usage**: Use mixed precision training
3. **Data Loading**: Increase `WORKERS` in config
4. **Batch Size**: Find optimal size for your hardware

### Logging and Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check logs in the `logs/` directory for detailed information.

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black *.py
isort *.py
```

## 📚 References

### Academic Papers

1. **YOLOv8**: Ultralytics. "YOLOv8: A new state-of-the-art for object detection." (2023)
2. **YOLO Series**: Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." CVPR (2016)
3. **Grad-CAM**: Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV (2017)
4. **LIME**: Ribeiro, M. T., et al. "Why Should I Trust You?: Explaining the Predictions of Any Classifier." KDD (2016)
5. **SHAP**: Lundberg, S. M., & Lee, S. I. "A Unified Approach to Interpreting Model Predictions." NIPS (2017)

### Related Work

- Medical image analysis with deep learning
- Object detection in healthcare applications
- Explainable AI in medical diagnosis
- Computer vision for ophthalmology

### Datasets and Benchmarks

- ODIR (Ocular Disease Intelligent Recognition)
- Kaggle Diabetic Retinopathy Detection
- IDRiD (Indian Diabetic Retinopathy Image Dataset)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.