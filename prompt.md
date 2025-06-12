Create a comprehensive eye/cataract detection system using the YOLO dataset located in `Dataset/` folder. Generate the following files:

### **1. Main Python File: `eye_detection_model.py`**

Create a single, well-documented Python file with proper markdown comments for each section that includes:

**Data Loading & Processing:**
- Load YOLO format dataset from `Dataset/train`, `Dataset/val`, and `Dataset/test` directories
- Parse image files (JPG) and corresponding label files (TXT with YOLO format: class_id, center_x, center_y, width, height)
- Implement data augmentation techniques (rotation, brightness, contrast, horizontal flip, etc.)
- Create proper train/validation/test data loaders with batch processing

**Visualizations:**
- Dataset statistics (class distribution, image sizes, bounding box sizes)
- Sample images with bounding box overlays
- Data augmentation examples showing before/after transformations
- Training progress plots (loss curves, accuracy metrics)
- Confusion matrix and classification reports
- ROC curves and precision-recall curves

**Model Architecture:**
- Implement a state-of-the-art object detection model (YOLOv8 or YOLOv5)
- Alternative: Custom CNN architecture with detection head
- Transfer learning from pretrained models (COCO or ImageNet)
- Model summary and architecture visualization

**Training Pipeline:**
- Proper loss functions for object detection (bbox regression + classification)
- Optimizer setup (Adam/SGD with learning rate scheduling)
- Training loop with validation monitoring
- Early stopping and model checkpointing
- Mixed precision training for efficiency
- Progress tracking with tqdm and logging

**Evaluation & Analytics:**
- mAP (mean Average Precision) calculation at different IoU thresholds
- Per-class performance metrics
- Inference speed benchmarking
- Test set evaluation with detailed metrics
- Error analysis (false positives, false negatives)

**Feature Analysis & Sensitivity Mapping:**
- Grad-CAM or similar attention visualization
- Feature map visualizations from intermediate layers
- Saliency maps showing regions of high sensitivity
- LIME or SHAP explanations for model predictions
- Occlusion sensitivity analysis

**Model Deployment:**
- Save trained model in multiple formats (PyTorch .pth, ONNX)
- Inference function for single image prediction
- Batch inference capabilities
- Performance optimization techniques

**Additional Features:**
- Comprehensive logging and experiment tracking
- Configuration management (hyperparameters, paths)
- GPU/CPU compatibility with automatic device selection
- Memory optimization techniques
- Error handling and input validation

### **2. Setup Script: `setup.sh`**

Create a bash script that:
- Creates a Python virtual environment (`eye_detection_env`)
- Activates the environment
- Upgrades pip to latest version
- Installs all required packages from requirements.txt
- Verifies installation success
- Provides instructions for activation

### **3. Requirements File: `requirements.txt`**

Include all necessary packages:
- PyTorch and torchvision (with CUDA support)
- OpenCV for image processing
- Matplotlib and seaborn for visualizations
- NumPy and pandas for data manipulation
- scikit-learn for metrics
- Pillow for image handling
- tqdm for progress bars
- ultralytics for YOLO implementation
- tensorboard for training monitoring
- LIME and SHAP for explainability
- albumentations for advanced augmentations
- wandb or mlflow for experiment tracking (optional)

### **4. README.md**

Create comprehensive documentation including:
- **Project Overview**: Eye/cataract detection using deep learning
- **Dataset Description**: YOLO format dataset structure and statistics
- **Installation Instructions**: Step-by-step setup guide
- **Usage Examples**: How to run training, evaluation, and inference
- **Model Architecture**: Technical details about the model
- **Performance Results**: Expected metrics and benchmarks
- **Feature Analysis**: Explanation of sensitivity mapping
- **Troubleshooting**: Common issues and solutions
- **Future Improvements**: Potential enhancements
- **References**: Citations to similar research papers and related work

### **5. Configuration File: `config.py`**

Create a YAML configuration file for:
- Dataset paths and parameters
- Model hyperparameters
- Training configuration
- Evaluation settings
- Visualization options

### **6. Folders**

Create the following folders to arrange the created files into
- reports: training, testing and evaluation progess of models as json data
- visualizations: all generated graphs for analysis and model evaluations
- models: trained models saved for future use

### **7. Test File: `test.py`**

A python file that does the following
- Loads the test/validation data
- Processes the data appropriately
- Loads the trained models
- Performs predictions on the data using the models
- Prints/logs the performance of the model to these data

### **Visual Requirements**

The main file must save visualizations for the following
- Any numerical analysis done prior to model training
- Results of model training, testing and validation for every metric
- Grad-CAM or other visualization plots used in the project

These must be properly documented in the readme file and the naming must be easily identifiable.

### **Technical Requirements:**

1. **Code Quality:**
   - Follow PEP 8 style guidelines
   - Include comprehensive docstrings
   - Add type hints where appropriate
   - Implement proper error handling
   - Use logging instead of print statements

2. **Performance:**
   - Optimize for both training and inference speed
   - Implement efficient data loading with multiprocessing
   - Use mixed precision training
   - Memory-efficient batch processing

3. **Reproducibility:**
   - Set random seeds for reproducible results
   - Save model checkpoints with metadata
   - Log hyperparameters and configurations
   - Version control integration ready

4. **Scalability:**
   - Support for different input image sizes
   - Configurable batch sizes
   - Multi-GPU training support (if available)
   - Flexible model architectures

The final system should be production-ready, well-documented, and provide comprehensive insights into both model performance and the features that drive eye/cataract detection decisions.
