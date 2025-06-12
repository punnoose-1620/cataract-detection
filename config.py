# Configuration file for Eye/Cataract Detection System
import os
from pathlib import Path

class Config:
    """
    Configuration class containing all hyperparameters and settings
    for the eye/cataract detection system.
    """
    
    # ========================
    # DATASET CONFIGURATION
    # ========================
    DATASET_ROOT = Path("Dataset")
    TRAIN_IMAGES_PATH = DATASET_ROOT / "train" / "images"
    TRAIN_LABELS_PATH = DATASET_ROOT / "train" / "labels"
    VAL_IMAGES_PATH = DATASET_ROOT / "val" / "images"
    VAL_LABELS_PATH = DATASET_ROOT / "val" / "labels"
    TEST_IMAGES_PATH = DATASET_ROOT / "test" / "images"
    TEST_LABELS_PATH = DATASET_ROOT / "test" / "labels"
    
    # Class configuration
    NUM_CLASSES = 2  # Background and Cataract/Eye condition
    CLASS_NAMES = ['normal', 'cataract']
    
    # ========================
    # MODEL CONFIGURATION
    # ========================
    MODEL_NAME = "yolov8n"  # YOLOv8 nano for faster training/inference
    MODEL_VARIANTS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
    PRETRAINED = True
    INPUT_SIZE = (640, 640)  # (width, height)
    
    # ========================
    # TRAINING CONFIGURATION
    # ========================
    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0005
    MOMENTUM = 0.937
    
    # Learning rate scheduler
    LR_SCHEDULER = "cosine"  # "cosine", "linear", "step"
    LR_FACTOR = 0.1
    LR_PATIENCE = 10
    
    # Early stopping
    EARLY_STOPPING = True
    PATIENCE = 15
    MIN_DELTA = 0.001
    
    # Mixed precision training
    MIXED_PRECISION = True
    
    # ========================
    # DATA AUGMENTATION
    # ========================
    DATA_AUGMENTATION = {
        'hsv_h': 0.015,        # HSV-Hue augmentation
        'hsv_s': 0.7,          # HSV-Saturation augmentation
        'hsv_v': 0.4,          # HSV-Value augmentation
        'degrees': 10.0,       # Rotation degrees
        'translate': 0.1,      # Translation
        'scale': 0.5,          # Scale
        'shear': 0.0,          # Shear
        'perspective': 0.0,    # Perspective
        'flipud': 0.0,         # Vertical flip probability
        'fliplr': 0.5,         # Horizontal flip probability
        'mosaic': 1.0,         # Mosaic augmentation probability
        'mixup': 0.0,          # MixUp augmentation probability
        'copy_paste': 0.0      # Copy-paste augmentation probability
    }
    
    # ========================
    # VALIDATION CONFIGURATION
    # ========================
    VAL_SPLIT = 0.2
    CONF_THRESHOLD = 0.25      # Confidence threshold for predictions
    IOU_THRESHOLD = 0.45       # IoU threshold for NMS
    MAX_DETECTIONS = 1000      # Maximum detections per image
    
    # ========================
    # EVALUATION METRICS
    # ========================
    IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    mAP_IOU_THRESHOLD = 0.5    # IoU threshold for mAP calculation
    
    # ========================
    # PATHS AND DIRECTORIES
    # ========================
    OUTPUT_DIR = Path("outputs")
    MODELS_DIR = Path("models")
    REPORTS_DIR = Path("reports")
    VISUALIZATIONS_DIR = Path("visualizations")
    LOGS_DIR = Path("logs")
    
    # Create directories if they don't exist
    for directory in [OUTPUT_DIR, MODELS_DIR, REPORTS_DIR, VISUALIZATIONS_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Model saving
    SAVE_BEST_MODEL = True
    SAVE_LAST_MODEL = True
    MODEL_SAVE_PATH = MODELS_DIR / "best_model.pt"
    CHECKPOINT_PATH = MODELS_DIR / "checkpoint.pt"
    
    # ========================
    # DEVICE CONFIGURATION
    # ========================
    DEVICE = "auto"  # "auto", "cpu", "cuda", "mps"
    WORKERS = 8      # Number of data loading workers
    
    # ========================
    # LOGGING AND MONITORING
    # ========================
    VERBOSE = True
    SAVE_PLOTS = True
    SAVE_CONFUSION_MATRIX = True
    TENSORBOARD_LOG = True
    WANDB_PROJECT = "eye-cataract-detection"
    WANDB_ENTITY = None  # Set your W&B entity/username
    
    # ========================
    # VISUALIZATION SETTINGS
    # ========================
    VIZ_SETTINGS = {
        'plot_samples': 10,        # Number of sample images to plot
        'plot_size': (12, 8),      # Figure size for plots
        'dpi': 300,                # DPI for saved plots
        'bbox_color': 'red',       # Bounding box color
        'bbox_thickness': 2,       # Bounding box line thickness
        'text_color': 'white',     # Text color for labels
        'font_size': 12,           # Font size for labels
    }
    
    # ========================
    # FEATURE ANALYSIS
    # ========================
    GRAD_CAM_LAYERS = ['model.24']  # Layers for Grad-CAM visualization
    LIME_SEGMENTS = 100             # Number of segments for LIME
    SHAP_BACKGROUND_SIZE = 50       # Background dataset size for SHAP
    
    # ========================
    # INFERENCE CONFIGURATION
    # ========================
    INFERENCE_BATCH_SIZE = 32
    INFERENCE_CONF_THRESH = 0.25
    INFERENCE_IOU_THRESH = 0.45
    SAVE_INFERENCE_RESULTS = True
    
    # ========================
    # EXPORT CONFIGURATION
    # ========================
    EXPORT_FORMATS = ['onnx', 'torchscript']  # Export formats
    ONNX_OPSET = 11                           # ONNX opset version
    SIMPLIFY_ONNX = True                      # Simplify ONNX model
    
    # ========================
    # REPRODUCIBILITY
    # ========================
    RANDOM_SEED = 42
    DETERMINISTIC = True
    
    @classmethod
    def get_yolo_config(cls):
        """
        Get configuration in YOLO format for training
        """
        return {
            'path': str(cls.DATASET_ROOT),
            'train': str(cls.TRAIN_IMAGES_PATH),
            'val': str(cls.VAL_IMAGES_PATH),
            'test': str(cls.TEST_IMAGES_PATH),
            'nc': cls.NUM_CLASSES,
            'names': cls.CLASS_NAMES
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("EYE/CATARACT DETECTION SYSTEM CONFIGURATION")
        print("=" * 60)
        
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and not callable(getattr(cls, attr_name)):
                attr_value = getattr(cls, attr_name)
                if not attr_name.startswith('get_') and not attr_name.startswith('print_'):
                    print(f"{attr_name}: {attr_value}")
        
        print("=" * 60)


# Create a global config instance
config = Config()

# YAML configuration string for YOLOv8
YOLO_CONFIG_YAML = f"""
# Eye/Cataract Detection Dataset Configuration

# Dataset paths
path: {config.DATASET_ROOT}
train: {config.TRAIN_IMAGES_PATH}
val: {config.VAL_IMAGES_PATH}
test: {config.TEST_IMAGES_PATH}

# Number of classes
nc: {config.NUM_CLASSES}

# Class names
names:
{chr(10).join([f"  {i}: {name}" for i, name in enumerate(config.CLASS_NAMES)])}

# Additional metadata
download: False
"""

if __name__ == "__main__":
    config.print_config() 