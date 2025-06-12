#!/usr/bin/env python3
"""
Eye/Cataract Detection System using YOLO
==========================================

Comprehensive deep learning system for detecting cataracts and eye conditions
using state-of-the-art object detection models.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import random
import numpy as np
import pandas as pd
from datetime import datetime
import yaml

# Computer Vision & Image Processing
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Deep Learning
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import timm

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Metrics and Analysis
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import torchmetrics

# Explainability
import lime
from lime import lime_image
import shap
try:
    from captum.attr import GradCAM, Saliency, IntegratedGradients
except ImportError:
    print("Warning: Captum imports failed. Feature analysis will be limited.")

# Progress tracking
from tqdm import tqdm
import wandb

# Configuration
from config import config, YOLO_CONFIG_YAML

# Suppress warnings
warnings.filterwarnings('ignore')
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')

class EyeDetectionDataset(Dataset):
    """
    Custom Dataset class for loading YOLO format eye/cataract detection data
    """
    
    def __init__(self, images_dir: str, labels_dir: str, transform=None, class_names=None):
        """
        Initialize the dataset
        
        Args:
            images_dir: Path to images directory
            labels_dir: Path to labels directory  
            transform: Data augmentation transforms
            class_names: List of class names
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.class_names = class_names or config.CLASS_NAMES
        
        # Get all image files
        self.image_files = list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))
        self.image_files.sort()
        
        logging.info(f"Loaded {len(self.image_files)} images from {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get item by index"""
        img_path = self.image_files[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load corresponding label
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    values = line.strip().split()
                    if len(values) >= 5:
                        class_id = int(values[0])
                        center_x = float(values[1])
                        center_y = float(values[2])
                        width = float(values[3])
                        height = float(values[4])
                        
                        # Convert YOLO format to bbox format
                        x1 = center_x - width/2
                        y1 = center_y - height/2
                        x2 = center_x + width/2
                        y2 = center_y + height/2
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'image_path': str(img_path)
        }

class DataAnalyzer:
    """Class for analyzing dataset statistics and creating visualizations"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.stats = {}
        
    def analyze_dataset(self) -> Dict:
        """Analyze dataset and return statistics"""
        logging.info("Analyzing dataset statistics...")
        
        splits = ['train', 'val', 'test']
        total_stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': {name: 0 for name in config.CLASS_NAMES},
            'image_sizes': [],
            'bbox_sizes': [],
            'split_stats': {}
        }
        
        for split in splits:
            images_dir = self.dataset_path / split / 'images'
            labels_dir = self.dataset_path / split / 'labels'
            
            if not images_dir.exists():
                continue
                
            split_stats = {
                'num_images': 0,
                'num_annotations': 0,
                'class_counts': {name: 0 for name in config.CLASS_NAMES}
            }
            
            # Analyze images
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            split_stats['num_images'] = len(image_files)
            
            for img_path in tqdm(image_files, desc=f"Analyzing {split} images"):
                # Get image size
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    total_stats['image_sizes'].append((w, h))
                
                # Analyze corresponding label
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f.readlines():
                            values = line.strip().split()
                            if len(values) >= 5:
                                class_id = int(values[0])
                                width = float(values[3])
                                height = float(values[4])
                                
                                if class_id < len(config.CLASS_NAMES):
                                    class_name = config.CLASS_NAMES[class_id]
                                    split_stats['class_counts'][class_name] += 1
                                    total_stats['class_distribution'][class_name] += 1
                                    total_stats['bbox_sizes'].append((width, height))
                                    split_stats['num_annotations'] += 1
            
            total_stats['split_stats'][split] = split_stats
            total_stats['total_images'] += split_stats['num_images']
            total_stats['total_annotations'] += split_stats['num_annotations']
        
        self.stats = total_stats
        
        # Save statistics
        with open(config.REPORTS_DIR / 'dataset_statistics.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            stats_copy = total_stats.copy()
            stats_copy['image_sizes'] = [list(size) for size in stats_copy['image_sizes']]
            stats_copy['bbox_sizes'] = [list(size) for size in stats_copy['bbox_sizes']]
            json.dump(stats_copy, f, indent=2)
        
        return total_stats
    
    def visualize_statistics(self):
        """Create visualizations for dataset statistics"""
        if not self.stats:
            self.analyze_dataset()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Class Distribution
        classes = list(self.stats['class_distribution'].keys())
        counts = list(self.stats['class_distribution'].values())
        
        axes[0, 0].pie(counts, labels=classes, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Class Distribution')
        
        # 2. Split Distribution
        splits = list(self.stats['split_stats'].keys())
        split_counts = [self.stats['split_stats'][split]['num_images'] for split in splits]
        
        axes[0, 1].bar(splits, split_counts, color=['blue', 'orange', 'green'])
        axes[0, 1].set_title('Images per Split')
        axes[0, 1].set_ylabel('Number of Images')
        
        # 3. Image Size Distribution
        if self.stats['image_sizes']:
            widths, heights = zip(*self.stats['image_sizes'])
            axes[0, 2].scatter(widths, heights, alpha=0.6)
            axes[0, 2].set_xlabel('Width')
            axes[0, 2].set_ylabel('Height')
            axes[0, 2].set_title('Image Size Distribution')
        
        # 4. Bounding Box Size Distribution
        if self.stats['bbox_sizes']:
            bbox_widths, bbox_heights = zip(*self.stats['bbox_sizes'])
            axes[1, 0].hist2d(bbox_widths, bbox_heights, bins=30, cmap='Blues')
            axes[1, 0].set_xlabel('Bbox Width (normalized)')
            axes[1, 0].set_ylabel('Bbox Height (normalized)')
            axes[1, 0].set_title('Bounding Box Size Distribution')
        
        # 5. Class Distribution by Split
        split_names = list(self.stats['split_stats'].keys())
        class_names = config.CLASS_NAMES
        
        x = np.arange(len(split_names))
        width = 0.35
        
        for i, class_name in enumerate(class_names):
            class_counts = [self.stats['split_stats'][split]['class_counts'][class_name] 
                          for split in split_names]
            axes[1, 1].bar(x + i*width, class_counts, width, label=class_name)
        
        axes[1, 1].set_xlabel('Dataset Split')
        axes[1, 1].set_ylabel('Number of Annotations')
        axes[1, 1].set_title('Class Distribution by Split')
        axes[1, 1].set_xticks(x + width/2)
        axes[1, 1].set_xticklabels(split_names)
        axes[1, 1].legend()
        
        # 6. Summary Statistics
        axes[1, 2].axis('off')
        summary_text = f"""
        Dataset Summary:
        
        Total Images: {self.stats['total_images']:,}
        Total Annotations: {self.stats['total_annotations']:,}
        
        Average Images per Class:
        """
        
        for class_name, count in self.stats['class_distribution'].items():
            summary_text += f"  {class_name}: {count:,}\n"
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(config.VISUALIZATIONS_DIR / 'dataset_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        logging.info(f"Dataset analysis saved to {config.VISUALIZATIONS_DIR / 'dataset_analysis.png'}")

def setup_logging():
    """Setup logging configuration"""
    log_file = config.LOGS_DIR / f"eye_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ModelTrainer:
    """Class for training YOLO models"""
    
    def __init__(self):
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    def create_model(self, model_name: str = None):
        """Create and initialize YOLO model"""
        model_name = model_name or config.MODEL_NAME
        
        try:
            # Initialize YOLO model
            self.model = YOLO(f"{model_name}.pt")
            self.logger.info(f"Initialized {model_name} model")
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"Failed to create model: {e}")
            raise
    
    def train_model(self):
        """Train the YOLO model"""
        if self.model is None:
            self.create_model()
        
        self.logger.info("Starting model training...")
        
        # Create dataset config
        dataset_config = {
            'path': str(config.DATASET_ROOT),
            'train': str(config.TRAIN_IMAGES_PATH.relative_to(config.DATASET_ROOT)),
            'val': str(config.VAL_IMAGES_PATH.relative_to(config.DATASET_ROOT)),
            'nc': config.NUM_CLASSES,
            'names': config.CLASS_NAMES
        }
        
        # Save dataset config
        dataset_yaml = config.OUTPUT_DIR / 'dataset.yaml'
        with open(dataset_yaml, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        try:
            # Train the model
            results = self.model.train(
                data=str(dataset_yaml),
                epochs=config.EPOCHS,
                batch=config.BATCH_SIZE,
                imgsz=config.INPUT_SIZE[0],
                lr0=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY,
                momentum=config.MOMENTUM,
                patience=config.PATIENCE if config.EARLY_STOPPING else 0,
                save=True,
                save_period=10,
                cache=False,
                device=config.DEVICE,
                workers=config.WORKERS,
                project=str(config.OUTPUT_DIR),
                name="training",
                exist_ok=True,
                pretrained=config.PRETRAINED,
                optimizer='auto',
                verbose=config.VERBOSE,
                seed=config.RANDOM_SEED,
                deterministic=config.DETERMINISTIC,
                single_cls=False,
                rect=False,
                cos_lr=config.LR_SCHEDULER == 'cosine',
                close_mosaic=10,
                resume=False,
                amp=config.MIXED_PRECISION,
                fraction=1.0,
                profile=False,
                freeze=None,
                **config.DATA_AUGMENTATION
            )
            
            self.logger.info("Training completed successfully!")
            
            # Save best model
            best_model_path = config.OUTPUT_DIR / "training" / "weights" / "best.pt"
            if best_model_path.exists():
                import shutil
                shutil.copy(best_model_path, config.MODEL_SAVE_PATH)
                self.logger.info(f"Best model saved to {config.MODEL_SAVE_PATH}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def validate_model(self):
        """Validate the trained model"""
        if self.model is None:
            self.logger.error("No model loaded for validation")
            return None
        
        self.logger.info("Validating model...")
        
        try:
            # Run validation
            results = self.model.val(
                data=str(config.OUTPUT_DIR / 'dataset.yaml'),
                split='val',
                imgsz=config.INPUT_SIZE[0],
                batch=config.BATCH_SIZE,
                conf=config.CONF_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                device=config.DEVICE,
                save_json=True,
                save_hybrid=False,
                verbose=config.VERBOSE,
                project=str(config.OUTPUT_DIR),
                name="validation"
            )
            
            self.logger.info("Validation completed!")
            return results
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return None

class VisualizationGenerator:
    """Class for generating various visualizations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def plot_training_results(self, results_path: str):
        """Plot training results"""
        try:
            # Read training results
            results_file = Path(results_path) / "results.csv"
            
            if not results_file.exists():
                self.logger.warning(f"Results file not found: {results_file}")
                return
            
            df = pd.read_csv(results_file)
            df.columns = df.columns.str.strip()  # Remove whitespace
            
            # Create training plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
            
            # Loss plots
            if 'train/box_loss' in df.columns:
                axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', color='blue')
                if 'val/box_loss' in df.columns:
                    axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', color='red')
                axes[0, 0].set_title('Box Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # mAP plots
            if 'metrics/mAP50(B)' in df.columns:
                axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='green')
                if 'metrics/mAP50-95(B)' in df.columns:
                    axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='purple')
                axes[0, 1].set_title('Mean Average Precision')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('mAP')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Precision/Recall
            if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
                axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='orange')
                axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='brown')
                axes[1, 0].set_title('Precision and Recall')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Learning rate
            if 'lr/pg0' in df.columns:
                axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='Learning Rate', color='red')
                axes[1, 1].set_title('Learning Rate Schedule')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(config.VISUALIZATIONS_DIR / 'training_curves.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            self.logger.info("Training curves saved!")
            
        except Exception as e:
            self.logger.error(f"Failed to plot training results: {e}")
    
    def plot_sample_predictions(self, model_path: str, num_samples: int = 6):
        """Plot sample predictions"""
        try:
            model = YOLO(model_path)
            
            # Get sample images
            test_images = list(config.TEST_IMAGES_PATH.glob("*.jpg"))[:num_samples]
            if len(test_images) < num_samples:
                test_images.extend(list(config.VAL_IMAGES_PATH.glob("*.jpg"))[:num_samples-len(test_images)])
            
            if not test_images:
                self.logger.warning("No test images found for visualization")
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
            
            for i, img_path in enumerate(test_images[:6]):
                if i >= 6:
                    break
                
                # Run inference
                results = model(str(img_path))
                
                # Plot results
                row, col = i // 3, i % 3
                
                # Get original image
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Draw predictions
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Draw bounding box
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        
                        # Draw label
                        label = f"{config.CLASS_NAMES[cls]}: {conf:.2f}"
                        cv2.putText(img, label, (int(x1), int(y1)-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"Image {i+1}")
                axes[row, col].axis('off')
            
            # Hide empty subplots
            for i in range(len(test_images), 6):
                row, col = i // 3, i % 3
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(config.VISUALIZATIONS_DIR / 'sample_predictions.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            self.logger.info("Sample predictions visualization saved!")
            
        except Exception as e:
            self.logger.error(f"Failed to create sample predictions: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Eye/Cataract Detection System')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze dataset')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Run inference')
    parser.add_argument('--source', type=str, help='Source for prediction')
    parser.add_argument('--export', action='store_true', help='Export model')
    parser.add_argument('--format', type=str, default='onnx', help='Export format')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    set_random_seed(config.RANDOM_SEED)
    
    logger.info("="*60)
    logger.info("EYE/CATARACT DETECTION SYSTEM")
    logger.info("="*60)
    
    # Print configuration
    config.print_config()
    
    # Analyze dataset
    logger.info("Analyzing dataset...")
    analyzer = DataAnalyzer(config.DATASET_ROOT)
    stats = analyzer.analyze_dataset()
    analyzer.visualize_statistics()
    
    if args.analyze_only:
        logger.info("Dataset analysis completed!")
        return
    
    # Training
    if args.train:
        logger.info("Starting training pipeline...")
        trainer = ModelTrainer()
        
        # Train model
        results = trainer.train_model()
        
        # Validate model
        val_results = trainer.validate_model()
        
        # Generate visualizations
        viz_gen = VisualizationGenerator()
        viz_gen.plot_training_results(config.OUTPUT_DIR / "training")
        viz_gen.plot_sample_predictions(str(config.MODEL_SAVE_PATH))
        
        logger.info("Training pipeline completed!")
    
    # Prediction
    if args.predict:
        if not config.MODEL_SAVE_PATH.exists():
            logger.error("No trained model found. Please train first.")
            return
        
        source = args.source or str(config.TEST_IMAGES_PATH)
        logger.info(f"Running inference on {source}")
        
        model = YOLO(str(config.MODEL_SAVE_PATH))
        results = model(source, save=True, conf=config.CONF_THRESHOLD, iou=config.IOU_THRESHOLD)
        
        logger.info("Inference completed!")
    
    # Export
    if args.export:
        if not config.MODEL_SAVE_PATH.exists():
            logger.error("No trained model found. Please train first.")
            return
        
        logger.info(f"Exporting model to {args.format} format...")
        
        model = YOLO(str(config.MODEL_SAVE_PATH))
        model.export(format=args.format)
        
        logger.info("Model export completed!")

if __name__ == "__main__":
    main() 