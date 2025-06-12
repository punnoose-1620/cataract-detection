#!/usr/bin/env python3
"""
Eye/Cataract Detection Testing Script
====================================

Script for testing and evaluating trained models on validation and test datasets.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime

# Computer Vision
import cv2
from PIL import Image

# Deep Learning
import torch
import torch.nn as nn
from ultralytics import YOLO

# Metrics
from sklearn.metrics import classification_report, confusion_matrix
import torchmetrics

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
from config import config

class ModelTester:
    """Class for testing and evaluating trained models"""
    
    def __init__(self, model_path: str):
        """
        Initialize the model tester
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.results = {}
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the tester"""
        log_file = config.LOGS_DIR / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load the trained model"""
        try:
            if self.model_path.suffix == '.pt':
                # Load YOLO model
                self.model = YOLO(str(self.model_path))
                self.logger.info(f"Successfully loaded YOLO model from {self.model_path}")
            else:
                raise ValueError(f"Unsupported model format: {self.model_path.suffix}")
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def test_on_dataset(self, dataset_split: str = 'test') -> Dict:
        """
        Test model on specified dataset split
        
        Args:
            dataset_split: Dataset split to test on ('test', 'val')
            
        Returns:
            Dictionary containing test results
        """
        if self.model is None:
            self.load_model()
        
        # Get dataset paths
        if dataset_split == 'test':
            images_dir = config.TEST_IMAGES_PATH
        elif dataset_split == 'val':
            images_dir = config.VAL_IMAGES_PATH
        else:
            raise ValueError(f"Invalid dataset split: {dataset_split}")
        
        if not images_dir.exists():
            self.logger.error(f"Dataset path does not exist: {images_dir}")
            return {}
        
        self.logger.info(f"Testing model on {dataset_split} dataset...")
        
        # Run inference
        results = self.model(
            source=str(images_dir),
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            save=True,
            save_dir=config.OUTPUT_DIR / f"{dataset_split}_results",
            name=f"{dataset_split}_inference"
        )
        
        # Calculate metrics
        metrics = self.calculate_metrics(results, dataset_split)
        
        # Save results
        self.save_results(metrics, dataset_split)
        
        return metrics
    
    def calculate_metrics(self, results, dataset_split: str) -> Dict:
        """
        Calculate evaluation metrics from inference results
        
        Args:
            results: YOLO inference results
            dataset_split: Dataset split name
            
        Returns:
            Dictionary containing calculated metrics
        """
        metrics = {
            'dataset_split': dataset_split,
            'model_path': str(self.model_path),
            'timestamp': datetime.now().isoformat(),
            'config': {
                'conf_threshold': config.CONF_THRESHOLD,
                'iou_threshold': config.IOU_THRESHOLD,
                'model_name': config.MODEL_NAME
            }
        }
        
        try:
            # Get validation metrics if available
            if hasattr(self.model, 'val'):
                val_results = self.model.val(
                    data=config.get_yolo_config(),
                    split=dataset_split,
                    conf=config.CONF_THRESHOLD,
                    iou=config.IOU_THRESHOLD
                )
                
                if val_results:
                    # Extract metrics
                    metrics.update({
                        'mAP50': float(val_results.box.map50),
                        'mAP50_95': float(val_results.box.map),
                        'precision': float(val_results.box.mp),
                        'recall': float(val_results.box.mr),
                        'f1_score': 2 * (float(val_results.box.mp) * float(val_results.box.mr)) / 
                                   (float(val_results.box.mp) + float(val_results.box.mr) + 1e-8)
                    })
                    
                    # Per-class metrics
                    if hasattr(val_results.box, 'ap_class_index'):
                        class_metrics = {}
                        for i, class_idx in enumerate(val_results.box.ap_class_index):
                            class_name = config.CLASS_NAMES[int(class_idx)]
                            class_metrics[class_name] = {
                                'precision': float(val_results.box.p[i]) if i < len(val_results.box.p) else 0.0,
                                'recall': float(val_results.box.r[i]) if i < len(val_results.box.r) else 0.0,
                                'mAP50': float(val_results.box.ap50[i]) if i < len(val_results.box.ap50) else 0.0,
                                'mAP50_95': float(val_results.box.ap[i]) if i < len(val_results.box.ap) else 0.0,
                            }
                        metrics['per_class_metrics'] = class_metrics
        
        except Exception as e:
            self.logger.warning(f"Could not calculate detailed metrics: {e}")
            
        return metrics
    
    def save_results(self, metrics: Dict, dataset_split: str):
        """
        Save test results to file
        
        Args:
            metrics: Calculated metrics
            dataset_split: Dataset split name
        """
        # Save as JSON
        results_file = config.REPORTS_DIR / f"{dataset_split}_results.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_file}")
        
        # Generate summary report
        self.generate_summary_report(metrics, dataset_split)
    
    def generate_summary_report(self, metrics: Dict, dataset_split: str):
        """
        Generate a human-readable summary report
        
        Args:
            metrics: Calculated metrics
            dataset_split: Dataset split name
        """
        report_file = config.REPORTS_DIR / f"{dataset_split}_summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"EYE/CATARACT DETECTION - {dataset_split.upper()} RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Model: {metrics.get('model_path', 'Unknown')}\n")
            f.write(f"Dataset Split: {metrics.get('dataset_split', 'Unknown')}\n")
            f.write(f"Test Date: {metrics.get('timestamp', 'Unknown')}\n\n")
            
            f.write("Configuration:\n")
            config_info = metrics.get('config', {})
            for key, value in config_info.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("Overall Metrics:\n")
            f.write(f"  mAP@0.5: {metrics.get('mAP50', 'N/A'):.4f}\n")
            f.write(f"  mAP@0.5:0.95: {metrics.get('mAP50_95', 'N/A'):.4f}\n")
            f.write(f"  Precision: {metrics.get('precision', 'N/A'):.4f}\n")
            f.write(f"  Recall: {metrics.get('recall', 'N/A'):.4f}\n")
            f.write(f"  F1-Score: {metrics.get('f1_score', 'N/A'):.4f}\n\n")
            
            if 'per_class_metrics' in metrics:
                f.write("Per-Class Metrics:\n")
                for class_name, class_metrics in metrics['per_class_metrics'].items():
                    f.write(f"  {class_name}:\n")
                    f.write(f"    Precision: {class_metrics.get('precision', 'N/A'):.4f}\n")
                    f.write(f"    Recall: {class_metrics.get('recall', 'N/A'):.4f}\n")
                    f.write(f"    mAP@0.5: {class_metrics.get('mAP50', 'N/A'):.4f}\n")
                    f.write(f"    mAP@0.5:0.95: {class_metrics.get('mAP50_95', 'N/A'):.4f}\n")
                    f.write("\n")
        
        self.logger.info(f"Summary report saved to {report_file}")
        
        # Print summary to console
        print("\n" + "=" * 60)
        print(f"TEST RESULTS SUMMARY - {dataset_split.upper()}")
        print("=" * 60)
        print(f"mAP@0.5: {metrics.get('mAP50', 'N/A'):.4f}")
        print(f"mAP@0.5:0.95: {metrics.get('mAP50_95', 'N/A'):.4f}")
        print(f"Precision: {metrics.get('precision', 'N/A'):.4f}")
        print(f"Recall: {metrics.get('recall', 'N/A'):.4f}")
        print(f"F1-Score: {metrics.get('f1_score', 'N/A'):.4f}")
        print("=" * 60)
    
    def benchmark_inference_speed(self, num_images: int = 100) -> Dict:
        """
        Benchmark model inference speed
        
        Args:
            num_images: Number of images to test
            
        Returns:
            Dictionary with speed benchmarks
        """
        if self.model is None:
            self.load_model()
        
        self.logger.info(f"Benchmarking inference speed on {num_images} images...")
        
        # Get test images
        test_images = list(config.TEST_IMAGES_PATH.glob("*.jpg"))[:num_images]
        if len(test_images) < num_images:
            test_images = list(config.VAL_IMAGES_PATH.glob("*.jpg"))[:num_images]
        
        if len(test_images) == 0:
            self.logger.error("No test images found for benchmarking")
            return {}
        
        # Warm up
        if len(test_images) > 0:
            _ = self.model(str(test_images[0]))
        
        # Benchmark
        import time
        start_time = time.time()
        
        for img_path in test_images:
            _ = self.model(str(img_path))
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_image = total_time / len(test_images)
        fps = 1.0 / avg_time_per_image
        
        benchmark_results = {
            'num_images': len(test_images),
            'total_time': total_time,
            'avg_time_per_image': avg_time_per_image,
            'fps': fps,
            'model_path': str(self.model_path)
        }
        
        # Save benchmark results
        benchmark_file = config.REPORTS_DIR / "inference_benchmark.json"
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        self.logger.info(f"Inference speed: {fps:.2f} FPS ({avg_time_per_image*1000:.2f} ms/image)")
        
        return benchmark_results

def main():
    """Main testing function"""
    print("=" * 60)
    print("EYE/CATARACT DETECTION - MODEL TESTING")
    print("=" * 60)
    
    # Check if model exists
    model_path = config.MODEL_SAVE_PATH
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first by running: python eye_detection_model.py")
        return
    
    # Create tester
    tester = ModelTester(model_path)
    
    # Test on validation set
    print("\nTesting on validation set...")
    val_results = tester.test_on_dataset('val')
    
    # Test on test set
    print("\nTesting on test set...")
    test_results = tester.test_on_dataset('test')
    
    # Benchmark inference speed
    print("\nBenchmarking inference speed...")
    speed_results = tester.benchmark_inference_speed()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)
    print(f"Results saved to: {config.REPORTS_DIR}")
    print(f"Visualizations saved to: {config.VISUALIZATIONS_DIR}")

if __name__ == "__main__":
    main() 