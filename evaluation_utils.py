"""
Advanced utilities for model evaluation and benchmarking.
"""

import time
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate deepfake detection model performance."""
    
    @staticmethod
    def benchmark_inference(model, test_images: List[np.ndarray], 
                          num_iterations: int = 10) -> Dict:
        """
        Benchmark model inference time.
        
        Args:
            model: Detection model
            test_images: List of test images
            num_iterations: Number of iterations for benchmarking
        
        Returns:
            Dictionary with benchmark results
        """
        times = []
        
        for _ in range(num_iterations):
            for image in test_images:
                start = time.time()
                model.predict(image)
                times.append(time.time() - start)
        
        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "total_iterations": num_iterations * len(test_images)
        }
    
    @staticmethod
    def calculate_metrics(predictions: List[float], ground_truth: List[int]) -> Dict:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions (confidence scores 0-1)
            ground_truth: Ground truth labels (0 or 1)
        
        Returns:
            Dictionary with precision, recall, f1-score, etc.
        """
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Binary predictions at threshold 0.5
        binary_preds = (predictions > 0.5).astype(int)
        
        # True Positives, False Positives, True Negatives, False Negatives
        tp = np.sum((binary_preds == 1) & (ground_truth == 1))
        fp = np.sum((binary_preds == 1) & (ground_truth == 0))
        tn = np.sum((binary_preds == 0) & (ground_truth == 0))
        fn = np.sum((binary_preds == 0) & (ground_truth == 1))
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }


class PerformanceMonitor:
    """Monitor system performance during inference."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {
            "total_images": 0,
            "total_videos": 0,
            "total_faces": 0,
            "average_inference_time": 0,
            "start_time": None
        }
    
    def start(self):
        """Start monitoring."""
        self.metrics["start_time"] = time.time()
    
    def record_image_analysis(self, processing_time: float, num_faces: int):
        """Record image analysis metrics."""
        self.metrics["total_images"] += 1
        self.metrics["total_faces"] += num_faces
        self._update_average_time(processing_time)
    
    def record_video_analysis(self, processing_time: float, num_faces: int):
        """Record video analysis metrics."""
        self.metrics["total_videos"] += 1
        self.metrics["total_faces"] += num_faces
        self._update_average_time(processing_time)
    
    def _update_average_time(self, new_time: float):
        """Update average inference time."""
        total = self.metrics["total_images"] + self.metrics["total_videos"]
        current_avg = self.metrics["average_inference_time"]
        self.metrics["average_inference_time"] = (
            (current_avg * (total - 1) + new_time) / total
        )
    
    def get_summary(self) -> Dict:
        """Get monitoring summary."""
        elapsed = time.time() - self.metrics["start_time"] if self.metrics["start_time"] else 0
        
        return {
            "total_images": self.metrics["total_images"],
            "total_videos": self.metrics["total_videos"],
            "total_faces": self.metrics["total_faces"],
            "average_inference_time": self.metrics["average_inference_time"],
            "elapsed_time": elapsed,
            "throughput": self.metrics["total_faces"] / elapsed if elapsed > 0 else 0
        }


class DataQualityChecker:
    """Check quality of media data for processing."""
    
    @staticmethod
    def check_image_quality(image: np.ndarray) -> Dict:
        """
        Check image quality metrics.
        
        Args:
            image: Image array
        
        Returns:
            Dictionary with quality metrics
        """
        # Check dimensions
        min_dim = min(image.shape[:2])
        
        # Check brightness
        brightness = np.mean(image.astype(float))
        
        # Check contrast
        contrast = np.std(image.astype(float))
        
        # Check for blur (simplified Laplacian variance)
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return {
            "min_dimension": int(min_dim),
            "brightness": float(brightness),
            "contrast": float(contrast),
            "blur_score": float(laplacian_var),
            "quality_warnings": []
        }
    
    @staticmethod
    def check_video_properties(video_props: Dict) -> Dict:
        """
        Check video properties for processing.
        
        Args:
            video_props: Video properties dictionary
        
        Returns:
            Dictionary with quality assessment
        """
        warnings = []
        
        if video_props.get("fps", 0) < 24:
            warnings.append("Low frame rate (< 24 fps)")
        
        if video_props.get("width", 0) < 320 or video_props.get("height", 0) < 240:
            warnings.append("Low resolution (< 320x240)")
        
        if video_props.get("duration_seconds", 0) > 3600:
            warnings.append("Very long video (> 1 hour) - may require optimization")
        
        return {
            "fps": video_props.get("fps", 0),
            "resolution": f"{video_props.get('width', 0)}x{video_props.get('height', 0)}",
            "duration": video_props.get("duration_seconds", 0),
            "quality_warnings": warnings
        }
