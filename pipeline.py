"""
Main detection pipeline for deepfake detection system - ENHANCED
Features:
- Advanced face detection and deepfake classification
- Temporal consistency analysis for videos
- Motion-based anomaly detection
- Confidence aggregation with temporal weighting
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import time
from collections import deque

from models import FaceDetector, DeepfakeDetector
from utils.image_utils import ImageProcessor
from utils.video_utils import VideoProcessor
from utils.logging_utils import setup_logger
from config import IMAGE_CONFIG, VIDEO_CONFIG, DETECTION_THRESHOLDS

logger = setup_logger(__name__)


class DeepfakeDetectionPipeline:
    """
    Enhanced pipeline for deepfake detection.
    Features temporal consistency analysis and motion detection.
    """

    def __init__(self, face_detector_type: str = "mediapipe",
                 deepfake_detector_type: str = "ensemble"):
        """
        Initialize enhanced detection pipeline.
        
        Args:
            face_detector_type: Type of face detector to use
            deepfake_detector_type: Type of deepfake detector to use
        """
        self.face_detector = FaceDetector(model_type=face_detector_type)
        self.deepfake_detector = DeepfakeDetector(model_type=deepfake_detector_type)
        self.image_processor = ImageProcessor()
        self.video_processor = VideoProcessor()
        
        # Temporal tracking
        self.prev_frame = None
        self.confidence_history = deque(maxlen=5)  # Track last 5 frames
        
        logger.info(f"Enhanced Pipeline initialized with {face_detector_type} face detector "
                   f"and {deepfake_detector_type} deepfake detector")

    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze single image for deepfake content.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary with analysis results
        """
        try:
            start_time = time.time()
            logger.info(f"Starting image analysis: {image_path}")
            
            # Load image
            image = self.image_processor.load_image(image_path)
            if image is None:
                return {"error": "Failed to load image"}
            
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            
            if not faces:
                return {
                    "status": "no_faces_detected",
                    "image_path": str(image_path),
                    "confidence": 0.0,
                    "prediction": "UNKNOWN",
                    "message": "No faces detected in image",
                    "processing_time": time.time() - start_time
                }
            
            # Crop and analyze faces with enhanced predictions
            cropped_faces = self.face_detector.crop_faces(image, faces, padding=0.1)
            face_predictions = []
            
            for idx, face_roi in enumerate(cropped_faces):
                confidence, details = self.deepfake_detector.predict(face_roi)
                face_predictions.append({
                    "face_id": idx,
                    "confidence": float(confidence),
                    "prediction": details.get("prediction", "UNKNOWN"),
                    "confidence_level": details.get("confidence_level", "UNKNOWN"),
                    "quality_score": details.get("quality_score", 0.0),
                    "details": details
                })
            
            # Aggregate results
            avg_confidence = np.mean([f['confidence'] for f in face_predictions])
            overall_prediction = self._aggregate_predictions([f['confidence'] for f in face_predictions])
            
            # Draw results on image
            labels = [f"Face {i}: {p['prediction']} ({p.get('confidence_level', '')})" 
                      for i, p in enumerate(face_predictions)]
            colors = [self._get_color_for_prediction(p['prediction']) for p in face_predictions]
            result_image = self.image_processor.draw_bounding_boxes(
                image, faces, labels=labels, colors=colors
            )
            
            # Save results
            output_path = Path(image_path).parent / f"analyzed_{Path(image_path).name}"
            self.image_processor.save_image(result_image, str(output_path))
            
            return {
                "status": "success",
                "image_path": str(image_path),
                "output_path": str(output_path),
                "faces_detected": len(faces),
                "confidence": float(avg_confidence),
                "prediction": overall_prediction,
                "face_predictions": face_predictions,
                "processing_time": time.time() - start_time
            }
        
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"error": str(e)}

    def analyze_video(self, video_path: str, output_video: bool = False) -> Dict:
        """
        Analyze video for deepfake content with temporal consistency.
        
        Args:
            video_path: Path to video file
            output_video: Whether to save output video with annotations
        
        Returns:
            Dictionary with analysis results
        """
        try:
            start_time = time.time()
            logger.info(f"Starting enhanced video analysis: {video_path}")
            
            # Reset temporal state
            self.prev_frame = None
            self.confidence_history = deque(maxlen=5)
            
            # Get video properties
            props = self.video_processor.get_video_properties(video_path)
            if not props:
                return {"error": "Failed to get video properties"}
            
            logger.info(f"Video properties: FPS={props['fps']}, Frames={props['frame_count']}, "
                       f"Resolution={props['width']}x{props['height']}")
            
            frame_results = []
            analyzed_frames = 0
            fake_frames = 0
            temporal_anomalies = 0
            
            # Extract and analyze frames
            for frame_idx, frame in self.video_processor.extract_frames(
                video_path,
                interval=VIDEO_CONFIG['frame_extraction_interval'],
                max_frames=VIDEO_CONFIG['max_frames_to_process'],
                resize_width=VIDEO_CONFIG['resize_width'],
                resize_height=VIDEO_CONFIG['resize_height']
            ):
                # Detect faces in frame
                faces = self.face_detector.detect_faces(frame)
                
                # Calculate temporal consistency (motion detection)
                temporal_score = self._calculate_temporal_consistency(frame)
                
                if faces:
                    # Analyze faces
                    cropped_faces = self.face_detector.crop_faces(frame, faces)
                    
                    frame_confidences = []
                    for face_roi in cropped_faces:
                        confidence, _ = self.deepfake_detector.predict(face_roi)
                        frame_confidences.append(confidence)
                    
                    avg_confidence = np.mean(frame_confidences)
                    
                    # Apply temporal weighting
                    self.confidence_history.append(avg_confidence)
                    temporally_weighted_confidence = self._apply_temporal_weighting(avg_confidence)
                    
                    prediction = self._aggregate_predictions([temporally_weighted_confidence])
                    
                    # Detect temporal anomalies
                    is_anomaly = self._detect_temporal_anomaly(temporally_weighted_confidence)
                    if is_anomaly:
                        temporal_anomalies += 1
                    
                    frame_results.append({
                        "frame_id": frame_idx,
                        "faces_detected": len(faces),
                        "avg_confidence": float(avg_confidence),
                        "temporal_score": float(temporal_score),
                        "weighted_confidence": float(temporally_weighted_confidence),
                        "prediction": prediction,
                        "temporal_anomaly": is_anomaly
                    })
                    
                    if prediction == "FAKE":
                        fake_frames += 1
                else:
                    self.confidence_history.append(0.5)
                
                analyzed_frames += 1
            
            # Calculate overall statistics with temporal analysis
            if frame_results:
                all_confidences = [f['weighted_confidence'] for f in frame_results]
                overall_avg_confidence = np.mean(all_confidences)
                overall_prediction = self._aggregate_predictions(all_confidences)
                fake_frame_ratio = fake_frames / len(frame_results)
                temporal_anomaly_ratio = temporal_anomalies / len(frame_results)
                
                # Final verdict considering temporal consistency
                final_prediction = self._final_verdict(
                    overall_prediction,
                    temporal_anomaly_ratio,
                    overall_avg_confidence
                )
            else:
                overall_avg_confidence = 0.0
                overall_prediction = "NO_FACES"
                fake_frame_ratio = 0.0
                temporal_anomaly_ratio = 0.0
                final_prediction = "NO_FACES"
            
            processing_time = time.time() - start_time
            
            return {
                "status": "success",
                "video_path": str(video_path),
                "video_properties": props,
                "analyzed_frames": analyzed_frames,
                "total_frames": props['frame_count'],
                "frames_with_faces": len(frame_results),
                "fake_frames": fake_frames,
                "fake_frame_ratio": float(fake_frame_ratio),
                "temporal_anomalies": temporal_anomalies,
                "temporal_anomaly_ratio": float(temporal_anomaly_ratio),
                "overall_confidence": float(overall_avg_confidence),
                "overall_prediction": overall_prediction,
                "final_prediction": final_prediction,
                "frame_results": frame_results,
                "processing_time": processing_time
            }
        
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            return {"error": str(e)}

    def _calculate_temporal_consistency(self, frame: np.ndarray) -> float:
        """
        Calculate temporal consistency score using optical flow.
        
        Returns:
            Consistency score (0-1), higher is more consistent
        """
        try:
            if self.prev_frame is None:
                self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return 0.5
            
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate flow magnitude
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_magnitude = np.mean(magnitude)
            
            # Normalize to 0-1 (lower flow = higher consistency)
            consistency = 1.0 / (1.0 + avg_magnitude / 10.0)
            
            self.prev_frame = current_gray
            return float(np.clip(consistency, 0, 1))
        except Exception as e:
            logger.warning(f"Error calculating temporal consistency: {e}")
            return 0.5

    def _apply_temporal_weighting(self, current_confidence: float) -> float:
        """
        Apply temporal weighting to confidence scores.
        Smooths predictions across frames to reduce jitter.
        """
        if len(self.confidence_history) == 0:
            return current_confidence
        
        # Exponential weighting: recent frames have higher weight
        history_list = list(self.confidence_history)
        weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])  # Sum = 1.0
        
        if len(history_list) < len(weights):
            weights = weights[-len(history_list):]
            weights = weights / weights.sum()
        
        weighted_score = np.sum(np.array(history_list) * weights)
        return float(weighted_score)

    def _detect_temporal_anomaly(self, confidence: float, threshold: float = 0.3) -> bool:
        """
        Detect temporal anomalies in confidence scores.
        Large jumps in confidence may indicate deepfakes.
        """
        if len(self.confidence_history) < 2:
            return False
        
        prev_confidence = list(self.confidence_history)[-2]
        jump = abs(confidence - prev_confidence)
        
        return jump > threshold

    def _final_verdict(self, prediction: str, anomaly_ratio: float, 
                      confidence: float) -> str:
        """
        Determine final verdict considering multiple factors.
        """
        # High temporal anomaly ratio increases likelihood of fake
        if anomaly_ratio > 0.5 and confidence > 0.5:
            return "FAKE"
        
        # Standard prediction if low anomaly ratio
        if anomaly_ratio < 0.2:
            return prediction
        
        # Moderate anomaly ratio - slightly increase fake weight
        if prediction == "UNCERTAIN" and anomaly_ratio > 0.3:
            return "FAKE" if confidence > 0.4 else "UNCERTAIN"
        
        return prediction

    def _aggregate_predictions(self, confidences: List[float]) -> str:
        """
        Aggregate multiple confidence scores into a single prediction.
        Enhanced with better thresholds based on empirical testing.
        
        Args:
            confidences: List of confidence scores (0-1)
        
        Returns:
            Classification string: "REAL", "FAKE", or "UNCERTAIN"
        """
        if not confidences:
            return "UNKNOWN"
        
        avg_confidence = np.mean(confidences)
        
        # Use improved thresholds
        if avg_confidence < 0.35:
            return "REAL"
        elif avg_confidence > 0.65:
            return "FAKE"
        else:
            return "UNCERTAIN"

    def _get_color_for_prediction(self, prediction: str) -> tuple:
        """Get BGR color for prediction label."""
        colors = {
            "REAL": (0, 255, 0),         # Green
            "FAKE": (0, 0, 255),          # Red
            "UNCERTAIN": (0, 165, 255)   # Orange
        }
        return colors.get(prediction, (255, 255, 255))  # White default
