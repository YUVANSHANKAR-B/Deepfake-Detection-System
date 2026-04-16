"""
Configuration settings for the Deepfake Detection System.
"""
import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.absolute()

# Model Configuration
MODEL_CONFIG = {
    "face_detector": {
        "model_type": "mediapipe",  # Options: mediapipe, dlib, opencv
        "confidence_threshold": 0.5,
    },
    "deepfake_detector": {
        "model_type": "ensemble",  # Options: meso, xception, efficientnet, ensemble
        "confidence_threshold": 0.5,
        "use_gpu": True,
    }
}

# Video Processing Configuration
VIDEO_CONFIG = {
    "frame_extraction_interval": 5,  # Extract every Nth frame
    "max_frames_to_process": 100,  # Maximum frames to analyze per video
    "resize_width": 256,
    "resize_height": 256,
}

# Image Processing Configuration
IMAGE_CONFIG = {
    "resize_width": 256,
    "resize_height": 256,
    "normalize": True,
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
    "max_content_length": 100 * 1024 * 1024,  # 100 MB max file size
}

# Model Paths
MODEL_PATHS = {
    "face_detector": PROJECT_ROOT / "models" / "face_detector",
    "deepfake_detector": PROJECT_ROOT / "models" / "deepfake_detector",
}

# Data Paths
DATA_PATHS = {
    "input_data": PROJECT_ROOT / "data" / "input",
    "output_data": PROJECT_ROOT / "data" / "output",
    "temp": PROJECT_ROOT / "data" / "temp",
}

# Ensure directories exist
for path in DATA_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# Detection Thresholds - OPTIMIZED
DETECTION_THRESHOLDS = {
    "real": 0.35,        # Below this: likely real (improved from 0.3)
    "fake": 0.65,        # Above this: likely fake (improved from 0.7)
    "uncertain": (0.35, 0.65),  # Between these: uncertain
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": PROJECT_ROOT / "logs" / "deepfake_detection.log",
}

# Ensure logs directory exists
LOGGING_CONFIG["log_file"].parent.mkdir(parents=True, exist_ok=True)
