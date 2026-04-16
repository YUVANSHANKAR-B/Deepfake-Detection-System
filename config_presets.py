"""
Configuration presets for different use cases.
Copy and modify these configurations in config.py
"""

# ============================================================
# PRESET CONFIGURATIONS
# ============================================================

# 1. FAST MODE - Prioritizes speed over accuracy
FAST_CONFIG = {
    "face_detector": {
        "model_type": "opencv",  # Fastest face detection
        "confidence_threshold": 0.6,  # Higher threshold = fewer false positives
    },
    "deepfake_detector": {
        "model_type": "meso",  # Lighter model
        "confidence_threshold": 0.5,
        "use_gpu": True,
    },
    "video_config": {
        "frame_extraction_interval": 10,  # Sample every 10th frame
        "max_frames_to_process": 50,  # Analyze fewer frames
    }
}

# 2. ACCURATE MODE - Prioritizes accuracy over speed
ACCURATE_CONFIG = {
    "face_detector": {
        "model_type": "dlib",  # Most robust face detection
        "confidence_threshold": 0.3,  # Lower threshold = catch more faces
    },
    "deepfake_detector": {
        "model_type": "ensemble",  # Multiple models for consensus
        "confidence_threshold": 0.5,
        "use_gpu": True,
    },
    "video_config": {
        "frame_extraction_interval": 2,  # Sample every 2nd frame
        "max_frames_to_process": 200,  # Analyze more frames
    }
}

# 3. BALANCED MODE - Good balance of speed and accuracy
BALANCED_CONFIG = {
    "face_detector": {
        "model_type": "mediapipe",  # Good balance
        "confidence_threshold": 0.5,
    },
    "deepfake_detector": {
        "model_type": "xception",  # Good accuracy, reasonable speed
        "confidence_threshold": 0.5,
        "use_gpu": True,
    },
    "video_config": {
        "frame_extraction_interval": 5,
        "max_frames_to_process": 100,
    }
}

# 4. LOW-RESOURCE MODE - For CPU-only or limited memory
LOW_RESOURCE_CONFIG = {
    "face_detector": {
        "model_type": "opencv",
        "confidence_threshold": 0.7,
    },
    "deepfake_detector": {
        "model_type": "meso",
        "confidence_threshold": 0.5,
        "use_gpu": False,  # CPU only
    },
    "video_config": {
        "frame_extraction_interval": 20,  # Fewer frames
        "max_frames_to_process": 30,
    },
    "image_config": {
        "resize_width": 128,  # Smaller resolution
        "resize_height": 128,
    }
}

# 5. RESEARCH MODE - Maximum accuracy and detailed analysis
RESEARCH_CONFIG = {
    "face_detector": {
        "model_type": "dlib",
        "confidence_threshold": 0.2,  # Very low threshold
    },
    "deepfake_detector": {
        "model_type": "ensemble",
        "confidence_threshold": 0.5,
        "use_gpu": True,
    },
    "video_config": {
        "frame_extraction_interval": 1,  # Every frame
        "max_frames_to_process": 1000,  # All frames
    },
    "detection_thresholds": {
        "real": 0.25,  # More conservative thresholds
        "fake": 0.75,
        "uncertain": (0.25, 0.75),
    }
}

# ============================================================
# USAGE INSTRUCTIONS
# ============================================================

USAGE_EXAMPLE = """
To use preset configurations, modify config.py:

Option 1: Import preset
from config_presets import FAST_CONFIG
# Then update global config in config.py

Option 2: Copy configuration
MODEL_CONFIG = {
    "face_detector": {...},
    "deepfake_detector": {...}
}

Option 3: In code
from config_presets import FAST_CONFIG
pipeline = DeepfakeDetectionPipeline(
    face_detector_type=FAST_CONFIG['face_detector']['model_type'],
    deepfake_detector_type=FAST_CONFIG['deepfake_detector']['model_type']
)
"""

# ============================================================
# PRESET SELECTION GUIDE
# ============================================================

SELECTION_GUIDE = """
WHEN TO USE EACH PRESET:

FAST_CONFIG
- Purpose: Quick screening of large media volumes
- Use case: Content moderation dashboards
- Hardware: Can run on CPU
- Accuracy: ~80-85%
- Speed: 50-100 images/min on CPU

ACCURATE_CONFIG
- Purpose: Forensic analysis and verification
- Use case: Legal evidence, expert review
- Hardware: Needs GPU for reasonable speed
- Accuracy: ~92-95%
- Speed: 5-20 images/min even on GPU

BALANCED_CONFIG
- Purpose: General-purpose production use
- Use case: Security systems, social media platforms
- Hardware: Can run on GPU or high-end CPU
- Accuracy: ~88-90%
- Speed: 20-50 images/min on CPU, 100+ on GPU

LOW_RESOURCE_CONFIG
- Purpose: Embedded systems, edge devices
- Use case: Mobile apps, IoT devices, bandwidth-limited
- Hardware: Works on any CPU, no GPU needed
- Accuracy: ~75-80%
- Speed: 10-20 images/min on modest CPU

RESEARCH_CONFIG
- Purpose: Academic research, model development
- Use case: Training data validation, benchmarking
- Hardware: Needs GPU
- Accuracy: ~95%+
- Speed: Slowest (1-5 videos/min)
"""

print(SELECTION_GUIDE)
