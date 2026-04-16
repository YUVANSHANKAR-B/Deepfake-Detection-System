"""
Utility functions for the deepfake detection system.
"""

from .image_utils import ImageProcessor
from .video_utils import VideoProcessor
from .logging_utils import setup_logger

__all__ = ["ImageProcessor", "VideoProcessor", "setup_logger"]
