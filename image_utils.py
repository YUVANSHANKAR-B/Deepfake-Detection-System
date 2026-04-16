"""
Image processing utilities for deepfake detection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Utility class for image processing operations."""

    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Image as numpy array (BGR format) or None if error
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> bool:
        """
        Save image to file.
        
        Args:
            image: Image as numpy array
            output_path: Path to save image
        
        Returns:
            True if successful, False otherwise
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(output_path), image)
            if success:
                logger.info(f"Image saved to {output_path}")
            return success
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {e}")
            return False

    @staticmethod
    def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Resize image to specified dimensions.
        
        Args:
            image: Input image
            width: Target width
            height: Target height
        
        Returns:
            Resized image
        """
        try:
            return cv2.resize(image, (width, height))
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image

    @staticmethod
    def normalize_image(image: np.ndarray, mean: List[float] = None,
                       std: List[float] = None) -> np.ndarray:
        """
        Normalize image using mean and standard deviation.
        
        Args:
            image: Input image (values 0-255)
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        
        Returns:
            Normalized image
        """
        try:
            # Convert to float and normalize to 0-1
            normalized = image.astype(np.float32) / 255.0
            
            if mean is not None and std is not None:
                # Apply standardization
                mean = np.array(mean, dtype=np.float32)
                std = np.array(std, dtype=np.float32)
                normalized = (normalized - mean) / std
            
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing image: {e}")
            return image.astype(np.float32) / 255.0

    @staticmethod
    def draw_bounding_boxes(image: np.ndarray, boxes: List[Tuple[int, int, int, int]],
                           labels: List[str] = None, colors: List[Tuple] = None) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image
            boxes: List of bounding boxes (x, y, width, height)
            labels: List of labels for each box
            colors: List of BGR colors for each box
        
        Returns:
            Image with drawn boxes
        """
        try:
            result = image.copy()
            
            if colors is None:
                colors = [(0, 255, 0)] * len(boxes)
            
            for idx, (x, y, w, h) in enumerate(boxes):
                color = colors[idx] if idx < len(colors) else (0, 255, 0)
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
                
                if labels is not None and idx < len(labels):
                    cv2.putText(result, labels[idx], (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return result
        except Exception as e:
            logger.error(f"Error drawing bounding boxes: {e}")
            return image

    @staticmethod
    def apply_face_mask(image: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Apply slight blur/mask to face regions for privacy.
        
        Args:
            image: Input image
            alpha: Transparency of blur
        
        Returns:
            Image with privacy mask applied
        """
        try:
            blurred = cv2.GaussianBlur(image, (51, 51), 0)
            return cv2.addWeighted(image, 1 - alpha, blurred, alpha, 0)
        except Exception as e:
            logger.error(f"Error applying face mask: {e}")
            return image

    @staticmethod
    def convert_color_space(image: np.ndarray, from_space: str = "BGR",
                           to_space: str = "RGB") -> np.ndarray:
        """
        Convert image between color spaces.
        
        Args:
            image: Input image
            from_space: Source color space
            to_space: Target color space
        
        Returns:
            Image in target color space
        """
        try:
            if from_space == "BGR" and to_space == "RGB":
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif from_space == "RGB" and to_space == "BGR":
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif from_space == "BGR" and to_space == "HSV":
                return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif from_space == "BGR" and to_space == "GRAY":
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                logger.warning(f"Unsupported color space conversion: {from_space} -> {to_space}")
                return image
        except Exception as e:
            logger.error(f"Error converting color space: {e}")
            return image

    @staticmethod
    def get_image_info(image: np.ndarray) -> dict:
        """Get metadata about an image."""
        return {
            "height": image.shape[0],
            "width": image.shape[1],
            "channels": image.shape[2] if len(image.shape) > 2 else 1,
            "dtype": str(image.dtype),
            "size_mb": image.nbytes / (1024 * 1024)
        }
