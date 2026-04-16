"""
Video processing utilities for deepfake detection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Utility class for video processing operations."""

    @staticmethod
    def get_video_properties(video_path: str) -> dict:
        """
        Get properties of a video file.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with video properties
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            properties = {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_seconds": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            }
            
            cap.release()
            return properties
        except Exception as e:
            logger.error(f"Error getting video properties: {e}")
            return {}

    @staticmethod
    def extract_frames(video_path: str, interval: int = 1, max_frames: int = None,
                      resize_width: int = None, resize_height: int = None) -> Generator:
        """
        Extract frames from video with optional interval and resizing.
        
        Args:
            video_path: Path to video file
            interval: Extract every Nth frame (1 = all frames)
            max_frames: Maximum number of frames to extract
            resize_width: Resize frame width (optional)
            resize_height: Resize frame height (optional)
        
        Yields:
            Tuples of (frame_index, frame_image)
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return
            
            frame_idx = 0
            extracted_count = 0
            
            while extracted_count < (max_frames or float('inf')):
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_idx % interval == 0:
                    if resize_width and resize_height:
                        frame = cv2.resize(frame, (resize_width, resize_height))
                    
                    yield frame_idx, frame
                    extracted_count += 1
                
                frame_idx += 1
            
            cap.release()
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")

    @staticmethod
    def save_video(output_path: str, frames: List[np.ndarray], fps: float = 30.0,
                  width: int = None, height: int = None) -> bool:
        """
        Save frames to video file.
        
        Args:
            output_path: Path to save video
            frames: List of frame images
            fps: Frames per second
            width: Video width (auto-detect if None)
            height: Video height (auto-detect if None)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not frames:
                logger.error("No frames to save")
                return False
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if width is None or height is None:
                height, width = frames[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error("Failed to create video writer")
                return False
            
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                out.write(frame)
            
            out.release()
            logger.info(f"Video saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving video: {e}")
            return False

    @staticmethod
    def get_frame_at_time(video_path: str, time_seconds: float) -> Optional[np.ndarray]:
        """
        Extract a specific frame at given time.
        
        Args:
            video_path: Path to video file
            time_seconds: Time in seconds
        
        Returns:
            Frame image or None if error
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_num = int(time_seconds * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            cap.release()
            
            return frame if ret else None
        except Exception as e:
            logger.error(f"Error getting frame at time {time_seconds}: {e}")
            return None

    @staticmethod
    def get_video_duration(video_path: str) -> float:
        """Get video duration in seconds."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            
            return frame_count / fps if fps > 0 else 0.0
        except Exception as e:
            logger.error(f"Error getting video duration: {e}")
            return 0.0

    @staticmethod
    def create_frame_montage(frames: List[np.ndarray], cols: int = 4,
                            rows: int = 2) -> np.ndarray:
        """
        Create a montage of frames in a grid.
        
        Args:
            frames: List of frame images
            cols: Number of columns in grid
            rows: Number of rows in grid
        
        Returns:
            Montage image
        """
        try:
            # Resize frames to uniform size
            target_height, target_width = frames[0].shape[:2]
            resized_frames = [cv2.resize(f, (target_width, target_height)) for f in frames[:cols*rows]]
            
            # Pad with black frames if needed
            while len(resized_frames) < cols * rows:
                resized_frames.append(np.zeros_like(resized_frames[0]))
            
            # Create rows
            row_images = []
            for r in range(rows):
                row_start = r * cols
                row_frames = resized_frames[row_start:row_start + cols]
                row_image = np.hstack(row_frames)
                row_images.append(row_image)
            
            # Stack rows
            montage = np.vstack(row_images)
            return montage
        except Exception as e:
            logger.error(f"Error creating frame montage: {e}")
            return None
