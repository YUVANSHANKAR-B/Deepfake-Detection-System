"""
Face Detection Module
Detects faces in images and video frames using multiple backends.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging
from config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detects faces in images and video frames.
    Supports multiple detection backends: MediaPipe, dlib, OpenCV.
    """

    def __init__(self, model_type: str = "mediapipe", confidence_threshold: float = 0.5):
        """
        Initialize face detector.
        
        Args:
            model_type: Type of face detector ("mediapipe", "dlib", "opencv")
            confidence_threshold: Confidence threshold for detections
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.detector = None
        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize the appropriate face detector based on model_type."""
        if self.model_type == "mediapipe":
            self._initialize_mediapipe()
        elif self.model_type == "opencv":
            self._initialize_opencv()
        elif self.model_type == "dlib":
            self._initialize_dlib()
        else:
            logger.warning(f"Unknown model type {self.model_type}, using MediaPipe")
            self._initialize_mediapipe()

    def _initialize_mediapipe(self):
        """Initialize MediaPipe FaceMesh for face detection."""
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.detector = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 1 for full-range, 0 for short-range
                min_detection_confidence=self.confidence_threshold
            )
            logger.info("MediaPipe face detector initialized")
        except (ImportError, AttributeError) as e:
            logger.warning(f"MediaPipe initialization failed: {e}")
            logger.info("Falling back to OpenCV face detector")
            self._initialize_opencv()
            self.model_type = "opencv"

    def _initialize_opencv(self):
        """Initialize OpenCV Haar Cascade for face detection."""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            if self.detector.empty():
                logger.error("Failed to load Haar Cascade classifier")
                self.detector = None
            else:
                logger.info("OpenCV Haar Cascade face detector initialized")
        except Exception as e:
            logger.error(f"Error initializing OpenCV detector: {e}")
            self.detector = None

    def _initialize_dlib(self):
        """Initialize dlib face detector."""
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            logger.info("dlib face detector initialized")
        except ImportError:
            logger.error("dlib not installed. Install with: pip install dlib")
            self.detector = None

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the image.
        
        Args:
            image: Input image as numpy array (BGR format for OpenCV)
        
        Returns:
            List of bounding boxes as (x, y, width, height)
        """
        if self.detector is None:
            logger.error("Detector not initialized")
            return []

        faces = []
        
        if self.model_type == "mediapipe":
            faces = self._detect_mediapipe(image)
        elif self.model_type == "opencv":
            faces = self._detect_opencv(image)
        elif self.model_type == "dlib":
            faces = self._detect_dlib(image)
        
        return faces

    def _detect_mediapipe(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe."""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb_image)
            faces = []
            
            if results.detections:
                h, w = image.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    faces.append((x, y, width, height))
            
            return faces
        except Exception as e:
            logger.error(f"Error in MediaPipe detection: {e}")
            return []

    def _detect_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV Haar Cascade."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return list(faces)
        except Exception as e:
            logger.error(f"Error in OpenCV detection: {e}")
            return []

    def _detect_dlib(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using dlib."""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dets = self.detector(rgb_image, 1)  # 1 is the upsampling parameter
            faces = []
            
            for det in dets:
                x = det.left()
                y = det.top()
                width = det.width()
                height = det.height()
                faces.append((x, y, width, height))
            
            return faces
        except Exception as e:
            logger.error(f"Error in dlib detection: {e}")
            return []

    def crop_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]],
                  padding: float = 0.1) -> List[np.ndarray]:
        """
        Crop face regions from image with optional padding.
        
        Args:
            image: Input image
            faces: List of face bounding boxes
            padding: Padding ratio (0-1) to expand crop area
        
        Returns:
            List of cropped face images
        """
        cropped_faces = []
        h, w = image.shape[:2]
        
        for x, y, face_w, face_h in faces:
            pad_x = int(face_w * padding)
            pad_y = int(face_h * padding)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + face_w + pad_x)
            y2 = min(h, y + face_h + pad_y)
            
            cropped = image[y1:y2, x1:x2]
            if cropped.size > 0:
                cropped_faces.append(cropped)
        
        return cropped_faces
