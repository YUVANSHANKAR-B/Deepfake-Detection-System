"""
Example usage of the deepfake detection system.
"""

from pipeline import DeepfakeDetectionPipeline
from utils.logging_utils import setup_logger
import logging

logger = setup_logger(__name__, level="INFO")


def example_image_analysis():
    """Example: Analyze a single image."""
    print("\n" + "="*60)
    print("EXAMPLE: Image Analysis")
    print("="*60)
    
    # Initialize pipeline
    pipeline = DeepfakeDetectionPipeline(
        face_detector_type="mediapipe",
        deepfake_detector_type="ensemble"
    )
    
    # Analyze image (replace with actual image path)
    image_path = "data/input/sample_image.jpg"
    results = pipeline.analyze_image(image_path)
    
    print(f"\nResults for {image_path}:")
    print(f"  Faces Detected: {results.get('faces_detected', 0)}")
    print(f"  Confidence: {results.get('confidence', 0):.4f}")
    print(f"  Prediction: {results.get('prediction', 'UNKNOWN')}")


def example_video_analysis():
    """Example: Analyze a video file."""
    print("\n" + "="*60)
    print("EXAMPLE: Video Analysis")
    print("="*60)
    
    # Initialize pipeline
    pipeline = DeepfakeDetectionPipeline(
        face_detector_type="mediapipe",
        deepfake_detector_type="ensemble"
    )
    
    # Analyze video (replace with actual video path)
    video_path = "data/input/sample_video.mp4"
    results = pipeline.analyze_video(video_path)
    
    if 'error' not in results:
        print(f"\nResults for {video_path}:")
        print(f"  Total Frames: {results.get('total_frames', 0)}")
        print(f"  Analyzed Frames: {results.get('analyzed_frames', 0)}")
        print(f"  Frames with Faces: {results.get('frames_with_faces', 0)}")
        print(f"  Fake Frames: {results.get('fake_frames', 0)}")
        print(f"  Fake Frame Ratio: {results.get('fake_frame_ratio', 0):.2%}")
        print(f"  Overall Prediction: {results.get('overall_prediction', 'UNKNOWN')}")


def example_batch_analysis():
    """Example: Batch analyze multiple files."""
    print("\n" + "="*60)
    print("EXAMPLE: Batch Analysis")
    print("="*60)
    
    from pathlib import Path
    
    # Initialize pipeline
    pipeline = DeepfakeDetectionPipeline()
    
    # Analyze all files in a folder
    folder = Path("data/input")
    image_extensions = {'.jpg', '.jpeg', '.png'}
    video_extensions = {'.mp4', '.avi', '.mov'}
    
    results_summary = {
        "total_files": 0,
        "analyzed_images": 0,
        "analyzed_videos": 0,
        "fake_detections": 0
    }
    
    for file in folder.iterdir():
        if file.suffix.lower() in image_extensions:
            results = pipeline.analyze_image(str(file))
            results_summary["analyzed_images"] += 1
            if results.get("prediction") == "FAKE":
                results_summary["fake_detections"] += 1
        
        elif file.suffix.lower() in video_extensions:
            results = pipeline.analyze_video(str(file))
            results_summary["analyzed_videos"] += 1
            if results.get("overall_prediction") == "FAKE":
                results_summary["fake_detections"] += 1
    
    print(f"\nBatch Analysis Summary:")
    print(f"  Total Images: {results_summary['analyzed_images']}")
    print(f"  Total Videos: {results_summary['analyzed_videos']}")
    print(f"  Fake Detections: {results_summary['fake_detections']}")


def example_custom_configuration():
    """Example: Use custom configuration."""
    print("\n" + "="*60)
    print("EXAMPLE: Custom Configuration")
    print("="*60)
    
    # Use different model combinations
    configs = [
        ("opencv", "xception"),
        ("mediapipe", "meso"),
        ("dlib", "efficientnet"),
    ]
    
    for face_model, deepfake_model in configs:
        print(f"\nTesting: {face_model} + {deepfake_model}")
        try:
            pipeline = DeepfakeDetectionPipeline(
                face_detector_type=face_model,
                deepfake_detector_type=deepfake_model
            )
            print(f"  ✓ Successfully initialized")
        except Exception as e:
            print(f"  ✗ Error: {e}")


if __name__ == "__main__":
    print("Deepfake Detection System - Usage Examples")
    print("==========================================\n")
    
    print("Note: Replace 'data/input/sample_*' paths with actual media files")
    print("      to run these examples.\n")
    
    # Uncomment the examples you want to run:
    # example_image_analysis()
    # example_video_analysis()
    # example_batch_analysis()
    # example_custom_configuration()
    
    print("\nTo use this system:")
    print("1. Command Line: python main.py --image <path>")
    print("2. Command Line: python main.py --video <path>")
    print("3. API Server:   python main.py --api")
    print("4. Programmatic: See examples above\n")
