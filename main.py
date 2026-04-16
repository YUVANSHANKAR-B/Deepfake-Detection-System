"""
Main entry point for the deepfake detection system.
Provides both CLI and programmatic interfaces.
"""

import sys
import argparse
import logging
from pathlib import Path

from pipeline import DeepfakeDetectionPipeline
from utils.logging_utils import setup_logger
from api.app import create_app
from config import API_CONFIG

logger = setup_logger(__name__)


def analyze_image_cli(image_path: str):
    """Analyze image from command line."""
    pipeline = DeepfakeDetectionPipeline()
    results = pipeline.analyze_image(image_path)
    
    print("\n" + "="*50)
    print("IMAGE ANALYSIS RESULTS")
    print("="*50)
    print(f"Image: {image_path}")
    print(f"Faces Detected: {results.get('faces_detected', 0)}")
    print(f"Overall Confidence: {results.get('confidence', 0):.4f}")
    print(f"Overall Prediction: {results.get('prediction', 'UNKNOWN')}")
    print(f"Processing Time: {results.get('processing_time', 0):.2f}s")
    
    if 'face_predictions' in results:
        print("\nFace-by-Face Results:")
        for face in results['face_predictions']:
            print(f"  Face {face['face_id']}: {face['prediction']} (confidence: {face['confidence']:.4f})")
    
    print(f"\nOutput Image: {results.get('output_path', 'N/A')}")
    print("="*50 + "\n")
    
    return results


def analyze_video_cli(video_path: str):
    """Analyze video from command line."""
    pipeline = DeepfakeDetectionPipeline()
    results = pipeline.analyze_video(video_path)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return results
    
    print("\n" + "="*50)
    print("VIDEO ANALYSIS RESULTS")
    print("="*50)
    print(f"Video: {video_path}")
    print(f"Duration: {results['video_properties']['duration_seconds']:.2f}s")
    print(f"Resolution: {results['video_properties']['width']}x{results['video_properties']['height']}")
    print(f"FPS: {results['video_properties']['fps']:.2f}")
    print(f"\nFrames Analyzed: {results['analyzed_frames']}/{results['total_frames']}")
    print(f"Frames With Faces: {results['frames_with_faces']}")
    print(f"Fake Frames: {results['fake_frames']}")
    print(f"Fake Frame Ratio: {results['fake_frame_ratio']:.2%}")
    print(f"\nOverall Confidence: {results['overall_confidence']:.4f}")
    print(f"Overall Prediction: {results['overall_prediction']}")
    print(f"Processing Time: {results['processing_time']:.2f}s")
    print("="*50 + "\n")
    
    return results


def run_api_server():
    """Run the Flask API server."""
    logger.info(f"Starting API server on {API_CONFIG['host']}:{API_CONFIG['port']}")
    print(f"\n{'='*50}")
    print("DEEPFAKE DETECTION API SERVER")
    print(f"{'='*50}")
    print(f"Server running at: http://{API_CONFIG['host']}:{API_CONFIG['port']}")
    print("\nAvailable Endpoints:")
    print("  POST /api/analyze-image      - Analyze single image")
    print("  POST /api/analyze-video      - Analyze single video")
    print("  POST /api/batch-analyze      - Batch analyze multiple files")
    print("  GET  /api/model-info         - Get model information")
    print("  GET  /api/health             - Health check")
    print(f"{'='*50}\n")
    
    app = create_app()
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug']
    )


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Deepfake Detection System - Analyze images and videos for manipulated content',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --image path/to/image.jpg
  python main.py --video path/to/video.mp4
  python main.py --batch path/to/folder
  python main.py --api
        """
    )
    
    parser.add_argument('--image', type=str, help='Analyze single image')
    parser.add_argument('--video', type=str, help='Analyze single video')
    parser.add_argument('--batch', type=str, help='Batch analyze all media in folder')
    parser.add_argument('--api', action='store_true', help='Start API server')
    parser.add_argument('--loglevel', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.loglevel))
    
    if args.image:
        if not Path(args.image).exists():
            print(f"Error: Image file not found: {args.image}")
            sys.exit(1)
        analyze_image_cli(args.image)
    
    elif args.video:
        if not Path(args.video).exists():
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        analyze_video_cli(args.video)
    
    elif args.batch:
        if not Path(args.batch).exists():
            print(f"Error: Folder not found: {args.batch}")
            sys.exit(1)
        
        print(f"Batch analyzing all media files in: {args.batch}")
        folder = Path(args.batch)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        
        files_found = False
        for file in folder.iterdir():
            if file.suffix.lower() in image_extensions:
                files_found = True
                analyze_image_cli(str(file))
            elif file.suffix.lower() in video_extensions:
                files_found = True
                analyze_video_cli(str(file))
        
        if not files_found:
            print("No supported media files found in folder")
    
    elif args.api:
        run_api_server()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
