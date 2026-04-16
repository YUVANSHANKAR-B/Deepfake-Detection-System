# Deepfake Detection System

A comprehensive deepfake detection system that identifies manipulated or synthetic media content in both images and videos.

## Features

✨ **Multi-Modal Detection**
- Image analysis for deepfake detection
- Video frame analysis with temporal consistency
- Face detection and tracking

🔍 **Advanced Detection Methods**
- Ensemble approach combining multiple models
- Xception architecture for binary classification
- MesoNet specialized for deepfake detection
- EfficientNet for efficient inference

🎯 **Face Detection Backends**
- MediaPipe for real-time face detection
- OpenCV Haar Cascade for quick detection
- dlib for robust face detection

📊 **Analysis Features**
- Per-face confidence scores
- Frame-by-frame video analysis
- Fake frame ratio calculation
- Uncertainty classification for borderline cases

🌐 **RESTful API**
- Single image/video analysis endpoints
- Batch processing capabilities
- Model information endpoints
- Health check endpoint

## Project Structure

```
deepfake-detection/
├── config.py                 # Configuration management
├── main.py                   # CLI entry point
├── pipeline.py               # Main detection pipeline
├── requirements.txt          # Python dependencies
│
├── models/
│   ├── __init__.py
│   ├── face_detector.py      # Face detection module
│   └── deepfake_detector.py  # Deepfake classification
│
├── utils/
│   ├── __init__.py
│   ├── image_utils.py        # Image processing utilities
│   ├── video_utils.py        # Video processing utilities
│   └── logging_utils.py      # Logging configuration
│
├── api/
│   ├── __init__.py
│   └── app.py                # Flask API application
│
├── data/
│   ├── input/                # Input media files
│   ├── output/               # Detection results
│   └── temp/                 # Temporary files
│
└── tests/                    # Unit tests
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda
- (Optional) CUDA for GPU acceleration

### Setup Instructions

1. **Clone/Extract the project:**
```bash
cd "Deepfake AI PJ"
```

2. **Create virtual environment (recommended):**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n deepfake python=3.9
conda activate deepfake
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **For GPU support (optional):**
```bash
# PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Command Line Interface

#### Analyze Single Image
```bash
python main.py --image path/to/image.jpg
```

#### Analyze Single Video
```bash
python main.py --video path/to/video.mp4
```

#### Batch Analysis
```bash
python main.py --batch path/to/media/folder
```

#### Start API Server
```bash
python main.py --api
```

### API Usage

#### Start the server:
```bash
python main.py --api
```

The API will run on `http://localhost:5000`

#### Endpoints:

**Analyze Image:**
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/analyze-image
```

**Analyze Video:**
```bash
curl -X POST -F "file=@video.mp4" http://localhost:5000/api/analyze-video
```

**Batch Analysis:**
```bash
curl -X POST -F "files=@image1.jpg" -F "files=@video.mp4" http://localhost:5000/api/batch-analyze
```

**Get Model Info:**
```bash
curl http://localhost:5000/api/model-info
```

**Health Check:**
```bash
curl http://localhost:5000/api/health
```

### Python API

```python
from pipeline import DeepfakeDetectionPipeline

# Initialize pipeline
pipeline = DeepfakeDetectionPipeline(
    face_detector_type="mediapipe",
    deepfake_detector_type="ensemble"
)

# Analyze image
results = pipeline.analyze_image("path/to/image.jpg")
print(f"Prediction: {results['prediction']}")
print(f"Confidence: {results['confidence']:.4f}")

# Analyze video
results = pipeline.analyze_video("path/to/video.mp4")
print(f"Overall Prediction: {results['overall_prediction']}")
print(f"Fake Frame Ratio: {results['fake_frame_ratio']:.2%}")
```

## Configuration

Edit `config.py` to customize:

- **Model Types**: Choose between different face detectors and deepfake classifiers
- **Detection Thresholds**: Adjust confidence thresholds for classification
- **Video Settings**: Frame extraction interval, max frames to process
- **API Settings**: Host, port, max file size
- **Logging**: Log level and output format

## Results Interpretation

### Prediction Classes:
- **REAL**: Confidence < 0.3 - Likely authentic content
- **FAKE**: Confidence > 0.7 - Likely manipulated content
- **UNCERTAIN**: Confidence 0.3-0.7 - Borderline case requiring manual review

### Example Output:

**Image Analysis:**
```json
{
  "status": "success",
  "image_path": "test.jpg",
  "faces_detected": 2,
  "confidence": 0.8234,
  "prediction": "FAKE",
  "face_predictions": [
    {"face_id": 0, "confidence": 0.91, "prediction": "FAKE"},
    {"face_id": 1, "confidence": 0.74, "prediction": "FAKE"}
  ],
  "processing_time": 2.34
}
```

**Video Analysis:**
```json
{
  "status": "success",
  "video_path": "test.mp4",
  "video_properties": {
    "fps": 30.0,
    "frame_count": 300,
    "duration_seconds": 10.0
  },
  "analyzed_frames": 60,
  "frames_with_faces": 58,
  "fake_frames": 45,
  "fake_frame_ratio": 0.775,
  "overall_prediction": "FAKE"
}
```

## Advanced Features

### Custom Model Loading
```python
from models import DeepfakeDetector

# Load specific model
detector = DeepfakeDetector(
    model_type="xception",  # or "meso", "efficientnet"
    confidence_threshold=0.5,
    use_gpu=True
)
```

### Face Detection Options
```python
from models import FaceDetector

# Try different detectors
detector = FaceDetector(
    model_type="mediapipe",  # Fast, real-time
    # or "opencv" for quick detection
    # or "dlib" for robust detection
    confidence_threshold=0.5
)
```

### Batch Processing
```python
pipeline = DeepfakeDetectionPipeline()

# Process multiple files
for file in media_files:
    if file.suffix in {'.jpg', '.png'}:
        results = pipeline.analyze_image(str(file))
    else:
        results = pipeline.analyze_video(str(file))
```

## Performance Optimization

- **GPU Acceleration**: Use CUDA for faster inference (requires GPU)
- **Frame Sampling**: Adjust `frame_extraction_interval` in config for faster video processing
- **Model Selection**: Choose lighter models (MesoNet) for speed vs accuracy
- **Batch Processing**: Process multiple files simultaneously using the batch API

## Limitations

- Accuracy depends on training data quality and diversity
- May struggle with heavily compressed videos
- Ensemble methods require more computational resources
- Performance varies with different face angles and lighting conditions

## Future Enhancements

- [ ] Temporal consistency analysis across frames
- [ ] Emotion and facial expression analysis
- [ ] Audio deepfake detection
- [ ] Fine-tuned models for specific deepfake techniques
- [ ] Web UI dashboard for visualizing results
- [ ] Model training interface for custom datasets
- [ ] Real-time stream analysis
- [ ] Explainability features (attention maps, etc.)

## Requirements

- Python 3.8+
- 4GB RAM minimum (8GB+ recommended)
- GPU recommended for real-time processing

## License

This project is provided for research and educational purposes.

## Support & Contributing

For issues, feature requests, or contributions, please refer to the project documentation.

## References

- FaceForensics++: Learning to Detect Manipulated Facial Images
- MesoNet: a Compact Facial Video Forgery Detection Network
- Xception: Deep Learning with Depthwise Separable Convolutions
- MediaPipe Face Detection: On-Device, Real-time Face Detection

## Disclaimer

This tool is intended for legitimate security research and verification purposes. Users are responsible for ensuring they have proper authorization before analyzing any media content.
