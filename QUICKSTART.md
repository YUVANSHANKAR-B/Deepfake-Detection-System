"""
Quick start guide for deepfake detection system.
"""

QUICK_START = """
# Quick Start Guide - Deepfake Detection System

## 5-Minute Setup

### 1. Install Python Dependencies (2 min)
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Media Files (1 min)
- Place images in: `data/input/`
- Place videos in: `data/input/`

### 3. Run Detection (2 min)

#### Option A: Analyze Single Image
```bash
python main.py --image data/input/my_image.jpg
```

#### Option B: Analyze Single Video
```bash
python main.py --video data/input/my_video.mp4
```

#### Option C: Analyze All Files in Folder
```bash
python main.py --batch data/input/
```

#### Option D: Start REST API Server
```bash
python main.py --api
# Then access at http://localhost:5000
```

## Understanding Results

### For Images:
```
Prediction: REAL/FAKE/UNCERTAIN
Confidence: 0.0 to 1.0
  0.0-0.3 = REAL (likely authentic)
  0.3-0.7 = UNCERTAIN (manual review needed)
  0.7-1.0 = FAKE (likely manipulated)
```

### For Videos:
```
Fake Frame Ratio: percentage of frames classified as fake
Overall Prediction: REAL/FAKE based on majority
Fake Frames: absolute count of suspicious frames
```

## API Quick Test

### Analyze Image via API:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/analyze-image
```

### Python Example:
```python
from pipeline import DeepfakeDetectionPipeline

pipeline = DeepfakeDetectionPipeline()
results = pipeline.analyze_image("test.jpg")
print(f"Prediction: {results['prediction']}")
print(f"Confidence: {results['confidence']:.2%}")
```

## Common Issues

**Issue: "Module not found" error**
Solution: 
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Issue: CUDA/GPU not found**
Solution: CPU mode is fine for testing. For faster processing, install GPU drivers.

**Issue: "No faces detected"**
Solution: Detected as authentic (no face to manipulate)

## Next Steps

1. Read full [README.md](README.md)
2. Check [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
3. Review [examples.py](examples.py)
4. Explore [config.py](config.py) for customization

## Performance Tips

- Use GPU for 3-5x faster processing
- Reduce video frame extraction interval for quicker analysis
- Use batch API for multiple files
- Lighter models (MesoNet) vs ensemble for speed/accuracy tradeoff

## Support

For detailed troubleshooting, see the full README.md in the project root.
"""

if __name__ == "__main__":
    print(QUICK_START)
