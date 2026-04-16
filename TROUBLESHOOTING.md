"""
Comprehensive Troubleshooting Guide
"""

TROUBLESHOOTING_GUIDE = """
# Troubleshooting Guide - Deepfake Detection System

## Installation Issues

### Problem: "ModuleNotFoundError: No module named 'torch'"
**Cause**: PyTorch not installed
**Solution**:
```bash
pip install torch torchvision
# Or for GPU support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problem: "ModuleNotFoundError: No module named 'mediapipe'"
**Cause**: MediaPipe not installed
**Solution**:
```bash
pip install mediapipe
```

### Problem: "ModuleNotFoundError: No module named 'cv2'"
**Cause**: OpenCV not installed
**Solution**:
```bash
pip install opencv-python opencv-contrib-python
```

### Problem: "Failed to import dlib"
**Cause**: dlib C++ compilation issue or not installed
**Solution**:
```bash
pip install dlib
# If that fails, install build tools first:
# Windows: Install Visual Studio Build Tools
# macOS: brew install cmake
# Linux: sudo apt-get install cmake
```

### Problem: Version Conflicts
**Cause**: Incompatible package versions
**Solution**:
```bash
pip install --upgrade -r requirements.txt
# Or create fresh virtual environment:
python -m venv venv_fresh
source venv_fresh/bin/activate
pip install -r requirements.txt
```

## Runtime Issues

### Problem: "CUDA out of memory"
**Cause**: GPU memory exhausted
**Solutions**:
1. Reduce batch size in config.py
2. Process shorter videos
3. Use CPU mode (set use_gpu=False)
4. Close other GPU applications
```python
# In config.py
VIDEO_CONFIG = {
    "frame_extraction_interval": 10,  # Increase to skip more frames
    "max_frames_to_process": 50,      # Reduce total frames
}
```

### Problem: "No module 'torch' with CUDA support"
**Cause**: PyTorch installed without GPU support
**Solution**:
```bash
pip uninstall torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

### Problem: Slow Performance on CPU
**Cause**: Single-threaded processing
**Solutions**:
1. Use GPU acceleration
2. Use Fast configuration preset
3. Use OpenCV face detector instead of dlib
4. Reduce video resolution
```python
from config_presets import FAST_CONFIG
# Use FAST_CONFIG in your pipeline
```

### Problem: "API server fails to start" on port 5000
**Cause**: Port already in use
**Solution**:
```bash
# Windows: Find and kill process using port 5000
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux:
lsof -ti:5000 | xargs kill -9

# Or change port in config.py:
API_CONFIG = {"port": 8000}
```

### Problem: "Address already in use"
**Cause**: Previous API process still running
**Solution**:
```bash
# Check running Python processes
ps aux | grep python

# Kill the process
kill -9 <PID>

# Or use different port
python main.py --api --port 8000
```

## Image Analysis Issues

### Problem: "No faces detected" for valid image
**Causes & Solutions**:

1. **Low image quality**
   ```python
   # Check image quality
   from utils.evaluation_utils import DataQualityChecker
   quality = DataQualityChecker.check_image_quality(image)
   ```

2. **Face too small**
   - Resize image larger or crop to face

3. **Unusual lighting**
   - Image too dark or too bright
   - Adjust brightness/contrast first

4. **Face angle too extreme**
   - Side/back view might not be detected
   - Try different detector (dlib more robust)

5. **Wrong face detector**
   ```python
   # Try different detector
   from models import FaceDetector
   detector = FaceDetector(model_type="dlib")  # More robust
   ```

### Problem: "Failed to load image"
**Cause**: File path or format issue
**Solution**:
```bash
# Verify file exists
ls -la /path/to/image.jpg

# Check file format
file /path/to/image.jpg

# Verify it's a valid image
python -c "from PIL import Image; Image.open('image.jpg').verify()"
```

### Problem: "FAKE prediction on clearly real image"
**Cause**: False positive
**Solutions**:
1. Adjust confidence threshold
2. Try different model
3. Check image compression level
```python
# Lower threshold for fewer false positives
MODEL_CONFIG["deepfake_detector"]["confidence_threshold"] = 0.7
```

## Video Analysis Issues

### Problem: "No frames extracted from video"
**Cause**: Codec or format issue
**Solution**:
```bash
# Check video format
ffprobe video.mp4

# Convert to MP4 if needed
ffmpeg -i input.avi -codec:v libx264 output.mp4
```

### Problem: "Video processing very slow"
**Cause**: Processing too many frames
**Solution**:
```python
# Increase frame extraction interval
VIDEO_CONFIG["frame_extraction_interval"] = 20  # Process every 20th frame

# Reduce max frames
VIDEO_CONFIG["max_frames_to_process"] = 50  # Analyze only 50 frames
```

### Problem: "Memory error on long video"
**Cause**: Too many frames in memory
**Solution**:
```python
# Reduce max frames
VIDEO_CONFIG["max_frames_to_process"] = 50

# Use lower resolution
VIDEO_CONFIG["resize_width"] = 256
VIDEO_CONFIG["resize_height"] = 256
```

### Problem: "Inconsistent predictions across video"
**Cause**: Temporal variability or quality changes
**Solution**:
1. Increase temporal window analysis
2. Check for compression artifacts
3. Use ensemble for more robust predictions

## API Issues

### Problem: "405 Method Not Allowed"
**Cause**: Using wrong HTTP method
**Solution**:
```bash
# Correct
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/analyze-image

# Wrong (using GET instead of POST)
curl http://localhost:5000/api/analyze-image
```

### Problem: "413 Payload Too Large"
**Cause**: File exceeds 100MB limit
**Solution**:
1. Compress/trim file
2. Change limit in config.py:
```python
API_CONFIG["max_content_length"] = 200 * 1024 * 1024  # 200MB
```

### Problem: CORS errors in web browser
**Cause**: Cross-origin request blocked
**Solution**:
```python
# In api/app.py, add CORS support
from flask_cors import CORS
CORS(app)
```

### Problem: "No response from API"
**Cause**: Server not running or crashed
**Solution**:
```bash
# Check if server is running
curl http://localhost:5000/api/health

# Start with debug mode to see errors
python main.py --api --debug

# Check logs
tail -f logs/deepfake_detection.log
```

## Configuration Issues

### Problem: Settings not taking effect
**Cause**: Module imported before changes
**Solution**:
```python
# Restart Python REPL or script
# Or reload module
import importlib
import config
importlib.reload(config)
```

### Problem: Using wrong configuration preset
**Solution**:
```python
# Check which preset you're using
from config_presets import FAST_CONFIG, BALANCED_CONFIG
print(FAST_CONFIG)
print(BALANCED_CONFIG)

# Apply correct preset to your config.py
```

## Docker Issues

### Problem: "Cannot find Docker daemon"
**Cause**: Docker not running
**Solution**:
```bash
# Windows
"C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe"

# macOS
open /Applications/Docker.app

# Linux
sudo service docker start
```

### Problem: "Permission denied while trying to connect"
**Cause**: User not in docker group
**Solution**:
```bash
# Linux: Add user to docker group
sudo usermod -aG docker $USER
```

### Problem: "Build context exceeds size limit"
**Cause**: Large files in project directory
**Solution**:
```bash
# Use .dockerignore to exclude files
echo "data/input/*" >> .dockerignore
echo "logs/*" >> .dockerignore
```

## Logging and Debugging

### Enable Debug Logging
```bash
python main.py --image test.jpg --loglevel DEBUG
```

### View Logs
```bash
# Recent logs
tail -f logs/deepfake_detection.log

# Filter by level
grep "ERROR" logs/deepfake_detection.log
```

### Python Debugger
```python
import pdb

# Set breakpoint
pdb.set_trace()

# Or in Python 3.7+
breakpoint()

# Commands: n (next), s (step), c (continue), p var (print variable)
```

## Performance Optimization

### Check System Resources
```python
import psutil
print(f"CPU: {psutil.cpu_percent()}%")
print(f"Memory: {psutil.virtual_memory().percent}%")
print(f"GPU: {torch.cuda.memory_allocated()}")
```

### Profile Code
```bash
python -m cProfile -s cumulative main.py --image test.jpg
```

## Common Prediction Issues

### High False Positives
**Problem**: Real images flagged as fake
**Solution**:
```python
# Use stricter threshold
DETECTION_THRESHOLDS["fake"] = 0.85  # Higher threshold

# Or use ensemble for consensus
model_type = "ensemble"
```

### High False Negatives
**Problem**: Fake images not detected
**Solution**:
```python
# Use looser threshold
DETECTION_THRESHOLDS["fake"] = 0.65  # Lower threshold

# Try different model
model_type = "dlib"  # More robust detector
```

## Data Issues

### Problem: Corrupted image file
**Solution**:
```python
from PIL import Image
try:
    img = Image.open('image.jpg')
    img.verify()
except Exception as e:
    print(f"Corrupted image: {e}")
```

### Problem: Unusual color space
**Solution**:
```python
# Ensure consistency
from utils.image_utils import ImageProcessor
processor = ImageProcessor()
image = processor.convert_color_space(image, "BGR", "RGB")
```

## Getting Help

### Information to collect
1. Python version: `python --version`
2. Package versions: `pip list | grep -E 'torch|opencv|mediapipe'`
3. GPU status: `nvidia-smi` (if applicable)
4. Full error message and traceback
5. Test image/video that reproduces issue
6. Your configuration settings

### Debug commands
```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Check OpenCV
python -c "import cv2; print(cv2.__version__)"

# Check MediaPipe
python -c "import mediapipe; print(mediapipe.__version__)"

# Check dlib
python -c "import dlib; print(dlib.__version__)"

# Check Flask
python -c "import flask; print(flask.__version__)"
```

## Performance Benchmarking

### Run benchmark
```python
from utils.evaluation_utils import ModelEvaluator
import numpy as np

# Create test images
test_images = [np.random.randint(0, 255, (224, 224, 3)) for _ in range(10)]

# Benchmark
results = ModelEvaluator.benchmark_inference(model, test_images)
print(f"Average time: {results['mean_time']:.4f}s")
```

## If All Else Fails

1. **Create fresh environment**:
   ```bash
   python -m venv fresh_env
   source fresh_env/bin/activate
   pip install -r requirements.txt
   ```

2. **Check Python version**:
   ```bash
   python --version  # Should be 3.8+
   ```

3. **Try example code**:
   ```bash
   python examples.py
   ```

4. **Review full logs**:
   ```bash
   cat logs/deepfake_detection.log
   ```

5. **Check project README**:
   See README.md for comprehensive documentation
"""

if __name__ == "__main__":
    print(TROUBLESHOOTING_GUIDE)
