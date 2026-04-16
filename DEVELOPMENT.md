"""
Development and Deployment Guide
"""

DEVELOPMENT_GUIDE = """
# Development & Deployment Guide

## Local Development Setup

### 1. Clone Repository
```bash
cd path/to/Deepfake\ AI\ PJ
git init
git add .
git commit -m "Initial commit: Deepfake Detection System"
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\\Scripts\\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Tests
```bash
python -m pytest tests/ -v
```

### 5. Start Development Server
```bash
# CLI mode
python main.py --image data/input/test.jpg

# API mode with debug
python main.py --api
```

## Code Structure & Conventions

### File Organization
- `config.py`: Global configuration settings
- `main.py`: CLI entry point
- `pipeline.py`: Core detection logic
- `models/`: Detection models and wrappers
- `utils/`: Reusable utilities
- `api/`: Flask API endpoints

### Naming Conventions
- Classes: PascalCase (e.g., `FaceDetector`)
- Functions: snake_case (e.g., `detect_faces`)
- Constants: UPPER_CASE (e.g., `MAX_FRAMES`)
- Private methods: _snake_case (e.g., `_initialize_detector`)

### Documentation
- Module docstrings: Describe purpose
- Function docstrings: Args, returns, examples
- Inline comments: Explain complex logic

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test
```bash
pytest tests/__init__.py::TestFaceDetector -v
```

### Generate Coverage Report
```bash
pytest --cov=. tests/
```

## Performance Optimization

### Profiling
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
pipeline.analyze_image("test.jpg")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Memory Usage Monitoring
```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

## Debugging

### Enable Debug Logging
```bash
python main.py --image test.jpg --loglevel DEBUG
```

### Python Debugger
```python
import pdb; pdb.set_trace()
```

### VS Code Debugging
Create `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal"
    }
  ]
}
```

## Docker Deployment

### Build Image
```bash
docker build -t deepfake-detector .
```

### Run Container
```bash
docker run -p 5000:5000 -v $(pwd)/data:/app/data deepfake-detector
```

### Using Docker Compose
```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

## Production Deployment

### 1. Using Gunicorn (WSGI Server)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api.app:create_app
```

### 2. Using Nginx (Reverse Proxy)
```nginx
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Using Systemd Service
Create `/etc/systemd/system/deepfake.service`:
```ini
[Unit]
Description=Deepfake Detection Service
After=network.target

[Service]
Type=notify
User=deepfake
WorkingDirectory=/opt/deepfake
Environment="PATH=/opt/deepfake/venv/bin"
ExecStart=/opt/deepfake/venv/bin/gunicorn -w 4 api.app:create_app

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
sudo systemctl start deepfake
sudo systemctl enable deepfake
```

## Database Integration (Optional)

For storing analysis results:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# PostgreSQL example
DATABASE_URL = "postgresql://user:password@localhost/deepfake_db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
```

## CI/CD Pipeline (GitHub Actions)

Create `.github/workflows/test.yml`:
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
```

## Monitoring & Logging

### Application Metrics
```python
from utils.evaluation_utils import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start()
# Run analysis
results = pipeline.analyze_image("test.jpg")
monitor.record_image_analysis(results['processing_time'], results['faces_detected'])
print(monitor.get_summary())
```

### Log Aggregation (ELK Stack)
```python
from pythonjsonlogger import jsonlogger
import logging
import sys

logHandler = logging.StreamHandler(sys.stdout)
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
```

## Model Training & Fine-tuning

### Transfer Learning Example
```python
import torch
import torch.nn as nn
from torchvision import models

# Load pre-trained model
base_model = models.resnet50(pretrained=True)

# Freeze base layers
for param in base_model.parameters():
    param.requires_grad = False

# Add custom classification head
num_ftrs = base_model.fc.in_features
base_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Linear(256, 2)
)

# Train only the new head
optimizer = torch.optim.Adam(base_model.fc.parameters(), lr=1e-3)
```

## Security Considerations

1. **Input Validation**: Validate all file uploads
2. **Rate Limiting**: Implement API rate limiting
3. **Authentication**: Add API key authentication for production
4. **HTTPS**: Enable HTTPS for production
5. **File Upload**: Restrict file types and sizes
6. **Error Handling**: Don't expose sensitive info in errors

## Performance Benchmarks

Expected performance on typical hardware:

```
Device: GPU (NVIDIA V100)
Image Analysis: ~100-200 images/minute
Video Analysis: ~0.5-1 video/minute (1 hour video)
Memory: ~2-4GB
```

```
Device: CPU (Intel i7)
Image Analysis: ~10-20 images/minute
Video Analysis: ~0.1 video/minute (1 hour video)
Memory: ~1-2GB
```

## Troubleshooting

### Import Errors
```bash
python -c "import torch; print(torch.__version__)"
```

### Memory Issues
- Reduce batch size
- Process shorter videos
- Use GPU acceleration

### API Not Responding
```bash
curl -v http://localhost:5000/api/health
```

## Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test
3. Commit: `git commit -am "Add feature"`
4. Push: `git push origin feature/your-feature`
5. Create Pull Request

## Version Control

```bash
# Tag releases
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

## Documentation

- Update docstrings for any new functions
- Keep README.md synchronized
- Add examples for new features
- Document breaking changes in CHANGELOG.md

## Resources

- PyTorch Docs: https://pytorch.org/docs/
- OpenCV Docs: https://docs.opencv.org/
- Flask Docs: https://flask.palletsprojects.com/
- MediaPipe Docs: https://mediapipe.dev/
"""

if __name__ == "__main__":
    print(DEVELOPMENT_GUIDE)
