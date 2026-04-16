"""
API Documentation for Deepfake Detection System
"""

API_DOCUMENTATION = """
# Deepfake Detection API Documentation

## Base URL
http://localhost:5000

## Authentication
Currently no authentication required. Implement API keys for production deployment.

## Endpoints

### 1. Health Check
**Request:**
```
GET /api/health
```

**Response (200):**
```json
{
  "status": "healthy",
  "service": "deepfake-detection",
  "version": "1.0.0"
}
```

---

### 2. Analyze Single Image
**Request:**
```
POST /api/analyze-image
Content-Type: multipart/form-data

file: [image_file]
```

**Supported Formats:** JPG, JPEG, PNG, BMP, GIF

**Response (200):**
```json
{
  "status": "success",
  "image_path": "/path/to/image.jpg",
  "output_path": "/path/to/analyzed_image.jpg",
  "faces_detected": 1,
  "confidence": 0.8234,
  "prediction": "FAKE",
  "face_predictions": [
    {
      "face_id": 0,
      "confidence": 0.8234,
      "prediction": "FAKE",
      "details": {
        "method": "ensemble",
        "ensemble_score": 0.8234,
        "individual_scores": {
          "xception": 0.82,
          "meso": 0.83
        }
      }
    }
  ],
  "processing_time": 2.34
}
```

**Error Response (400):**
```json
{
  "error": "Image format not allowed"
}
```

**Example cURL:**
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/analyze-image
```

---

### 3. Analyze Single Video
**Request:**
```
POST /api/analyze-video
Content-Type: multipart/form-data

file: [video_file]
```

**Supported Formats:** MP4, AVI, MOV, MKV, FLV, WMV

**Response (200):**
```json
{
  "status": "success",
  "video_path": "/path/to/video.mp4",
  "video_properties": {
    "fps": 30.0,
    "frame_count": 300,
    "width": 1920,
    "height": 1080,
    "duration_seconds": 10.0
  },
  "analyzed_frames": 60,
  "total_frames": 300,
  "frames_with_faces": 58,
  "fake_frames": 45,
  "fake_frame_ratio": 0.775,
  "overall_confidence": 0.82,
  "overall_prediction": "FAKE",
  "frame_results": [
    {
      "frame_id": 0,
      "faces_detected": 1,
      "avg_confidence": 0.85,
      "prediction": "FAKE"
    }
  ],
  "processing_time": 15.42
}
```

**Example cURL:**
```bash
curl -X POST -F "file=@video.mp4" http://localhost:5000/api/analyze-video
```

---

### 4. Batch Analysis
**Request:**
```
POST /api/batch-analyze
Content-Type: multipart/form-data

files: [file1, file2, file3, ...]
```

**Response (200):**
```json
{
  "total_files": 3,
  "results": [
    {
      "filename": "image1.jpg",
      "type": "image",
      "status": "success",
      "faces_detected": 1,
      "confidence": 0.75,
      "prediction": "FAKE"
    },
    {
      "filename": "video1.mp4",
      "type": "video",
      "status": "success",
      "overall_prediction": "REAL",
      "fake_frame_ratio": 0.15
    },
    {
      "filename": "image2.png",
      "type": "image",
      "error": "Failed to process"
    }
  ]
}
```

**Example cURL:**
```bash
curl -X POST -F "files=@image1.jpg" -F "files=@video1.mp4" -F "files=@image2.png" \\
  http://localhost:5000/api/batch-analyze
```

---

### 5. Get Model Information
**Request:**
```
GET /api/model-info
```

**Response (200):**
```json
{
  "models_loaded": {
    "face_detector": {
      "model_type": "mediapipe",
      "confidence_threshold": 0.5
    },
    "deepfake_detector": {
      "model_type": "ensemble",
      "confidence_threshold": 0.5,
      "use_gpu": true
    }
  }
}
```

**Example cURL:**
```bash
curl http://localhost:5000/api/model-info
```

---

## Error Handling

### Common Error Responses

**413 - Request Entity Too Large:**
```json
{
  "error": "File too large. Maximum size: 100MB"
}
```

**400 - Bad Request:**
```json
{
  "error": "No file provided"
}
```

**404 - Not Found:**
```json
{
  "error": "Endpoint not found"
}
```

**500 - Internal Server Error:**
```json
{
  "error": "Internal server error"
}
```

---

## Prediction Classes

- **REAL** (confidence < 0.3): Likely authentic content
- **FAKE** (confidence > 0.7): Likely manipulated content
- **UNCERTAIN** (0.3 ≤ confidence ≤ 0.7): Borderline case requiring manual review
- **UNKNOWN**: No detectable content or processing error

---

## Rate Limiting

Current implementation has no rate limiting. 
For production, consider implementing:
- Rate limiting per IP
- Batch processing queues
- API key-based limits

---

## Best Practices

1. **File Size:** Keep files under 100MB for optimal performance
2. **Batch Processing:** Use batch endpoint for multiple files (more efficient)
3. **Video Optimization:** For very long videos, use frame extraction interval settings
4. **Error Handling:** Always check response status and handle errors gracefully
5. **Async Processing:** Consider implementing async task queue for large files

---

## Python Client Example

```python
import requests
import json

# Initialize API endpoint
API_URL = "http://localhost:5000"

def analyze_image(image_path):
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/api/analyze-image", files=files)
    return response.json()

def analyze_video(video_path):
    with open(video_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/api/analyze-video", files=files)
    return response.json()

def batch_analyze(file_paths):
    files = []
    for path in file_paths:
        files.append(('files', open(path, 'rb')))
    response = requests.post(f"{API_URL}/api/batch-analyze", files=files)
    return response.json()

# Usage
if __name__ == "__main__":
    # Analyze single image
    result = analyze_image("test.jpg")
    print(json.dumps(result, indent=2))
    
    # Batch analyze
    results = batch_analyze(["image1.jpg", "video1.mp4"])
    print(json.dumps(results, indent=2))
```

---

## JavaScript/Node.js Client Example

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const API_URL = "http://localhost:5000";

async function analyzeImage(imagePath) {
  const formData = new FormData();
  formData.append('file', fs.createReadStream(imagePath));
  
  try {
    const response = await axios.post(
      `${API_URL}/api/analyze-image`,
      formData,
      { headers: formData.getHeaders() }
    );
    return response.data;
  } catch (error) {
    console.error('Error:', error.message);
  }
}

async function analyzeVideo(videoPath) {
  const formData = new FormData();
  formData.append('file', fs.createReadStream(videoPath));
  
  try {
    const response = await axios.post(
      `${API_URL}/api/analyze-video`,
      formData,
      { headers: formData.getHeaders() }
    );
    return response.data;
  } catch (error) {
    console.error('Error:', error.message);
  }
}

// Usage
(async () => {
  const result = await analyzeImage('test.jpg');
  console.log(JSON.stringify(result, null, 2));
})();
```

---

## Deployment Considerations

1. **Production Server:** Use Gunicorn or uWSGI instead of Flask development server
2. **Load Balancing:** Deploy multiple instances behind a load balancer
3. **GPU Support:** Ensure GPU drivers are installed for better performance
4. **Monitoring:** Set up logging and monitoring for API usage
5. **Security:** Implement authentication, rate limiting, and input validation

---

## Common Issues & Troubleshooting

### Model Loading Fails
- Ensure PyTorch/TensorFlow is installed
- Check CUDA availability for GPU acceleration
- Verify model files exist in models/ directory

### Out of Memory Error
- Reduce batch size
- Process shorter videos
- Use lighter models (MesoNet)

### Slow Performance
- Enable GPU acceleration
- Use lighter face detector (OpenCV)
- Reduce video processing interval

---

## Version History

- **v1.0.0** (Current)
  - Initial release
  - Image and video analysis
  - Ensemble detection
  - REST API with batch processing
"""

print(API_DOCUMENTATION)
