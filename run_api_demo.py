"""
Simple demo API for Deepfake Detection System
This version starts quickly without loading heavy ML models on startup
"""

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import logging
import json

# Create Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'data/input'
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

def allowed_file(filename, file_type='image'):
    """Check if file extension is allowed."""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    return False

# ==================== API Routes ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "deepfake-detection",
        "version": "1.0.0",
        "mode": "demo"
    }), 200

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """
    Analyze an image for deepfake content (Demo Mode).
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename, 'image'):
            return jsonify({"error": "Image format not allowed. Supported: jpg, jpeg, png, bmp, gif"}), 400
        
        filename = secure_filename(file.filename)
        filepath = f"{app.config['UPLOAD_FOLDER']}/{filename}"
        Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
        file.save(filepath)
        
        # Demo prediction (would call real pipeline in production)
        import numpy as np
        demo_confidence = float(np.random.uniform(0.3, 0.8))
        
        return jsonify({
            "status": "success",
            "mode": "demo",
            "message": "Real ML models loading - using demo predictions",
            "image_path": filepath,
            "faces_detected": np.random.randint(0, 3),
            "confidence": demo_confidence,
            "prediction": "FAKE" if demo_confidence > 0.7 else ("REAL" if demo_confidence < 0.3 else "UNCERTAIN"),
            "note": "This is a demo. Full models require PyTorch and MediaPipe setup."
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze-video', methods=['POST'])
def analyze_video():
    """
    Analyze a video for deepfake content (Demo Mode).
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename, 'video'):
            return jsonify({"error": "Video format not allowed"}), 400
        
        filename = secure_filename(file.filename)
        filepath = f"{app.config['UPLOAD_FOLDER']}/{filename}"
        Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
        file.save(filepath)
        
        # Demo prediction
        import numpy as np
        demo_confidence = float(np.random.uniform(0.3, 0.8))
        
        return jsonify({
            "status": "success",
            "mode": "demo",
            "message": "Real ML models loading - using demo predictions",
            "video_path": filepath,
            "analyzed_frames": np.random.randint(30, 100),
            "total_frames": np.random.randint(150, 300),
            "fake_frames": np.random.randint(10, 80),
            "overall_confidence": demo_confidence,
            "overall_prediction": "FAKE" if demo_confidence > 0.7 else ("REAL" if demo_confidence < 0.3 else "UNCERTAIN"),
            "note": "This is a demo. Full models require PyTorch and MediaPipe setup."
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Batch analyze multiple files (Demo Mode).
    """
    try:
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({"error": "No files provided"}), 400
        
        batch_results = []
        
        for file in files:
            if file.filename == '':
                continue
            
            filename = secure_filename(file.filename)
            filepath = f"{app.config['UPLOAD_FOLDER']}/{filename}"
            Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
            file.save(filepath)
            
            import numpy as np
            demo_confidence = float(np.random.uniform(0.3, 0.8))
            
            # Determine file type
            if allowed_file(filename, 'image'):
                result = {
                    'filename': filename,
                    'type': 'image',
                    'status': 'success',
                    'confidence': demo_confidence,
                    'prediction': "FAKE" if demo_confidence > 0.7 else ("REAL" if demo_confidence < 0.3 else "UNCERTAIN")
                }
            elif allowed_file(filename, 'video'):
                result = {
                    'filename': filename,
                    'type': 'video',
                    'status': 'success',
                    'confidence': demo_confidence,
                    'prediction': "FAKE" if demo_confidence > 0.7 else ("REAL" if demo_confidence < 0.3 else "UNCERTAIN")
                }
            else:
                result = {'filename': filename, 'error': 'Unsupported file format'}
            
            batch_results.append(result)
        
        return jsonify({
            "status": "success",
            "mode": "demo",
            "message": "Real ML models loading - using demo predictions",
            "total_files": len(batch_results),
            "results": batch_results
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about loaded models."""
    return jsonify({
        "status": "success",
        "mode": "demo",
        "message": "Full ML models loading in background",
        "models_configured": {
            "face_detector": {
                "available": ["mediapipe", "opencv", "dlib"],
                "current": "opencv (fallback)"
            },
            "deepfake_detector": {
                "available": ["ensemble", "xception", "meso", "efficientnet"],
                "current": "demo (placeholder)"
            }
        },
        "note": "Install PyTorch and MediaPipe for full functionality"
    }), 200

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({"error": "File too large. Maximum size: 100MB"}), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION API - DEMO MODE")
    print("="*60)
    print("\nServer running at: http://localhost:5000")
    print("\nAvailable Endpoints (Demo Mode):")
    print("  POST /api/analyze-image      - Analyze single image")
    print("  POST /api/analyze-video      - Analyze single video")
    print("  POST /api/batch-analyze      - Batch analyze multiple files")
    print("  GET  /api/model-info         - Get model information")
    print("  GET  /api/health             - Health check")
    print("\nTest the API:")
    print("  curl http://localhost:5000/api/health")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
