"""
Flask API application for deepfake detection.
"""

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pathlib import Path
import logging
import os

from config import API_CONFIG, DATA_PATHS, MODEL_CONFIG, PROJECT_ROOT
from pipeline import DeepfakeDetectionPipeline
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__)
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = API_CONFIG['max_content_length']
    app.config['UPLOAD_FOLDER'] = str(DATA_PATHS['input_data'])
    
    # Initialize pipeline
    pipeline = DeepfakeDetectionPipeline(
        face_detector_type=MODEL_CONFIG['face_detector']['model_type'],
        deepfake_detector_type=MODEL_CONFIG['deepfake_detector']['model_type']
    )
    
    # Ensure upload folder exists
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    
    # CORS headers
    @app.before_request
    def handle_cors():
        """Handle CORS preflight requests."""
        if request.method == 'OPTIONS':
            response = app.make_default_options_response()
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
            return response
    
    @app.after_request
    def add_cors_headers(response):
        """Add CORS headers to all responses."""
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    
    # Allowed file extensions
    ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}
    
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
            "version": "1.0.0"
        }), 200
    
    @app.route('/api/analyze-image', methods=['POST'])
    def analyze_image():
        """
        Analyze an image for deepfake content.
        
        Expected: POST request with image file
        Returns: JSON with detection results
        """
        try:
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            if not allowed_file(file.filename, 'image'):
                return jsonify({"error": "Image format not allowed"}), 400
            
            filename = secure_filename(file.filename)
            filepath = Path(app.config['UPLOAD_FOLDER']) / filename
            file.save(str(filepath))
            
            logger.info(f"Analyzing image: {filename}")
            
            results = pipeline.analyze_image(str(filepath))
            output_image_path = Path(DATA_PATHS['output_data']) / f"output_{filename}"
            results['output_image_path'] = str(output_image_path)
            
            return jsonify(results), 200
        
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/analyze-video', methods=['POST'])
    def analyze_video():
        """
        Analyze a video for deepfake content.
        
        Expected: POST request with video file
        Returns: JSON with detection results
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
            filepath = Path(app.config['UPLOAD_FOLDER']) / filename
            file.save(str(filepath))
            
            logger.info(f"Analyzing video: {filename}")
            
            results = pipeline.analyze_video(str(filepath))
            
            return jsonify(results), 200
        
        except Exception as e:
            logger.error(f"Error analyzing video: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/batch-analyze', methods=['POST'])
    def batch_analyze():
        """
        Batch analyze multiple images or videos.
        
        Expected: POST request with multiple files
        Returns: JSON with results for each file
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
                filepath = Path(app.config['UPLOAD_FOLDER']) / filename
                file.save(str(filepath))
                
                # Determine file type
                if allowed_file(filename, 'image'):
                    result = pipeline.analyze_image(str(filepath))
                    result['filename'] = filename
                    result['type'] = 'image'
                elif allowed_file(filename, 'video'):
                    result = pipeline.analyze_video(str(filepath))
                    result['filename'] = filename
                    result['type'] = 'video'
                else:
                    result = {'filename': filename, 'error': 'Unsupported file format'}
                
                batch_results.append(result)
            
            return jsonify({
                "total_files": len(batch_results),
                "results": batch_results
            }), 200
        
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/model-info', methods=['GET'])
    def model_info():
        """Get information about loaded models."""
        return jsonify({
            "models_loaded": {
                "face_detector": MODEL_CONFIG['face_detector'],
                "deepfake_detector": MODEL_CONFIG['deepfake_detector']
            }
        }), 200
    
    @app.route('/', methods=['GET'])
    def serve_index():
        """Serve the web interface."""
        index_path = PROJECT_ROOT / "index.html"
        if index_path.exists():
            return send_file(str(index_path))
        return jsonify({"error": "Web interface not found"}), 404
    
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
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug']
    )
