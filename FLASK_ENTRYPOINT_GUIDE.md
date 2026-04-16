# Flask Entrypoint Configuration

## Problem Solved
The error "No flask entrypoint found" has been resolved by creating proper entrypoint files and configuration.

## Available Entrypoints

### 1. **wsgi.py** (Recommended for Production)
```bash
# Using gunicorn (production)
gunicorn wsgi:app

# Using uvicorn
uvicorn wsgi:app

# Direct run
python wsgi.py
```

### 2. **app.py** (Alternative)
```bash
# Using gunicorn
gunicorn app:app

# Direct run
python app.py
```

### 3. **main.py** (Development)
```bash
# CLI interface
python main.py --api
```

## Configuration Files Created

### **pyproject.toml**
- Defines the `app` script entrypoint pointing to `wsgi:app`
- Includes all project dependencies
- Modern Python packaging standard

### **setup.py**
- Alternative packaging configuration
- Includes console script `deepfake-api=wsgi:app`
- Compatible with older deployment systems

## Deployment Examples

### **Gunicorn (Production)**
```bash
# Install gunicorn
pip install gunicorn

# Run with wsgi.py
gunicorn --bind 0.0.0.0:5000 wsgi:app

# Run with app.py
gunicorn --bind 0.0.0.0:5000 app:app
```

### **Docker**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]
```

### **Heroku**
```yaml
# Procfile
web: gunicorn wsgi:app
```

### **Railway/Vercel**
The `wsgi.py` file serves as the entrypoint for these platforms.

## Testing Entrypoints

Both entrypoints have been tested and work correctly:

```bash
# Test wsgi.py
python -c "from wsgi import app; print('WSGI works!')"

# Test app.py
python -c "from app import app; print('App works!')"
```

## File Structure
```
deepfake-detection-api/
├── wsgi.py          # Production entrypoint
├── app.py           # Alternative entrypoint
├── main.py          # CLI entrypoint
├── pyproject.toml   # Modern packaging
├── setup.py         # Legacy packaging
└── api/
    └── app.py       # Flask app factory
```