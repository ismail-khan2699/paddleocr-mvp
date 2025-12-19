# PaddleOCR FastAPI Service

A FastAPI-based OCR service using PaddleOCR with PP-OCRv4 and angle classification, optimized for CPU usage.

## Features

- **PP-OCRv4**: Latest PaddleOCR model for text detection and recognition
- **Angle Classification**: Automatically corrects rotated text
- **Confidence Threshold**: Filter OCR results by confidence score
- **CPU Optimized**: Runs efficiently on CPU without GPU requirements
- **Docker Support**: Containerized for easy deployment

## API Endpoints

### Health Check
```
GET /health
```

### OCR Endpoint
```
POST /ocr
```

**Parameters:**
- `file` (multipart/form-data): Image file to process
- `confidence_threshold` (query parameter, optional): Minimum confidence score (0.0-1.0). Default: 0.5

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "text": "detected text",
      "confidence": 0.95,
      "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
  ],
  "total_detections": 1,
  "confidence_threshold": 0.5
}
```

## Usage

### Using Docker

1. Build the Docker image:
```bash
docker build -t paddleocr-service .
```

2. Run the container:
```bash
docker run -p 8000:8000 paddleocr-service
```

3. Test the API:
```bash
curl -X POST "http://localhost:8000/ocr?confidence_threshold=0.7" \
  -F "file=@your_image.jpg"
```

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

3. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Example Request

```bash
curl -X POST "http://localhost:8000/ocr?confidence_threshold=0.6" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_image.png"
```

## Notes

- The service uses CPU mode by default (no GPU required)
- First request may take longer as models are downloaded and loaded
- Supported image formats: JPEG, PNG, BMP, etc.
- Confidence threshold filters out low-confidence detections
- Raw OCR results are returned without field parsing

