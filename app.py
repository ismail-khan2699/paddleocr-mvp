from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import uvicorn
from typing import Optional
import io
from PIL import Image

app = FastAPI(title="PaddleOCR Service", version="1.0.0")

# Initialize PaddleOCR with PP-OCRv4 and angle classification
# CPU mode is default, enable angle classification
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
)


@app.get("/")
async def root():
    return {"message": "PaddleOCR Service API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = 0.5
):
    """
    Perform OCR on uploaded image using PaddleOCR with PP-OCRv4 and angle classification.
    
    Args:
        file: Image file to process
        confidence_threshold: Minimum confidence score (0.0-1.0) to include text in results. Default: 0.5
    
    Returns:
        JSON response with OCR results filtered by confidence threshold
    """
    # Validate confidence threshold
    if not 0.0 <= confidence_threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="confidence_threshold must be between 0.0 and 1.0"
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL Image to numpy array for PaddleOCR
        import numpy as np
        image_array = np.array(image)
        
        # Perform OCR (angle classification is already enabled via use_angle_cls=True)
        result = ocr.ocr(image_array)
        
        # Process results and filter by confidence threshold
        ocr_results = []
        if result:
            # PaddleOCR returns a list, where result[0] contains the detections
            # Handle both list and None cases
            detections = result[0] if isinstance(result, list) and len(result) > 0 and result[0] is not None else []
            
            for detection in detections:
                try:
                    if not detection:
                        continue
                    
                    # Handle different result structures:
                    # Format 1: [bbox, (text, confidence)]
                    # Format 2: [bbox, text, confidence]
                    if len(detection) == 2:
                        # Format 1: [bbox, (text, confidence)]
                        bbox = detection[0]
                        text_info = detection[1]
                        
                        if isinstance(text_info, (list, tuple)) and len(text_info) == 2:
                            text, confidence = text_info[0], text_info[1]
                        else:
                            # Fallback: treat as text only
                            text = str(text_info)
                            confidence = 1.0
                            
                    elif len(detection) == 3:
                        # Format 2: [bbox, text, confidence]
                        bbox, text, confidence = detection[0], detection[1], detection[2]
                    else:
                        # Unknown format, skip
                        continue
                    
                    # Ensure confidence is numeric
                    try:
                        confidence = float(confidence)
                    except (ValueError, TypeError):
                        confidence = 1.0
                    
                    if confidence >= confidence_threshold:
                        ocr_results.append({
                            "text": str(text),
                            "confidence": float(confidence),
                            "bbox": [[float(point[0]), float(point[1])] for point in bbox]
                        })
                except (ValueError, TypeError, IndexError) as e:
                    # Skip malformed detections
                    continue
        
        return JSONResponse(content={
            "success": True,
            "results": ocr_results,
            "total_detections": len(ocr_results),
            "confidence_threshold": confidence_threshold
        })
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

