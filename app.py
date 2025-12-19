from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from paddleocr import PaddleOCR
import uvicorn
from typing import Optional
import io
import numpy as np
from PIL import Image

app = FastAPI(title="PaddleOCR Service", version="1.0.0")

# Initialize PaddleOCR with PP-OCRv4 and angle classification
# CPU mode is default, enable angle classification
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
) 


@app.on_event("startup")
def warmup():
    """Warm up PaddleOCR models on startup to avoid slow first request."""
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)
    ocr.ocr(dummy)


@app.get("/")
async def root():
    return {"message": "PaddleOCR Service API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


async def process_image_file(file: UploadFile):
    """Helper function to read and process image file."""
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    # Read image file
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    image = Image.open(io.BytesIO(image_bytes))
    # Convert to RGB if necessary (PaddleOCR expects RGB format)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = np.array(image)
    
    # Convert RGB to BGR (PaddleOCR uses OpenCV which expects BGR)
    image_array = image_array[:, :, ::-1]
    
    # Perform OCR (angle classification is already enabled via use_angle_cls=True)
    # Run blocking CPU-bound OCR operation in thread pool to maintain async concurrency
    result = await run_in_threadpool(ocr.ocr, image_array)
    
    # Debug logging
    print(f"OCR result type: {type(result)}")
    if isinstance(result, list):
        print(f"OCR result length: {len(result)}")
        if len(result) > 0:
            print(f"First element type: {type(result[0])}")
            if isinstance(result[0], list):
                 print(f"First element length: {len(result[0])}")
            elif result[0] is None:
                 print("First element is None")
            else:
                 print(f"First element: {str(result[0])[:100]}...")
    
    return result


def parse_ocr_results(result, confidence_threshold: float = 0.0):
    """Parse PaddleOCR results and return normalized, sorted results."""
    ocr_results = []
    
    if not result:
        return ocr_results

    # Handle new PaddleOCR/PaddleX result format (dict-like object with 'rec_texts', 'dt_polys', etc.)
    # In newer versions, result is [OCRResult] where OCRResult behaves like a dict
    is_new_format = False
    if isinstance(result, list) and len(result) > 0:
        first_item = result[0]
        # Check if it has the keys specific to the new format
        try:
            if hasattr(first_item, '__getitem__') and 'rec_texts' in first_item and 'dt_polys' in first_item:
                is_new_format = True
        except (TypeError, AttributeError):
            pass
            
    if is_new_format:
        try:
            item = result[0]
            rec_texts = item['rec_texts']
            rec_scores = item['rec_scores']
            dt_polys = item['dt_polys']
            
            for i in range(len(rec_texts)):
                text = rec_texts[i]
                confidence = rec_scores[i]
                bbox = dt_polys[i]
                
                # Ensure confidence is numeric
                try:
                    confidence = float(confidence)
                except (ValueError, TypeError):
                    confidence = 1.0
                    
                if confidence >= confidence_threshold:
                    normalized_text = " ".join(str(text).split())
                    
                    # Handle bbox (numpy array)
                    if hasattr(bbox, 'tolist'):
                        bbox_list = bbox.tolist()
                    else:
                        bbox_list = bbox
                    
                    # Ensure bbox points are floats
                    bbox_formatted = [[float(p) for p in point] for point in bbox_list]
                    
                    ocr_results.append({
                        "text": normalized_text,
                        "confidence": float(confidence),
                        "bbox": bbox_formatted
                    })
        except Exception as e:
            print(f"Error parsing new format: {e}")
            # Fall through to try old format
            pass

    # Handle legacy PaddleOCR format
    if not ocr_results and result:
        # PaddleOCR returns a list, where result[0] contains the detections
        detections = result[0] if isinstance(result, list) and len(result) > 0 and result[0] is not None else []
        
        # If detections is the dict-like OCRResult (new format) but we fell through, 
        # iterating it will yield keys (strings), which won't match the old format logic below.
        # So we verify it's a list before iterating for old format.
        if isinstance(detections, list):
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
                        # Normalize text: trim junk and normalize whitespace
                        normalized_text = " ".join(str(text).split())
                        
                        ocr_results.append({
                            "text": normalized_text,
                            "confidence": float(confidence),
                        })
                except (ValueError, TypeError, IndexError) as e:
                    # Skip malformed detections
                    continue
    
    # Sort results top-to-bottom, left-to-right (by y-coordinate first, then x-coordinate)
    ocr_results.sort(key=lambda x: (x["bbox"][0][1], x["bbox"][0][0]))
    
    return ocr_results


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
        JSON response with OCR results filtered by confidence threshold, normalized and sorted
    """
    # Validate confidence threshold
    if not 0.0 <= confidence_threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="confidence_threshold must be between 0.0 and 1.0"
        )
    
    try:
        result = await process_image_file(file)
        ocr_results = parse_ocr_results(result, confidence_threshold)
        
        return JSONResponse(content={
            "success": True,
            "results": ocr_results,
            "total_detections": len(ocr_results),
            "confidence_threshold": confidence_threshold
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )


def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable types to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        # If array is too large (likely an image), don't serialize the whole thing
        if obj.size > 10000:
            return f"<numpy.ndarray shape={obj.shape} dtype={obj.dtype}>"
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        # Handle non-serializable objects (like PIL Font objects)
        try:
            import json
            # Try to serialize directly first
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # If serialization fails, check for specific types
            obj_type = type(obj).__name__
            if 'Font' in obj_type or 'PIL' in str(type(obj)):
                # Handle PIL Font and other PIL objects - convert to string representation
                return str(obj)
            elif hasattr(obj, '__dict__'):
                # Try to convert object to dict
                try:
                    return {key: convert_to_serializable(value) for key, value in obj.__dict__.items()}
                except:
                    return str(obj)
            else:
                # Fallback: convert to string
                return str(obj)


@app.post("/ocr/raw")
async def ocr_raw_endpoint(file: UploadFile = File(...)):
    """
    Perform OCR and return unfiltered PaddleOCR output for debugging.
    
    Args:
        file: Image file to process
    
    Returns:
        JSON response with raw PaddleOCR output (no filtering, normalization, or sorting)
    """
    try:
        result = await process_image_file(file)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_result = convert_to_serializable(result)
        
        return JSONResponse(content={
            "raw": serializable_result
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

