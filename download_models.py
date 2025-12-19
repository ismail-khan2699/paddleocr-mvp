from paddleocr import PaddleOCR
import numpy as np
import logging

# Set logging to avoid console clutter during build
logging.getLogger("ppocr").setLevel(logging.ERROR)

print("Downloading PaddleOCR models...")
# Initialize PaddleOCR to trigger model download
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en'
)

# Run a dummy inference to ensure everything is loaded/compiled
dummy = np.zeros((10, 10, 3), dtype=np.uint8)
ocr.ocr(dummy)
print("PaddleOCR models downloaded and verified.")
