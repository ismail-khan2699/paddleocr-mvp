from paddleocr import PaddleOCR
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Create an image with text
img = Image.new('RGB', (300, 100), color=(255, 255, 255))
d = ImageDraw.Draw(img)
# Use default font
d.text((10, 10), "Hello World", fill=(0, 0, 0))
img_np = np.array(img)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Run OCR
result = ocr.ocr(img_np, cls=True)

print(f"Result type: {type(result)}")
print(f"Result length: {len(result)}")
print(f"Result: {result}")

if isinstance(result, list) and len(result) > 0:
    print(f"Result[0] type: {type(result[0])}")
    print(f"Result[0]: {result[0]}")
