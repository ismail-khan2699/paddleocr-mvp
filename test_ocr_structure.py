from paddleocr import PaddleOCR
import numpy as np
from PIL import Image, ImageDraw
import logging

# Suppress logging
logging.getLogger('ppocr').setLevel(logging.ERROR)

img = Image.new('RGB', (300, 100), color=(255, 255, 255))
d = ImageDraw.Draw(img)
d.text((10, 10), "Hello World", fill=(0, 0, 0))
img_np = np.array(img)

ocr = PaddleOCR(use_angle_cls=True, lang='en')

print("Running OCR...")
result = ocr.ocr(img_np)

if isinstance(result, list) and len(result) > 0:
    obj = result[0]
    print(f"Is instance of dict: {isinstance(obj, dict)}")
    print(f"Has __dict__: {hasattr(obj, '__dict__')}")
    print(f"Keys: {list(obj.keys()) if hasattr(obj, 'keys') else 'No keys attribute'}")
