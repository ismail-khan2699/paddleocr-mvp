FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    paddleocr \
    paddlepaddle \
    fastapi \
    uvicorn \
    python-multipart \
    pillow \
    numpy

COPY download_models.py .
RUN python download_models.py && rm download_models.py

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

