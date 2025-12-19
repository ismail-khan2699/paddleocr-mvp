FROM paddlepaddle/paddle:2.6.1-gpu-cuda11.7-cudnn8.4-trt8.4

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
# Note: paddlepaddle-gpu is already in the base image
RUN pip install --no-cache-dir -r requirements.txt

COPY download_models.py .
# Run model download (inference on CPU safe during build)
RUN python download_models.py && rm download_models.py

COPY app.py .

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
