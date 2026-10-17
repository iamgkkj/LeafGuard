# LeafGuard AI backend — Hugging Face Spaces (Docker SDK)
FROM python:3.11-slim

WORKDIR /app

# System libs needed by grad-cam / matplotlib headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (lean backend set)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r backend/requirements.txt

# Backend source
COPY backend ./backend

# Keep model checkpoints out of the image; they are downloaded at startup
# from HF_MODEL_REPO (or can be pre-staged in backend/runs/models locally).

WORKDIR /app/backend

# Headless matplotlib + single-threaded torch for low-RAM containers
ENV MPLBACKEND=Agg
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TORCH_CPP_LOG_LEVEL=ERROR

# HF Spaces expect the app to listen on 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
