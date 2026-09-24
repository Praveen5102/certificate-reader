# Certificate Reader - web app + OCR + LiLT model (CPU)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2

# runtime libraries for onnxruntime / opencv / torch
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-deploy.txt

COPY src ./src
COPY scripts/serve.py ./scripts/serve.py
COPY configs ./configs
COPY deploy ./deploy

# download the OCR models and join the model parts at build time, so startup is fast
RUN python -c "import sys; sys.path.insert(0, '.'); from src.ocr.engine import OCREngine; OCREngine()" \
    && python -c "import sys; sys.path.insert(0, '.'); from src.models.layout_tagger import load_model; load_model('deploy/model')"

EXPOSE 8000
CMD ["python", "scripts/serve.py"]
