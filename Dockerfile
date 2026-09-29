# Certificate Reader - web app + OCR + LiLT model (CPU)
FROM python:3.12-slim

# PIP_RETRIES / PIP_DEFAULT_TIMEOUT: builder networks sometimes drop a download
# mid-way ("Connection broken: Broken pipe"); retry instead of failing the build
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_RETRIES=10 PIP_DEFAULT_TIMEOUT=120 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# runtime libraries for onnxruntime / opencv / torch
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-deploy.txt .
# whole install retried up to 3 times (a download cut mid-stream is not always retried by pip)
RUN for i in 1 2 3; do \
        pip install torch --index-url https://download.pytorch.org/whl/cpu \
        && pip install -r requirements-deploy.txt \
        && ok=1 && break; \
        echo "pip install failed (attempt $i), retrying"; sleep 15; \
    done; [ "$ok" = 1 ] \
    # rapidocr pulls the desktop OpenCV build (needs X11/libxcb); servers need the headless one
    && pip uninstall -y opencv-python \
    && pip install opencv-python-headless==5.0.0.93

COPY src ./src
COPY scripts/serve.py ./scripts/serve.py
COPY configs ./configs
COPY deploy ./deploy

# download the OCR models and join the model parts at build time, so startup is fast
RUN python -c "import sys; sys.path.insert(0, '.'); from src.ocr.engine import OCREngine; OCREngine()" \
    && python -c "import sys; sys.path.insert(0, '.'); from src.models.layout_tagger import load_model; load_model('deploy/model')"

EXPOSE 8000
CMD ["python", "scripts/serve.py"]
