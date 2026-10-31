# VERONICA API (reconstruction + inference).
#   CPU (default, any machine):  docker build -t veronica-api backend
#   NVIDIA GPU:                  docker build --build-arg TORCH_BACKEND=cu126 -t veronica-api backend
# Note: containers cannot use the Apple GPU (Metal/MPS); on a Mac, run natively
# (scripts/install.sh + scripts/run.sh) for GPU speed.
FROM python:3.12-slim-bookworm

ARG TORCH_BACKEND=cpu
ARG PREFETCH_MODELS=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
    VERONICA_DATA=/data HF_HOME=/models

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
# Locked dependencies; torch comes from the PyTorch index matching TORCH_BACKEND.
# CPU builds drop the CUDA runtime packages the lock carries for Linux (~4 GB).
RUN uv export --frozen --extra multiview --no-hashes --no-emit-project -o /tmp/requirements.txt \
 && if [ "$TORCH_BACKEND" = "cpu" ]; then grep -v -E '^(nvidia-|cuda-|triton)' /tmp/requirements.txt > /tmp/r && mv /tmp/r /tmp/requirements.txt; fi \
 && uv venv /opt/venv \
 && uv pip install --no-cache --torch-backend "$TORCH_BACKEND" -r /tmp/requirements.txt \
 && uv pip install --no-cache --no-deps depth-anything-3==0.1.1

COPY veronica ./veronica
RUN if [ "$PREFETCH_MODELS" = "1" ]; then python -m veronica.cli download-models; fi

VOLUME ["/data", "/models"]
EXPOSE 8787
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8787/api/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "veronica.api:app", "--host", "0.0.0.0", "--port", "8787"]
