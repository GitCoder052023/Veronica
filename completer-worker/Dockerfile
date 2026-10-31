# VERONICA completer worker: TRELLIS image-to-3D on an NVIDIA GPU (≥16 GB VRAM).
# Build:  docker build -t veronica-completer .
# Run:    docker run --gpus all -p 8790:8790 -e VERONICA_COMPLETER_TOKEN=change-me \
#           -v $HOME/.cache/huggingface:/root/.cache/huggingface veronica-completer
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" SPCONV_ALGO=native ATTN_BACKEND=xformers
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# TRELLIS (MIT) and its CUDA extensions, via the project's own setup script
RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /opt/TRELLIS
WORKDIR /opt/TRELLIS
RUN bash -c ". ./setup.sh --basic --xformers --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast"
ENV PYTHONPATH=/opt/TRELLIS

WORKDIR /app
COPY requirements.txt worker.py ./
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8790
CMD ["uvicorn", "worker:app", "--host", "0.0.0.0", "--port", "8790"]
