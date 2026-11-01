#!/usr/bin/env bash
# Install VERONICA's dependencies natively (macOS on Apple Silicon, or Linux).
#   ./scripts/install.sh            backend (uv) + frontend (npm)
#   ./scripts/install.sh --cpu      Linux without an NVIDIA GPU: CPU-only PyTorch (much smaller)
#   ./scripts/install.sh --no-multiview   skip Depth Anything 3 (multi-photo reconstruction)
#   ./scripts/install.sh --no-completer   skip the local hidden-side generator (Apple Silicon)
set -euo pipefail
. "$(dirname "$0")/lib.sh"

CPU=0; MULTIVIEW=1; COMPLETER=1
for a in "$@"; do case "$a" in
  --cpu) CPU=1 ;;
  --no-multiview) MULTIVIEW=0 ;;
  --no-completer) COMPLETER=0 ;;
  -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
  *) die "unknown option $a" ;;
esac; done

banner
OS="$(uname -s)"; ARCH="$(uname -m)"
say "Platform: $OS $ARCH"
[ "$OS" = "Darwin" ] && [ "$ARCH" != "arm64" ] && warn "Intel Macs have no MPS GPU; inference will run on the CPU (slow)."

# --- prerequisites -------------------------------------------------------------
if ! have uv; then
  die "uv is required (Python package manager). Install it, then re-run:
    curl -LsSf https://astral.sh/uv/install.sh | sh        (or: brew install uv)"
fi
ok "uv $(uv --version | awk '{print $2}')"
if ! have node || ! have npm; then
  die "Node.js ≥ 20 with npm is required: https://nodejs.org (or: brew install node)"
fi
NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
[ "$NODE_MAJOR" -ge 20 ] || die "Node.js ≥ 20 required (found $(node -v))"
ok "node $(node -v)"

# --- backend -------------------------------------------------------------------
cd "$ROOT/backend"
EXTRA=(); [ "$MULTIVIEW" = 1 ] && EXTRA=(--extra multiview)
if [ "$CPU" = 1 ]; then
  say "Installing backend with CPU-only PyTorch…"
  uv export --frozen ${EXTRA[@]+"${EXTRA[@]}"} --group dev --no-hashes --no-emit-project -o .requirements.txt
  # the lock lists Linux CUDA runtime packages; a CPU install does not need them (~4 GB)
  grep -v -E '^(nvidia-|cuda-|triton)' .requirements.txt > .requirements.cpu.txt && mv .requirements.cpu.txt .requirements.txt
  uv venv --allow-existing --python 3.12
  uv pip install --torch-backend cpu -r .requirements.txt
  rm -f .requirements.txt
else
  say "Installing backend (Python 3.12, PyTorch with MPS / CUDA)…"
  uv sync --python 3.12 ${EXTRA[@]+"${EXTRA[@]}"} --inexact
fi
if [ "$MULTIVIEW" = 1 ]; then
  # its declared dependencies include CUDA-only xformers; the code does not need it
  uv pip install --no-deps "depth-anything-3==0.1.1"
fi
ok "backend ready ($(./.venv/bin/python -c 'import torch;print("torch", torch.__version__)'))"

# --- local completer (Apple Silicon) --------------------------------------------
# Hunyuan3D-2mini generates hidden sides on the Mac's GPU. It lives in its own venv so its
# dependencies never touch the backend's; run.sh starts it. Weights (~4 GB) download on
# first use, or with setup.sh --download-models.
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ] && [ "$COMPLETER" = 1 ]; then
  say "Installing the local hidden-side generator (Hunyuan3D-2mini on Metal)…"
  cd "$ROOT/completer-worker"
  uv venv --allow-existing --python 3.12 -q
  uv pip install --python .venv/bin/python -q -r requirements-mac.txt
  uv pip install --python .venv/bin/python -q --no-deps "hy3dgen==2.0.2"
  ok "local completer ready"
  warn "Hunyuan3D-2mini weights use the Tencent Hunyuan 3D 2.0 Community License, which does not
    grant rights in the EU, the UK or South Korea. Opt out with --no-completer (or
    VERONICA_LOCAL_COMPLETER=0 in .env) if that applies to you."
fi

# --- frontend ------------------------------------------------------------------
cd "$ROOT/frontend"
say "Installing frontend…"
if [ -f package-lock.json ]; then npm ci --no-audit --no-fund; else npm install --no-audit --no-fund; fi
ok "frontend ready"

echo
ok "Installed. Next: ${B}./scripts/setup.sh${N} (config + model download), then ${B}./scripts/run.sh${N}"
