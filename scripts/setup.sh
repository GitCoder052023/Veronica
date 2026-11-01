#!/usr/bin/env bash
# Configure VERONICA and (optionally) pre-download model weights (~3.5 GB).
#   ./scripts/setup.sh                    create .env, ask whether to download models
#   ./scripts/setup.sh --download-models  download without asking
#   ./scripts/setup.sh --skip-models      never download (models then download on first use)
set -euo pipefail
. "$(dirname "$0")/lib.sh"

DL=ask
for a in "$@"; do case "$a" in
  --download-models|-y) DL=yes ;;
  --skip-models) DL=no ;;
  -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
  *) die "unknown option $a" ;;
esac; done

banner
[ -x "$ROOT/backend/.venv/bin/python" ] || die "Dependencies missing — run ./scripts/install.sh first."

if [ ! -f "$ROOT/.env" ]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  ok "Created .env from .env.example"
else
  ok ".env exists (left unchanged)"
fi
load_env

PY="$ROOT/backend/.venv/bin/python"
cd "$ROOT/backend"
if [ "$DL" = ask ]; then
  if [ -t 0 ]; then
    read -r -p "Download model weights now (~3.5 GB)? [Y/n] " ans
    case "${ans:-y}" in [Nn]*) DL=no ;; *) DL=yes ;; esac
  else
    DL=no
  fi
fi
if [ "$DL" = yes ]; then
  say "Downloading model weights into the Hugging Face cache…"
  "$PY" -m veronica.cli download-models
  CW="$ROOT/completer-worker/.venv/bin/python"
  if [ -x "$CW" ] && [ "${VERONICA_LOCAL_COMPLETER:-1}" != 0 ]; then
    say "Downloading Hunyuan3D-2mini for the local completer (~3.8 GB)…"
    "$CW" -c "from huggingface_hub import snapshot_download as d; d('tencent/Hunyuan3D-2mini', allow_patterns=['hunyuan3d-dit-v2-mini-turbo/config.yaml', 'hunyuan3d-dit-v2-mini-turbo/model.fp16.safetensors', 'LICENSE', 'NOTICE'])"
  fi
else
  warn "Skipped model download; weights will download on the first reconstruction."
fi

echo
"$PY" -m veronica.cli doctor
echo
ok "Setup done. Start VERONICA with ${B}./scripts/run.sh${N}"
