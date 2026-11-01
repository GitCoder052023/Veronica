#!/usr/bin/env bash
# One command for everything: install → setup → run. Safe to re-run; finished steps are
# skipped, and dependencies reinstall by themselves when the lockfiles change (git pull).
#   ./start.sh                  install if needed, download models once, start VERONICA
#   ./start.sh --skip-models    don't pre-download weights (they download on first use)
#   ./start.sh --reinstall      force the install step
#   ./start.sh --prod           serve the production build of the studio
#   ./start.sh --no-open        don't open the browser
#   ./start.sh --cpu | --no-multiview | --no-completer   passed to scripts/install.sh
set -euo pipefail
. "$(dirname "$0")/scripts/lib.sh"

INSTALL_ARGS=(); RUN_ARGS=(); MODELS=yes; FORCE=0
for a in "$@"; do case "$a" in
  --cpu|--no-multiview) INSTALL_ARGS+=("$a") ;;
  --no-completer) INSTALL_ARGS+=("$a"); export VERONICA_LOCAL_COMPLETER=0 ;;
  --prod|--no-open) RUN_ARGS+=("$a") ;;
  --skip-models) MODELS=no ;;
  --reinstall) FORCE=1 ;;
  -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
  *) die "unknown option $a (see ./start.sh --help)" ;;
esac; done

STATE="$ROOT/.state"; mkdir -p "$STATE"

# --- 1. install: when anything it depends on changed since the last successful run ----
fingerprint() {
  {
    printf '%s\n' ${INSTALL_ARGS[@]+"${INSTALL_ARGS[@]}"}
    cat "$ROOT/backend/uv.lock" "$ROOT/frontend/package-lock.json" "$ROOT/completer-worker/requirements-mac.txt" \
        "$ROOT/scripts/install.sh" 2>/dev/null
  } | shasum -a 256 | cut -c1-16
}
FP="$(fingerprint)"
installed() {
  [ -x "$ROOT/backend/.venv/bin/uvicorn" ] && [ -d "$ROOT/frontend/node_modules" ] && \
    [ "$(cat "$STATE/installed" 2>/dev/null)" = "$FP" ]
}
if [ "$FORCE" = 1 ] || ! installed; then
  say "Step 1/3 — installing dependencies"
  "$ROOT/scripts/install.sh" ${INSTALL_ARGS[@]+"${INSTALL_ARGS[@]}"}
  echo "$FP" > "$STATE/installed"
else
  ok "Step 1/3 — dependencies up to date"
fi

# --- 2. setup: .env plus a one-time model download --------------------------------------
if [ ! -f "$ROOT/.env" ] || { [ "$MODELS" = yes ] && [ ! -f "$STATE/models" ]; }; then
  say "Step 2/3 — configuring"
  if [ "$MODELS" = yes ]; then
    "$ROOT/scripts/setup.sh" --download-models
    touch "$STATE/models"
  else
    "$ROOT/scripts/setup.sh" --skip-models
  fi
else
  ok "Step 2/3 — configured (.env present, models downloaded)"
fi

# --- 3. run -----------------------------------------------------------------------------
say "Step 3/3 — starting VERONICA"
exec "$ROOT/scripts/run.sh" ${RUN_ARGS[@]+"${RUN_ARGS[@]}"}
