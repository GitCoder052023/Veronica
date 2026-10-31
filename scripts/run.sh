#!/usr/bin/env bash
# Run VERONICA natively: API (FastAPI) + studio (Vite). Ctrl-C stops both.
#   ./scripts/run.sh             development servers, opens the browser
#   ./scripts/run.sh --prod      production build of the studio (vite preview)
#   ./scripts/run.sh --no-open   do not open a browser
# Ports come from .env (VERONICA_API_PORT, VERONICA_WEB_PORT).
set -euo pipefail
. "$(dirname "$0")/lib.sh"

PROD=0; OPEN=1
for a in "$@"; do case "$a" in
  --prod) PROD=1 ;;
  --no-open) OPEN=0 ;;
  -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
  *) die "unknown option $a" ;;
esac; done

banner
load_env
API_PORT="${VERONICA_API_PORT:-8787}"
WEB_PORT="${VERONICA_WEB_PORT:-5173}"
[ -x "$ROOT/backend/.venv/bin/uvicorn" ] || die "Dependencies missing — run ./scripts/install.sh first."
[ -d "$ROOT/frontend/node_modules" ] || die "Frontend dependencies missing — run ./scripts/install.sh first."
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
port_busy "$API_PORT" && die "Port $API_PORT is in use (set VERONICA_API_PORT in .env)"
port_busy "$WEB_PORT" && die "Port $WEB_PORT is in use (set VERONICA_WEB_PORT in .env)"

LOGS="$ROOT/.logs"; mkdir -p "$LOGS"
PIDS=()
cleanup() {
  trap - INT TERM EXIT
  echo; say "Stopping…"
  for p in ${PIDS[@]+"${PIDS[@]}"}; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
  ok "Stopped."
  exit 0
}
trap cleanup INT TERM EXIT

say "Starting API on :$API_PORT (log: .logs/api.log)"
( cd "$ROOT/backend" && exec ./.venv/bin/uvicorn veronica.api:app --host 127.0.0.1 --port "$API_PORT" ) >"$LOGS/api.log" 2>&1 &
PIDS+=($!)

for i in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1; then break; fi
  kill -0 "${PIDS[0]}" 2>/dev/null || { tail -n 30 "$LOGS/api.log" >&2; die "API failed to start"; }
  sleep 1
done
curl -fsS "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1 || die "API did not become healthy (see .logs/api.log)"
DEVICE="$(curl -fsS "http://127.0.0.1:$API_PORT/api/health" | sed -n 's/.*"device":"\([a-z]*\)".*/\1/p')"
ok "API healthy (inference device: ${DEVICE:-?})"

cd "$ROOT/frontend"
export VERONICA_API="http://127.0.0.1:$API_PORT"
if [ "$PROD" = 1 ]; then
  say "Building studio…"
  npm run build >"$LOGS/web-build.log" 2>&1 || { tail -n 30 "$LOGS/web-build.log" >&2; die "build failed"; }
  ( exec npx vite preview --port "$WEB_PORT" --strictPort --host 127.0.0.1 ) >"$LOGS/web.log" 2>&1 &
else
  ( exec npx vite --port "$WEB_PORT" --strictPort --host 127.0.0.1 ) >"$LOGS/web.log" 2>&1 &
fi
PIDS+=($!)
for i in $(seq 1 60); do curl -fsS "http://127.0.0.1:$WEB_PORT/" >/dev/null 2>&1 && break; sleep 0.5; done
curl -fsS "http://127.0.0.1:$WEB_PORT/" >/dev/null 2>&1 || { tail -n 30 "$LOGS/web.log" >&2; die "studio failed to start"; }

URL="http://localhost:$WEB_PORT"
ok "VERONICA is running → ${B}$URL${N}   (Ctrl-C to stop)"
if [ "$OPEN" = 1 ]; then
  if have open; then open "$URL"; elif have xdg-open; then xdg-open "$URL" >/dev/null 2>&1 || true; fi
fi
# macOS ships bash 3.2 (no `wait -n`): poll until either server exits
while kill -0 "${PIDS[0]}" 2>/dev/null && kill -0 "${PIDS[1]}" 2>/dev/null; do sleep 1; done
warn "A server exited — see .logs/"
