#!/usr/bin/env bash
# Docker shortcuts for VERONICA.
#   ./scripts/docker.sh up [--gpu] [--completer]   build and start (studio on :8080)
#   ./scripts/docker.sh down | logs | ps | build [--gpu]
# On a Mac, containers cannot use the Apple GPU: inference runs on the CPU.
set -euo pipefail
. "$(dirname "$0")/lib.sh"

have docker || die "Docker is required: https://docs.docker.com/get-docker/"
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is required (docker compose …)"
cd "$ROOT"
[ -f .env ] || { cp .env.example .env; ok "Created .env from .env.example"; }

CMD="${1:-up}"; shift || true
FILES=(-f docker-compose.yml); PROFILES=()
for a in "$@"; do case "$a" in
  --gpu) FILES+=(-f docker-compose.gpu.yml) ;;
  --completer) PROFILES+=(--profile completer) ;;
  *) die "unknown option $a" ;;
esac; done
DC=(docker compose "${FILES[@]}" ${PROFILES[@]+"${PROFILES[@]}"})

case "$CMD" in
  up)
    banner
    [ "$(uname -s)" = "Darwin" ] && warn "Docker on macOS runs inference on the CPU (no Apple GPU). For full speed use ./scripts/run.sh"
    MEM="$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo 0)"
    [ "$MEM" -lt 7500000000 ] && warn "Docker has $((MEM / 1024 / 1024 / 1024)) GB RAM; give it ≥ 8 GB (Docker Desktop → Settings → Resources)."
    load_env
    WEB="${VERONICA_DOCKER_WEB_PORT:-8080}"; API="${VERONICA_API_PORT:-8787}"
    if ! "${DC[@]}" ps --status running --quiet 2>/dev/null | grep -q .; then
      port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
      port_busy "$WEB" && die "Port $WEB is in use — set VERONICA_DOCKER_WEB_PORT in .env (e.g. 8090)"
      port_busy "$API" && die "Port $API is in use — set VERONICA_API_PORT in .env (e.g. 8797)"
    fi
    "${DC[@]}" up --build -d
    ok "VERONICA → http://localhost:$WEB   (first reconstruction downloads ~3.5 GB of models)"
    ;;
  down) "${DC[@]}" down ;;
  logs) "${DC[@]}" logs -f ;;
  ps) "${DC[@]}" ps ;;
  build) "${DC[@]}" build ;;
  *) die "unknown command $CMD (up | down | logs | ps | build)" ;;
esac
