# Shared helpers for VERONICA scripts (sourced, not executed).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -t 1 ]; then B=$'\e[1m'; C=$'\e[36m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; N=$'\e[0m'; else B= C= G= Y= R= N=; fi
say()  { printf '%s▸%s %s\n' "$C" "$N" "$*"; }
ok()   { printf '%s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '%s!%s %s\n' "$Y" "$N" "$*" >&2; }
die()  { printf '%s✗ %s%s\n' "$R" "$*" "$N" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
# Load .env without overriding variables already set in the environment.
load_env() {
  [ -f "$ROOT/.env" ] || return 0
  local line key val
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|\#*) continue ;; esac
    key="${line%%=*}"; val="${line#*=}"
    case "$key" in *[!A-Za-z0-9_]*|'') continue ;; esac
    [ -n "$val" ] || continue  # blank means "use the default"
    if [ -z "${!key+x}" ]; then export "$key=$val"; fi
  done < "$ROOT/.env"
}
banner() { printf '%s\n' "${B}VERONICA${N} — The Spatial Intelligence System"; }
