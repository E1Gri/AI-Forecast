set -Eeuo pipefail

python -u back/page_updater.py &
UPDATER_PID=$!

python -u -m http.server 8000 \
    --bind 0.0.0.0 \
    --directory front &
SERVER_PID=$!

cleanup() {
    trap - EXIT INT TERM
    kill "$UPDATER_PID" "$SERVER_PID" 2>/dev/null || true
    wait "$UPDATER_PID" "$SERVER_PID" 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait -n "$UPDATER_PID" "$SERVER_PID"