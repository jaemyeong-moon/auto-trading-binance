#!/usr/bin/env bash
# Auto-Trader 종료 스크립트
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

for pidfile in "$APP_DIR/trader.pid" "$APP_DIR/mobile.pid"; do
    if [ -f "$pidfile" ]; then
        PID=$(cat "$pidfile")
        NAME=$(basename "$pidfile" .pid)
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            echo "[$NAME] 종료 (PID: $PID)"
        else
            echo "[$NAME] 이미 종료됨"
        fi
        rm -f "$pidfile"
    fi
done

echo "완료"
