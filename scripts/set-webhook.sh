#!/usr/bin/env bash
# 웹훅 설정 CLI 래퍼 (venv 자동 활성화)
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$APP_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "가상환경이 없습니다. 먼저 setup.sh를 실행하세요."
    exit 1
fi
source "$VENV_DIR/bin/activate"

python "$APP_DIR/scripts/set-webhook.py" "$@"
