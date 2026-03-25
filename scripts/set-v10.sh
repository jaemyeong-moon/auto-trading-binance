#!/usr/bin/env bash
# v10 역추세 전략으로 전환
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$APP_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "가상환경이 없습니다. setup.sh를 먼저 실행하세요."
    exit 1
fi

# .env 로드
if [ -f "$APP_DIR/.env.real" ]; then
    set -a; source "$APP_DIR/.env.real"; set +a
elif [ -f "$APP_DIR/.env" ]; then
    set -a; source "$APP_DIR/.env"; set +a
fi

cd "$APP_DIR"

echo "=== v10 Contrarian Scalper 설정 ==="

"$VENV_DIR/bin/python3" -c "
from src.core import database as db
db.init_db()

# 전략 전환
db.set_setting('strategy', 'contrarian_scalper')

# v10 권장 설정
db.set_setting('leverage', '5')
db.set_setting('tick_interval', '15')

print('설정 완료:')
s = db.get_all_settings()
print(f'  전략: {s[\"strategy\"]}')
print(f'  레버리지: x{s[\"leverage\"]}')
print(f'  분석주기: {s[\"tick_interval\"]}초')
print()
print('봇 재시작 필요: ./scripts/stop.sh && ./scripts/run.sh --daemon')
"
