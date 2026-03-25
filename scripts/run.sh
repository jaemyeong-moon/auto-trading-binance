#!/usr/bin/env bash
# Auto-Trader 실행 스크립트 (봇 + 모바일 대시보드)
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$APP_DIR/.venv"
LOG_DIR="$APP_DIR/logs"
BOT_LOG="$LOG_DIR/trader.log"
MOBILE_LOG="$LOG_DIR/mobile.log"
PID_FILE_BOT="$APP_DIR/trader.pid"
PID_FILE_MOBILE="$APP_DIR/mobile.pid"

# venv 활성화
if [ ! -d "$VENV_DIR" ]; then
    echo "가상환경이 없습니다. 먼저 setup.sh를 실행하세요."
    exit 1
fi
source "$VENV_DIR/bin/activate"

# 디렉토리 확보
mkdir -p "$LOG_DIR" "$APP_DIR/data" "$APP_DIR/models"

# .env 로드
if [ -f "$APP_DIR/.env.real" ]; then
    set -a; source "$APP_DIR/.env.real"; set +a
elif [ -f "$APP_DIR/.env" ]; then
    set -a; source "$APP_DIR/.env"; set +a
fi

cd "$APP_DIR"

# ─── 종료 함수 ───
cleanup() {
    echo ""
    echo "종료 중..."
    [ -f "$PID_FILE_BOT" ] && kill "$(cat "$PID_FILE_BOT")" 2>/dev/null && rm -f "$PID_FILE_BOT"
    [ -f "$PID_FILE_MOBILE" ] && kill "$(cat "$PID_FILE_MOBILE")" 2>/dev/null && rm -f "$PID_FILE_MOBILE"
    echo "완료"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ─── 실행 ───
echo "=== Auto-Trader 시작 ==="

if [ "${1:-}" = "--daemon" ]; then
    # 백그라운드: 봇 + 모바일 대시보드 둘 다 데몬으로
    nohup python -m src.main >> "$BOT_LOG" 2>&1 &
    echo $! > "$PID_FILE_BOT"
    echo "[봇] 백그라운드 실행 (PID: $(cat "$PID_FILE_BOT")) | 로그: $BOT_LOG"

    nohup python -m src.dashboard.mobile >> "$MOBILE_LOG" 2>&1 &
    echo $! > "$PID_FILE_MOBILE"
    echo "[모바일] 백그라운드 실행 (PID: $(cat "$PID_FILE_MOBILE")) | http://0.0.0.0:8503"
    echo ""
    echo "중지: ./scripts/stop.sh"

elif [ "${1:-}" = "--bot-only" ]; then
    # 봇만 포그라운드
    echo "[봇] 포그라운드 실행 | 로그: $BOT_LOG"
    python -m src.main 2>&1 | tee -a "$BOT_LOG"

elif [ "${1:-}" = "--mobile-only" ]; then
    # 모바일 대시보드만 포그라운드
    echo "[모바일] 포그라운드 실행 | http://0.0.0.0:8503"
    python -m src.dashboard.mobile 2>&1 | tee -a "$MOBILE_LOG"

else
    # 기본: 봇(백그라운드) + 모바일(포그라운드)
    nohup python -m src.main >> "$BOT_LOG" 2>&1 &
    echo $! > "$PID_FILE_BOT"
    echo "[봇] 백그라운드 실행 (PID: $(cat "$PID_FILE_BOT")) | 로그: $BOT_LOG"
    echo "[모바일] http://0.0.0.0:8503"
    echo "종료: Ctrl+C (봇+모바일 동시 종료)"
    echo ""
    python -m src.dashboard.mobile 2>&1 | tee -a "$MOBILE_LOG"
fi
