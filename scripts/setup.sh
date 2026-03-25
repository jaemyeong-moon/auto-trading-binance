#!/usr/bin/env bash
# Auto-Trader 설치 스크립트 (Ubuntu 1core/1GB)
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$APP_DIR/.venv"

echo "=== Auto-Trader 설치 ==="

# 1) 시스템 패키지
echo "[1/5] 시스템 패키지 설치..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev python3-pip gcc

# 2) swap 설정 (1GB 서버용 — 이미 있으면 스킵)
if [ ! -f /swapfile ]; then
    echo "[2/5] swap 1GB 설정..."
    sudo fallocate -l 1G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null
else
    echo "[2/5] swap 이미 존재 — 스킵"
fi

# 3) venv 생성
echo "[3/5] Python 가상환경 생성..."
python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# 4) 의존성 설치 (torch 제외, xgboost+optuna+mobile 설치)
echo "[4/5] Python 패키지 설치..."
pip install --upgrade pip setuptools wheel -q
pip install -e "$APP_DIR[mobile,agent]" -q
pip install xgboost optuna joblib -q

# 5) 디렉토리 생성
echo "[5/5] 데이터 디렉토리 생성..."
mkdir -p "$APP_DIR/data" "$APP_DIR/logs" "$APP_DIR/models"

# .env 확인
if [ ! -f "$APP_DIR/.env" ] && [ ! -f "$APP_DIR/.env.real" ]; then
    echo ""
    echo "!! .env 파일이 없습니다. .env.example을 복사해서 API 키를 설정하세요:"
    echo "  cp $APP_DIR/.env.example $APP_DIR/.env"
fi

echo ""
echo "=== 설치 완료 ==="
echo "실행: ./scripts/run.sh"
