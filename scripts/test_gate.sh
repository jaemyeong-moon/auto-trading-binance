#!/usr/bin/env bash
# 테스트 게이트 — push 전 자동 검증
set -e

# 프로젝트 루트 기준으로 .venv 활성화 (이미 활성화된 경우 스킵)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

echo "=== Lint Check ==="
ruff check src/ tests/

echo "=== Tests ==="
python -m pytest tests/ -q --tb=short

echo "=== All checks passed ==="
