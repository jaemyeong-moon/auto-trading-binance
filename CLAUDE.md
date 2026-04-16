# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automated cryptocurrency trading bot for Binance. Python 3.11+, async-first architecture using python-binance, pandas, and structlog. Supports technical analysis (RSI, MACD, Bollinger Bands) and ML-based (XGBoost) strategies, plus a combined mode that merges both with weighted voting.

## Commands

```bash
# Install
pip install -e ".[dev]"           # core + dev tools
pip install -e ".[ml]"            # ML dependencies (torch, xgboost, optuna)
pip install -e ".[dashboard]"     # fastapi + uvicorn (mobile dashboard)
pip install -e ".[dev,ml,dashboard]"  # everything

# Run
python -m src.main                # or: auto-trader (after install)

# Mobile Dashboard
python -m src.dashboard.mobile    # FastAPI 모바일 대시보드 (port 8502)

# Test
pytest                         # all tests
pytest tests/test_models.py    # single file
pytest -k "test_signal"        # by name pattern
pytest --cov=src               # with coverage

# Lint & type check
ruff check src/ tests/         # lint
ruff check --fix src/ tests/   # autofix
mypy src/                      # type check
```

## Architecture

The trading loop follows a pipeline: **Exchange -> Strategy -> Signal -> Engine -> Order**.

- **`src/core/engine.py`** — `TradingEngine` runs the main async loop. Each tick fetches candles for configured symbols, feeds them to the active strategy, and executes resulting signals.
- **`src/core/config.py`** — Pydantic Settings that merge `config/settings.yaml` (committed defaults) with `.env` (secrets) and optional `config/settings.local.yaml` (local overrides, gitignored).
- **`src/core/models.py`** — Shared domain types: `Signal`, `Position`, `Trade`. Signals carry a `confidence` (0-1) and `source` field.
- **`src/exchange/binance_client.py`** — Async Binance wrapper. All exchange I/O goes through this single client. Always uses testnet by default.
- **`src/strategies/base.py`** — `Strategy` ABC with `evaluate(symbol, candles) -> Signal`.
- **`src/strategies/technical.py`** — RSI + MACD + Bollinger Bands voting (2-of-3 agreement triggers signal).
- **`src/strategies/ml_strategy.py`** — XGBoost classifier. `train()` to build model, `load_model()` to restore. Models saved to `models/` as joblib files.
- **`src/strategies/combined.py`** — Weighted merge of technical (0.4) and ML (0.6) scores. Threshold of ±0.3 for action.
- **`src/backtesting/backtest.py`** — `Backtester.run(symbol, candles) -> BacktestResult`. Simulates strategy with commission, stop-loss/take-profit, and produces equity curve + trade list.
- **`src/notifications/notifier.py`** — `TelegramNotifier` (requires bot token + chat ID) and `ConsoleNotifier` fallback. Both implement `notify_signal()` and `notify_trade()`.
- **`src/dashboard/mobile.py`** — FastAPI 모바일 대시보드 (PWA). 실시간 현황, 거래 내역, 설정, 전략 판단 로그 등. `python -m src.dashboard.mobile`로 실행.
- **`src/core/database.py`** — SQLAlchemy trade persistence. `TradeRecord` table, `save_trade()` helper, `init_db()` to create tables.

## Configuration

Settings load order (later overrides earlier):
1. `config/settings.yaml` — checked-in defaults
2. `config/settings.local.yaml` — local overrides (gitignored)
3. `.env` — API keys and secrets (gitignored)

Key env vars: `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `BINANCE_TESTNET`, `DATABASE_URL`, `LOG_LEVEL`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`.

## AI Strategy Agent

- **`src/core/strategy_agent.py`** — 멀티 LLM 기반 자동 전략 생성/평가 에이전트. 주기적으로 트레이딩 성과를 분석하고, 성과 부진 시 신규 전략 코드를 LLM이 작성 → 구문 검증 → 보안 검증 → 백테스트 → 동적 로드 → 핫 스왑.
- **`src/core/llm_provider.py`** — LLM 추상화 레이어. Anthropic(Claude), OpenAI(GPT), Google(Gemini) 지원. `.env`에 API 키를 넣으면 자동 감지.
- **`src/strategies/ai_generated/`** — AI가 생성한 전략 파일 저장 디렉토리. 최대 5개 유지, 오래된 것부터 자동 정리.
- 활성화: `.env`에 `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` 중 하나 설정. `pip install -e ".[agent]"`.
- AI Agent는 FuturesEngine의 메인 루프에서 약 1시간 주기로 실행됨 (`AI_AGENT_INTERVAL_TICKS`).
- 생성된 전략은 `@register` 데코레이터로 레지스트리에 자동 등록, 엔진에서 핫 스왑.

## Conventions

- All exchange interaction is async (`await client.method()`).
- New strategies must subclass `Strategy` (in `src/strategies/base.py`) and implement `name` and `evaluate`.
- Signals always include `confidence` (0.0-1.0) so consumers can threshold.
- Testnet mode is on by default — never set `testnet: false` without explicit user intent.
