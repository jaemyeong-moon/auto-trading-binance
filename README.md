# Auto-Trader

Binance 암호화폐 자동매매 봇. 기술적 분석(TA)과 머신러닝(ML) 전략을 결합하여 매매 신호를 생성합니다.

## 기술 스택

| 영역 | 기술 |
|------|------|
| 언어 | Python 3.11+ |
| 거래소 | Binance (python-binance) |
| 데이터 처리 | pandas, numpy |
| 기술적 지표 | ta (Technical Analysis Library) |
| ML | XGBoost, scikit-learn, (선택) PyTorch |
| 설정 관리 | pydantic-settings + YAML |
| DB | SQLAlchemy + aiosqlite (SQLite) |
| 로깅 | structlog + rich |
| 비동기 | asyncio, websockets |

## 설치

```bash
# 기본 설치 (개발 도구 포함)
pip install -e ".[dev]"

# ML 의존성까지 포함
pip install -e ".[dev,ml]"
```

## 설정

### 1단계: API 키 설정

```bash
cp .env.example .env
```

`.env` 파일에 Binance API 키를 입력합니다:

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true
```

### 2단계: 매매 설정

`config/settings.yaml`에서 매매 파라미터를 조정합니다. 로컬 환경에서만 적용할 설정은 `config/settings.local.yaml`을 생성하면 됩니다 (gitignore 대상).

**설정 로드 순서** (나중 것이 우선):
1. `config/settings.yaml` — 기본값 (커밋됨)
2. `config/settings.local.yaml` — 로컬 오버라이드 (gitignore)
3. `.env` — API 키, 시크릿 (gitignore)

### 주요 설정 항목

```yaml
trading:
  symbols: [BTCUSDT, ETHUSDT]   # 매매 대상 심볼
  interval: 1h                   # 캔들 간격
  max_open_positions: 3          # 최대 동시 포지션 수
  position_size_pct: 0.1         # 포트폴리오 대비 포지션 크기 (10%)

risk:
  stop_loss_pct: 0.03            # 손절 (3%)
  take_profit_pct: 0.06          # 익절 (6%)
  max_daily_loss_pct: 0.05       # 일일 최대 손실 (5%)
```

## 실행

```bash
# 봇 실행 (기본: testnet 모드)
python -m src.main

# 또는 패키지 설치 후
auto-trader
```

> **주의**: 기본적으로 testnet 모드로 실행됩니다. 실거래 전환은 `config/settings.yaml`에서 `testnet: false`로 변경해야 합니다.

## 프로젝트 구조

```
auto-trader/
├── src/
│   ├── main.py                  # 애플리케이션 진입점
│   ├── core/
│   │   ├── config.py            # 설정 관리 (YAML + .env 병합)
│   │   ├── engine.py            # 트레이딩 엔진 (메인 루프)
│   │   └── models.py            # 도메인 모델 (Signal, Position, Trade)
│   ├── exchange/
│   │   └── binance_client.py    # Binance 비동기 API 클라이언트
│   ├── strategies/
│   │   ├── base.py              # Strategy 추상 클래스
│   │   ├── technical.py         # 기술적 분석 전략
│   │   ├── ml_strategy.py       # ML(XGBoost) 전략
│   │   └── combined.py          # TA + ML 결합 전략
│   ├── indicators/              # 커스텀 지표 (확장용)
│   ├── models/                  # 학습된 ML 모델 저장
│   └── utils/                   # 유틸리티 (확장용)
├── tests/                       # 테스트
├── config/                      # 설정 파일
├── data/                        # SQLite DB
└── logs/                        # 로그 파일
```

## 아키텍처

### 매매 파이프라인

```
거래소(Binance) → 캔들 데이터 → 전략(Strategy) → 신호(Signal) → 엔진(Engine) → 주문(Order)
```

**TradingEngine** (`src/core/engine.py`)이 전체 흐름을 조율합니다:

1. 설정된 심볼별로 Binance에서 캔들 데이터를 가져옴
2. 활성 전략의 `evaluate()` 메서드에 캔들을 전달
3. 전략이 `Signal` (BUY / SELL / HOLD)을 반환
4. BUY/SELL 신호일 경우 포지션 크기를 계산하고 주문 실행
5. 설정된 interval만큼 대기 후 반복

### 전략 시스템

모든 전략은 `Strategy` ABC를 상속하며, `evaluate(symbol, candles) -> Signal` 메서드를 구현합니다.

#### TechnicalStrategy (기술적 분석)

3개 지표 투표 방식:
- **RSI** (14기간): 30 이하 → 매수, 70 이상 → 매도
- **MACD**: diff > 0 → 매수, diff < 0 → 매도
- **Bollinger Bands**: 하단 접근 → 매수, 상단 접근 → 매도

3개 중 **2개 이상 일치**하면 신호를 발생시킵니다. confidence = 일치 비율.

#### MLStrategy (머신러닝)

XGBoost 분류기를 사용합니다:
- **피처**: 수익률(1/5/10기간), 변동성, RSI, MACD diff, BB 위치, 거래량 비율
- **라벨**: 향후 5기간 수익률 기반 (>2% → BUY, <-2% → SELL, 그 외 → HOLD)
- **임계값**: confidence 0.65 미만이면 HOLD로 처리
- 모델은 `models/xgboost_model.joblib`로 저장/로드

```python
# 학습
strategy = MLStrategy()
strategy.train(candles)

# 로드 후 사용
strategy.load_model()
signal = strategy.evaluate("BTCUSDT", candles)
```

#### CombinedStrategy (결합)

기술적 분석과 ML 신호를 가중 병합합니다:
- 기술적 분석 가중치: **0.4**
- ML 가중치: **0.6**
- 결합 점수 > 0.3 → BUY, < -0.3 → SELL, 그 외 → HOLD

### Signal 모델

모든 전략이 반환하는 공통 데이터 구조:

```python
Signal(
    symbol="BTCUSDT",      # 심볼
    type=SignalType.BUY,    # BUY / SELL / HOLD
    confidence=0.85,        # 신뢰도 (0.0 ~ 1.0)
    source="technical",     # 생성한 전략 이름
    metadata={},            # 전략별 추가 정보
)
```

### Binance 클라이언트

`BinanceClient` (`src/exchange/binance_client.py`)는 모든 거래소 통신을 담당하는 비동기 래퍼입니다.

```python
client = BinanceClient()
await client.connect()

candles = await client.get_candles("BTCUSDT", interval="1h")
price = await client.get_ticker_price("BTCUSDT")
balance = await client.get_balance("USDT")
await client.place_order("BTCUSDT", side="BUY", quantity=0.001)

await client.disconnect()
```

## 테스트

```bash
pytest                          # 전체 테스트
pytest tests/test_models.py     # 특정 파일
pytest -k "test_signal"         # 이름 패턴으로 필터
pytest --cov=src                # 커버리지 포함
```

## 린트 & 타입 체크

```bash
ruff check src/ tests/          # 린트
ruff check --fix src/ tests/    # 자동 수정
mypy src/                       # 타입 체크
```

## 새 전략 추가하기

1. `src/strategies/base.py`의 `Strategy`를 상속
2. `name` 프로퍼티와 `evaluate()` 메서드 구현
3. `src/main.py`의 `_build_strategy()`에 등록

```python
from src.strategies.base import Strategy
from src.core.models import Signal, SignalType

class MyStrategy(Strategy):
    @property
    def name(self) -> str:
        return "my_strategy"

    def evaluate(self, symbol: str, candles: pd.DataFrame) -> Signal:
        # 분석 로직 구현
        return Signal(
            symbol=symbol,
            type=SignalType.BUY,
            confidence=0.8,
            source=self.name,
        )
```
