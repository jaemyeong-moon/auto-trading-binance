"""Unit tests for v13 OrderFlow Scalper strategy."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.core.models import SignalType


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_candles(prices: list[float]) -> pd.DataFrame:
    """Create OHLCV DataFrame from close prices."""
    n = len(prices)
    return pd.DataFrame(
        {
            "open": [prices[0]] + prices[:-1],
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "close": prices,
            "volume": [5000.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="min"),
    )


def _trending_candles(direction: str = "up", n: int = 100) -> pd.DataFrame:
    """상승/하락 추세 캔들 생성 (EMA9 > EMA21 또는 반대)."""
    if direction == "up":
        prices = list(np.linspace(100, 120, n))
    else:
        prices = list(np.linspace(120, 100, n))
    return _make_candles(prices)


def _inject_orderbook(df: pd.DataFrame, ratio: float = 1.0) -> pd.DataFrame:
    """candles.attrs 에 오더북 데이터 주입.

    ratio = bid_total / ask_total 가 되도록 설정.
    """
    df.attrs["orderbook"] = {
        "bids": [[100.0, ratio * 100.0]],
        "asks": [[100.1, 100.0]],
    }
    return df


def _inject_funding(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    """candles.attrs 에 펀딩비 데이터 주입."""
    df.attrs["funding_rates"] = [rate]
    return df


def _inject_oi(df: pd.DataFrame, current: float, prev: float) -> pd.DataFrame:
    """candles.attrs 에 OI 데이터 주입."""
    df.attrs["oi_current"] = current
    df.attrs["oi_prev"] = prev
    return df


# ── 1. 전략 메타 속성 테스트 ───────────────────────────────────────────────

class TestOrderFlowScalperMeta:
    def setup_method(self):
        # 임포트는 테스트에서 직접 수행하여 레지스트리 부작용 최소화
        from src.strategies.orderflow_v13 import OrderFlowScalper
        self.strategy = OrderFlowScalper()

    def test_name(self):
        assert self.strategy.name == "orderflow_v13"

    def test_timeframes(self):
        assert self.strategy.TIMEFRAMES == ["5m", "15m", "1h"]

    def test_mode_is_signal_only(self):
        from src.strategies.base import ExecutionMode
        assert self.strategy.mode == ExecutionMode.SIGNAL_ONLY

    def test_leverage_and_params(self):
        assert self.strategy.LEVERAGE == 5
        assert self.strategy.POSITION_SIZE_PCT == 0.15
        assert self.strategy.MAX_HOLD_HOURS == 6.0
        assert self.strategy.SL_ATR_MULT == 2.0
        assert self.strategy.TP_ATR_MULT == 4.0


# ── 2. 데이터 부족 시 HOLD ────────────────────────────────────────────────

class TestInsufficientData:
    def setup_method(self):
        from src.strategies.orderflow_v13 import OrderFlowScalper
        from src.core.models import SignalType
        self.strategy = OrderFlowScalper()
        self.SignalType = SignalType

    def test_hold_when_fewer_than_50_candles(self):
        small_candles = _make_candles([100.0] * 30)
        signal = self.strategy.evaluate("BTCUSDT", small_candles)
        assert signal.type == self.SignalType.HOLD
        assert signal.metadata.get("reason") == "insufficient_data"

    def test_hold_with_zero_candles(self):
        empty = _make_candles([100.0] * 5)
        signal = self.strategy.evaluate("BTCUSDT", empty)
        assert signal.type == self.SignalType.HOLD


# ── 3. BUY 신호 생성 (오더북 LONG 편향 + 상승추세 + 펀딩 SHORT 극단치) ──

class TestBuySignalGeneration:
    def setup_method(self):
        from src.strategies.orderflow_v13 import OrderFlowScalper
        from src.core.models import SignalType
        self.strategy = OrderFlowScalper()
        self.SignalType = SignalType

    def test_buy_signal_with_long_orderbook_and_uptrend(self):
        """LONG 조건: 상승추세(tf_score≥1) + 오더북 bid 우세(ob_score=2)."""
        candles = _trending_candles("up", 100)
        # bid:ask = 1.5 > 1.2 → ob_score=2
        _inject_orderbook(candles, ratio=1.5)
        # 펀딩비 극단적 양수 → SHORT 역방향 신호 (LONG 엔트리와 일치 안함) → fr_score=0
        # 단, tf+ob 만으로도 임계치(3점) 도달 가능한지 확인
        # tf_score=1(또는2), ob_score=2 → 합산 ≥3 → 진입
        signal = self.strategy.evaluate("BTCUSDT", candles)
        # 상승추세 + ob_score=2 로 total_score≥3 → BUY
        assert signal.type == self.SignalType.BUY
        assert signal.confidence > 0
        assert signal.metadata["direction"] == "LONG"

    def test_buy_signal_metadata_keys(self):
        """BUY 신호 metadata 에 필수 키가 존재하는지 확인."""
        candles = _trending_candles("up", 100)
        _inject_orderbook(candles, ratio=1.5)
        signal = self.strategy.evaluate("BTCUSDT", candles)
        if signal.type == self.SignalType.BUY:
            required_keys = {"direction", "total_score", "tf_score", "ob_score",
                             "fr_score", "oi_score", "ob_ratio", "sl_price", "tp_price", "atr"}
            assert required_keys.issubset(signal.metadata.keys())


# ── 4. SELL 신호 생성 (오더북 SHORT 편향 + 하락추세) ─────────────────────

class TestSellSignalGeneration:
    def setup_method(self):
        from src.strategies.orderflow_v13 import OrderFlowScalper
        from src.core.models import SignalType
        self.strategy = OrderFlowScalper()
        self.SignalType = SignalType

    def test_sell_signal_with_short_orderbook_and_downtrend(self):
        """SHORT 조건: 하락추세(tf_score≥1) + 오더북 ask 우세(ob_score=2)."""
        candles = _trending_candles("down", 100)
        # bid:ask = 0.5 < 0.83 → ob_score=2
        _inject_orderbook(candles, ratio=0.5)
        signal = self.strategy.evaluate("BTCUSDT", candles)
        assert signal.type == self.SignalType.SELL
        assert signal.metadata["direction"] == "SHORT"

    def test_sell_signal_sl_above_entry(self):
        """SELL 신호의 SL 가격이 현재가보다 높아야 함."""
        candles = _trending_candles("down", 100)
        _inject_orderbook(candles, ratio=0.5)
        signal = self.strategy.evaluate("BTCUSDT", candles)
        if signal.type == self.SignalType.SELL:
            current_price = float(candles["close"].iloc[-1])
            assert signal.metadata["sl_price"] > current_price
            assert signal.metadata["tp_price"] < current_price


# ── 5. 레지스트리 등록 확인 ──────────────────────────────────────────────

class TestRegistryRegistration:
    def test_registered_in_registry(self):
        """orderflow_v13 가 전략 레지스트리에 등록되어 있는지 확인."""
        from src.strategies.registry import _REGISTRY
        # orderflow_v13 모듈이 임포트되어야 등록됨
        import src.strategies.orderflow_v13  # noqa: F401
        assert "orderflow_v13" in _REGISTRY

    def test_get_strategy_returns_instance(self):
        """get_strategy 로 인스턴스 생성 가능한지 확인."""
        import src.strategies.orderflow_v13  # noqa: F401
        from src.strategies.registry import get_strategy
        strategy = get_strategy("orderflow_v13")
        assert strategy.name == "orderflow_v13"


# ── 6. 점수 부족 시 HOLD ─────────────────────────────────────────────────

class TestLowScoreHold:
    def setup_method(self):
        from src.strategies.orderflow_v13 import OrderFlowScalper
        from src.core.models import SignalType
        self.strategy = OrderFlowScalper()
        self.SignalType = SignalType

    def test_hold_when_score_below_threshold(self):
        """오더북 없고 추세만 있으면 점수 부족 → HOLD."""
        candles = _trending_candles("up", 100)
        # attrs 없음 → ob_score=0, fr_score=0, oi_score=0
        # tf_score=1 → total=1 < 3 → HOLD
        signal = self.strategy.evaluate("BTCUSDT", candles)
        assert signal.type == self.SignalType.HOLD
        assert signal.metadata.get("reason") == "low_score"

    def test_hold_when_orderbook_neutral(self):
        """오더북 ratio 가 LONG/SHORT 임계치 사이 → ob_score=0 → HOLD."""
        candles = _trending_candles("up", 100)
        _inject_orderbook(candles, ratio=1.0)  # 1.0 < 1.2 → no score
        signal = self.strategy.evaluate("BTCUSDT", candles)
        assert signal.type == self.SignalType.HOLD


# ── 7. 펀딩비 + OI 복합 시나리오 ─────────────────────────────────────────

class TestFundingAndOIScoring:
    def setup_method(self):
        from src.strategies.orderflow_v13 import OrderFlowScalper
        from src.core.models import SignalType
        self.strategy = OrderFlowScalper()
        self.SignalType = SignalType

    def test_funding_extreme_adds_score(self):
        """펀딩비 극단적 음수 → LONG 신호 → fr_score=1, 총점 증가."""
        candles = _trending_candles("up", 100)
        _inject_orderbook(candles, ratio=1.5)  # ob_score=2
        _inject_funding(candles, rate=-0.005)  # 극단적 음수 → fr="LONG"
        signal = self.strategy.evaluate("BTCUSDT", candles)
        # tf(1) + ob(2) + fr(1) = 4점 → BUY
        assert signal.type == self.SignalType.BUY
        assert signal.metadata["fr_score"] == 1

    def test_oi_confirm_adds_score(self):
        """OI 증가 + 가격 상승 = BULLISH_CONFIRM → oi_score=1."""
        candles = _trending_candles("up", 100)
        _inject_orderbook(candles, ratio=1.5)
        _inject_oi(candles, current=1100.0, prev=1000.0)  # OI 증가
        signal = self.strategy.evaluate("BTCUSDT", candles)
        if signal.type == self.SignalType.BUY:
            assert signal.metadata["oi_score"] == 1

    def test_htf_alignment_boosts_score(self):
        """HTF 캔들이 같은 방향 → tf_score=3, 임계치 충분히 초과."""
        candles = _trending_candles("up", 100)
        htf = _trending_candles("up", 50)
        # ob 없이도 tf_score=3 이면 임계치(3점) 도달
        signal = self.strategy.evaluate("BTCUSDT", candles, htf_candles=htf)
        assert signal.type == self.SignalType.BUY
        assert signal.metadata["tf_score"] == 3
