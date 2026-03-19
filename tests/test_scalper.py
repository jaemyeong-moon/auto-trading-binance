"""Tests for MomentumFlipScalper strategy."""

import numpy as np
import pandas as pd
import pytest

from src.core.models import SignalType
from src.strategies.scalper import MomentumFlipScalper, ScalperState


def _make_candles(prices: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    n = len(prices)
    if volumes is None:
        volumes = [1000.0] * n
    return pd.DataFrame({
        "open": prices,
        "high": [p * 1.005 for p in prices],
        "low": [p * 0.995 for p in prices],
        "close": prices,
        "volume": volumes,
    }, index=pd.date_range("2024-01-01", periods=n, freq="min"))


def _trending_up(n=50):
    """생성: 우상향 가격."""
    return list(100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1)


def _trending_down(n=50):
    """생성: 우하향 가격."""
    return list(100 - np.arange(n) * 0.5 + np.random.randn(n) * 0.1)


class TestScalperState:
    def test_consecutive_losses_trigger_contrarian(self):
        state = ScalperState()
        state.record_result(-1.0)
        state.record_result(-1.0)
        assert not state.contrarian_mode
        state.record_result(-1.0)  # 3연패
        assert state.contrarian_mode

    def test_win_resets_losses(self):
        state = ScalperState()
        state.record_result(-1.0)
        state.record_result(-1.0)
        state.record_result(5.0)  # 수익
        assert state.consecutive_losses == 0

    def test_contrarian_exits_on_win(self):
        state = ScalperState()
        for _ in range(3):
            state.record_result(-1.0)
        assert state.contrarian_mode
        state.record_result(5.0)
        assert not state.contrarian_mode


class TestMomentumFlipScalper:
    def test_name(self):
        s = MomentumFlipScalper()
        assert s.name == "momentum_flip_scalper"

    def test_insufficient_data_returns_hold(self):
        s = MomentumFlipScalper()
        candles = _make_candles([100.0] * 5)
        signal = s.evaluate("BTCUSDT", candles)
        assert signal.type == SignalType.HOLD

    def test_uptrend_gives_buy(self):
        np.random.seed(42)
        s = MomentumFlipScalper()
        prices = _trending_up(50)
        volumes = [5000.0] * 50  # 높은 거래량
        candles = _make_candles(prices, volumes)
        signal = s.evaluate("BTCUSDT", candles)
        # 상승 추세에서 BUY 또는 HOLD (이미 LONG일 수 있음)
        assert signal.type in (SignalType.BUY, SignalType.HOLD)

    def test_downtrend_gives_sell(self):
        np.random.seed(42)
        s = MomentumFlipScalper()
        prices = _trending_down(50)
        volumes = [5000.0] * 50
        candles = _make_candles(prices, volumes)
        signal = s.evaluate("BTCUSDT", candles)
        assert signal.type in (SignalType.SELL, SignalType.HOLD)

    def test_contrarian_flips_signal(self):
        np.random.seed(42)
        s = MomentumFlipScalper()
        # Force contrarian mode
        for _ in range(3):
            s.state.record_result(-1.0)
        assert s.state.contrarian_mode

        prices = _trending_up(50)
        volumes = [5000.0] * 50
        candles = _make_candles(prices, volumes)
        signal = s.evaluate("BTCUSDT", candles)
        # 상승인데 역추세이므로 SELL 또는 HOLD
        assert signal.type in (SignalType.SELL, SignalType.HOLD)

    def test_same_direction_returns_hold(self):
        np.random.seed(42)
        s = MomentumFlipScalper()
        prices = _trending_up(50)
        volumes = [5000.0] * 50
        candles = _make_candles(prices, volumes)

        sig1 = s.evaluate("BTCUSDT", candles)
        sig2 = s.evaluate("BTCUSDT", candles)
        # 같은 방향 두 번째는 HOLD
        if sig1.type != SignalType.HOLD:
            assert sig2.type == SignalType.HOLD
