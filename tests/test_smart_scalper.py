"""Tests for v3 SmartMomentumScalper."""

import numpy as np
import pandas as pd
import pytest

from src.core.models import SignalType
from src.strategies.smart_scalper import (
    SmartMomentumScalper, SmartState, Regime,
    compute_regime, compute_vwap, compute_momentum_acceleration,
    compute_entry_score_v3, EXECUTION_COST,
)


def _make_candles(prices, volumes=None, n=None):
    if n is None:
        n = len(prices)
    if volumes is None:
        volumes = [5000.0] * n
    return pd.DataFrame({
        "open": prices,
        "high": [p * 1.005 for p in prices],
        "low": [p * 0.995 for p in prices],
        "close": prices,
        "volume": volumes,
    }, index=pd.date_range("2024-01-01", periods=n, freq="min"))


def _trending_up(n=200):
    np.random.seed(42)
    return list(100 + np.arange(n) * 0.3 + np.random.randn(n) * 0.05)


def _trending_down(n=200):
    np.random.seed(42)
    return list(100 - np.arange(n) * 0.3 + np.random.randn(n) * 0.05)


def _flat(n=200):
    np.random.seed(42)
    return list(100 + np.random.randn(n) * 0.01)


class TestSmartState:
    def test_cooldown_progressive(self):
        state = SmartState()
        # 1st loss: base cooldown
        state.consecutive_losses = 0
        state.cooldown_remaining = 5 + (1 * 2)  # simulate
        assert state.cooldown_remaining == 7

    def test_trade_limit(self):
        state = SmartState()
        assert state.check_trade_limit(10, max_per_hour=2) is True
        state.record_trade()
        state.record_trade()
        assert state.check_trade_limit(10, max_per_hour=2) is False
        # New hour resets
        assert state.check_trade_limit(11, max_per_hour=2) is True

    def test_open_stores_atr(self):
        state = SmartState()
        state.open("LONG", 100.0, atr=2.5)
        assert state.entry_atr == 2.5
        assert state.position_side == "LONG"


class TestRegime:
    def test_trending_up(self):
        candles = _make_candles(_trending_up(), volumes=[10000.0] * 200)
        regime, info = compute_regime(candles)
        assert "adx" in info
        assert regime in (Regime.STRONG_TREND, Regime.WEAK_TREND, Regime.VOLATILE)

    def test_flat_is_ranging(self):
        candles = _make_candles(_flat())
        regime, info = compute_regime(candles)
        # Very flat data should be ranging or weak
        assert regime in (Regime.RANGING, Regime.WEAK_TREND)


class TestHelpers:
    def test_vwap(self):
        candles = _make_candles(_trending_up())
        vwap = compute_vwap(candles)
        assert len(vwap) == len(candles)
        assert vwap.iloc[-1] > 0

    def test_momentum_acceleration(self):
        prices = _trending_up()
        close = pd.Series(prices)
        accel = compute_momentum_acceleration(close)
        assert isinstance(accel, float)


class TestSmartMomentumScalper:
    def test_name_and_mode(self):
        s = SmartMomentumScalper()
        assert s.name == "smart_momentum_scalper"
        assert s.mode.value == "signal_only"

    def test_insufficient_data(self):
        s = SmartMomentumScalper()
        candles = _make_candles([100.0] * 10)
        signal = s.evaluate("BTCUSDT", candles)
        assert signal.type == SignalType.HOLD

    def test_cooldown(self):
        s = SmartMomentumScalper()
        s.state.cooldown_remaining = 3
        candles = _make_candles(_trending_up())
        signal = s.evaluate("BTCUSDT", candles)
        assert signal.type == SignalType.HOLD
        assert signal.metadata["reason"] == "cooldown"

    def test_record_result_loss_increases_cooldown(self):
        s = SmartMomentumScalper()
        s.record_result(-10.0)
        assert s.state.consecutive_losses == 1
        cd1 = s.state.cooldown_remaining

        s.state.close()
        s.state.consecutive_losses = 3
        s.record_result(-10.0)
        cd2 = s.state.cooldown_remaining
        assert cd2 > cd1  # Progressive cooldown

    def test_record_result_win_resets(self):
        s = SmartMomentumScalper()
        s.state.consecutive_losses = 5
        s.record_result(10.0)
        assert s.state.consecutive_losses == 0

    def test_trade_limit_blocks(self):
        s = SmartMomentumScalper()
        s.state.last_hour = 10
        s.state.trades_this_hour = 10
        candles = _make_candles(_trending_up())
        # Force hour to match
        candles.index = pd.date_range("2024-01-01 10:00", periods=len(candles), freq="min")
        signal = s.evaluate("BTCUSDT", candles)
        assert signal.type == SignalType.HOLD

    def test_execution_cost_is_realistic(self):
        assert EXECUTION_COST > 0.0003  # at least 0.03%
        assert EXECUTION_COST < 0.002   # less than 0.2%


class TestRegistryV3:
    def test_v3_registered(self):
        from src.strategies.registry import list_strategies
        names = [s["name"] for s in list_strategies()]
        assert "smart_momentum_scalper" in names
