"""Tests for v9 AggressiveMomentumRider (Trend Pullback)."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.core.models import SignalType
from src.strategies.aggressive_scalper import (
    AggressiveMomentumRider, V9State,
    COOLDOWN_WIN, COOLDOWN_LOSS, MAX_DAILY_TRADES,
)


def _make_candles(prices, volumes=None):
    n = len(prices)
    if volumes is None:
        volumes = [5000.0] * n
    return pd.DataFrame({
        "open": [prices[0]] + prices[:-1],
        "high": [p * 1.005 for p in prices],
        "low": [p * 0.995 for p in prices],
        "close": prices,
        "volume": volumes,
    }, index=pd.date_range("2024-01-01", periods=n, freq="min"))


def _trending_up_htf(n=220):
    """15m 상승 추세: EMA50 > EMA200."""
    np.random.seed(42)
    return list(100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1)


def _trending_down_htf(n=220):
    """15m 하락 추세: EMA50 < EMA200."""
    np.random.seed(42)
    return list(200 - np.arange(n) * 0.5 + np.random.randn(n) * 0.1)


def _flat(n=100):
    np.random.seed(42)
    return list(100 + np.random.randn(n) * 0.01)


# ── V9State 단위 테스트 ──


class TestV9State:
    def test_open_and_close(self):
        state = V9State()
        state.open("LONG", 100.0, 98.0, 104.0, 1.0, 0.02, 1.5)
        assert state.position_side == "LONG"
        assert state.entry_price == 100.0
        assert state.sl_price == 98.0
        assert state.tp_price == 104.0
        assert state.entry_atr == 1.5
        state.close()
        assert state.position_side == "NONE"
        assert state.entry_price == 0.0

    def test_update_price(self):
        state = V9State()
        state.open("LONG", 100.0, 98.0, 104.0, 1.0, 0.02, 1.5)
        state.update_price(105.0)
        assert state.highest_since_entry == 105.0
        state.update_price(95.0)
        assert state.lowest_since_entry == 95.0

    def test_daily_trade_tracking(self):
        state = V9State()
        state.daily_trades = MAX_DAILY_TRADES
        assert state.daily_trades >= MAX_DAILY_TRADES


# ── record_result 테스트 ──


class TestRecordResult:
    def test_loss_triggers_cooldown(self):
        s = AggressiveMomentumRider()
        s.record_result(-1.0)
        assert s.state.cooldown_remaining == COOLDOWN_LOSS
        assert s.state.losses == 1

    def test_win_triggers_short_cooldown(self):
        s = AggressiveMomentumRider()
        s.record_result(5.0)
        assert s.state.cooldown_remaining == COOLDOWN_WIN
        assert s.state.wins == 1

    def test_trade_count_increments(self):
        s = AggressiveMomentumRider()
        s.record_result(-1.0)
        s.record_result(5.0)
        assert s.state.total_trades == 2
        assert s.state.daily_trades == 2


# ── evaluate 기본 경로 테스트 ──


class TestEvaluateBasicPaths:
    @patch("src.core.time_filter.is_tradeable_hour", return_value=True)
    def test_no_htf_returns_hold(self, _mock_hour):
        s = AggressiveMomentumRider()
        candles = _make_candles(_flat())
        signal = s.evaluate("BTCUSDT", candles, htf_candles=None)
        assert signal.type == SignalType.HOLD
        assert signal.metadata["reason"] == "insufficient_htf"

    @patch("src.core.time_filter.is_tradeable_hour", return_value=True)
    def test_short_htf_returns_hold(self, _mock_hour):
        s = AggressiveMomentumRider()
        candles = _make_candles(_flat())
        htf = _make_candles([100.0] * 50)
        signal = s.evaluate("BTCUSDT", candles, htf_candles=htf)
        assert signal.type == SignalType.HOLD
        assert signal.metadata["reason"] == "insufficient_htf"

    @patch("src.core.time_filter.is_tradeable_hour", return_value=False)
    def test_blocked_hour(self, _mock_hour):
        s = AggressiveMomentumRider()
        candles = _make_candles(_flat())
        signal = s.evaluate("BTCUSDT", candles)
        assert signal.type == SignalType.HOLD
        assert signal.metadata["reason"] == "blocked_hour"

    @patch("src.core.time_filter.is_tradeable_hour", return_value=True)
    def test_cooldown_returns_hold(self, _mock_hour):
        s = AggressiveMomentumRider()
        s.state.cooldown_remaining = 3
        candles = _make_candles(_flat())
        htf = _make_candles(_trending_up_htf())
        signal = s.evaluate("BTCUSDT", candles, htf_candles=htf)
        assert signal.type == SignalType.HOLD
        assert signal.metadata["reason"] == "cooldown"
        assert s.state.cooldown_remaining == 2  # decremented

    @patch("src.core.time_filter.is_tradeable_hour", return_value=True)
    def test_daily_limit_returns_hold(self, _mock_hour):
        s = AggressiveMomentumRider()
        s.state.daily_trades = MAX_DAILY_TRADES
        from datetime import datetime
        s.state.last_trade_day = datetime.now().day
        candles = _make_candles(_flat())
        htf = _make_candles(_trending_up_htf())
        signal = s.evaluate("BTCUSDT", candles, htf_candles=htf)
        assert signal.type == SignalType.HOLD
        assert signal.metadata["reason"] == "daily_limit"


# ── _manage_position 테스트 ──


class TestManagePosition:
    def test_stop_loss_long(self):
        s = AggressiveMomentumRider()
        s.state.open("LONG", 100.0, 97.0, 106.0, 1.0, 0.01, 1.5)
        prices = [97.0] * 20  # price at SL level
        candles = _make_candles(prices)
        signal = s._manage_position("BTCUSDT", 97.0, candles)
        assert signal.type == SignalType.CLOSE
        assert signal.metadata["reason"] == "stop_loss"

    def test_stop_loss_short(self):
        s = AggressiveMomentumRider()
        s.state.open("SHORT", 100.0, 103.0, 94.0, 1.0, 0.01, 1.5)
        prices = [103.0] * 20
        candles = _make_candles(prices)
        signal = s._manage_position("BTCUSDT", 103.0, candles)
        assert signal.type == SignalType.CLOSE
        assert signal.metadata["reason"] == "stop_loss"

    def test_hold_within_range(self):
        s = AggressiveMomentumRider()
        s.state.open("LONG", 100.0, 97.0, 106.0, 1.0, 0.01, 1.5)
        prices = [101.0] * 20  # within SL/TP range
        candles = _make_candles(prices)
        signal = s._manage_position("BTCUSDT", 101.0, candles)
        assert signal.type in (SignalType.HOLD, SignalType.CLOSE)


# ── _check_momentum 테스트 ──


class TestCheckMomentum:
    def test_rising_momentum_long(self):
        s = AggressiveMomentumRider()
        prices = list(np.linspace(100, 110, 20))
        candles = _make_candles(prices)
        m = s._check_momentum("LONG", candles)
        assert m > 0

    def test_falling_momentum_short(self):
        s = AggressiveMomentumRider()
        prices = list(np.linspace(110, 100, 20))
        candles = _make_candles(prices)
        m = s._check_momentum("SHORT", candles)
        assert m > 0  # negative change but SHORT → positive

    def test_insufficient_data(self):
        s = AggressiveMomentumRider()
        candles = _make_candles([100.0] * 5)
        m = s._check_momentum("LONG", candles)
        assert m == 0.0


# ── Strategy 메타데이터 ──


class TestStrategyMeta:
    def test_name(self):
        s = AggressiveMomentumRider()
        assert s.name == "aggressive_momentum_rider"
        assert s.mode.value == "signal_only"

    def test_constants(self):
        s = AggressiveMomentumRider()
        assert s.LEVERAGE == 5
        assert s.MAX_HOLD_HOURS == 6.0
        # Self-managed SL/TP: engine SL/TP disabled
        assert s.SL_ATR_MULT == 99.0
        assert s.TP_ATR_MULT == 99.0


class TestRegistryV9:
    def test_registered(self):
        from src.strategies.registry import list_strategies
        names = [s["name"] for s in list_strategies()]
        assert "aggressive_momentum_rider" in names
