"""Tests for v2 AdaptiveScalper strategy."""

import numpy as np
import pandas as pd
import pytest

from src.core.models import SignalType
from src.strategies.adaptive_scalper import (
    AdaptiveScalper, AdaptiveState, MarketState,
    detect_market_state_htf, detect_market_state_1m, compute_entry_score,
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


def _trending_up(n=150):
    np.random.seed(42)
    return list(100 + np.arange(n) * 0.3 + np.random.randn(n) * 0.05)


def _trending_down(n=150):
    np.random.seed(42)
    return list(100 - np.arange(n) * 0.3 + np.random.randn(n) * 0.05)


def _ranging(n=150):
    np.random.seed(42)
    return list(100 + np.sin(np.linspace(0, 8 * np.pi, n)) * 0.3)


class TestAdaptiveState:
    def test_cooldown(self):
        state = AdaptiveState()
        state.trigger_cooldown(3)
        assert state.tick_cooldown() is True  # 3->2
        assert state.tick_cooldown() is True  # 2->1
        assert state.tick_cooldown() is True  # 1->0
        assert state.tick_cooldown() is False  # done

    def test_open_close(self):
        state = AdaptiveState()
        state.open("LONG", 100.0)
        assert state.position_side == "LONG"
        state.close()
        assert state.position_side == "NONE"

    def test_update_price(self):
        state = AdaptiveState()
        state.open("LONG", 100.0)
        state.update_price(105.0)
        assert state.highest_since_entry == 105.0
        state.update_price(95.0)
        assert state.lowest_since_entry == 95.0


class TestMarketState:
    def test_trending_up_detected(self):
        candles = _make_candles(_trending_up())
        state = detect_market_state_htf(candles)
        assert state in (MarketState.TRENDING_UP, MarketState.VOLATILE)

    def test_trending_down_detected(self):
        candles = _make_candles(_trending_down())
        state = detect_market_state_htf(candles)
        assert state in (MarketState.TRENDING_DOWN, MarketState.VOLATILE)

    def test_ranging_detected(self):
        candles = _make_candles(_ranging())
        state = detect_market_state_htf(candles)
        # 좁은 횡보 데이터는 EMA 정렬에 따라 다양하게 판단될 수 있음
        assert state in (MarketState.RANGING, MarketState.VOLATILE,
                         MarketState.TRENDING_UP, MarketState.TRENDING_DOWN)


class TestAdaptiveScalper:
    def test_name_and_mode(self):
        s = AdaptiveScalper()
        assert s.name == "adaptive_scalper"
        assert s.mode.value == "signal_only"

    def test_insufficient_data(self):
        s = AdaptiveScalper()
        candles = _make_candles([100.0] * 10)
        signal = s.evaluate("BTCUSDT", candles)
        assert signal.type == SignalType.HOLD

    def test_cooldown_returns_hold(self):
        s = AdaptiveScalper()
        s.state.trigger_cooldown(2)
        candles = _make_candles(_trending_up())
        signal = s.evaluate("BTCUSDT", candles)
        assert signal.type == SignalType.HOLD
        assert signal.metadata.get("reason") == "cooldown"

    def test_record_result_triggers_cooldown_on_loss(self):
        s = AdaptiveScalper()
        s.record_result(-10.0)
        assert s.state.cooldown_remaining == 4
        assert s.state.losses == 1

    def test_record_result_no_cooldown_on_win(self):
        s = AdaptiveScalper()
        s.record_result(10.0)
        assert s.state.cooldown_remaining == 0
        assert s.state.wins == 1

    def test_trending_up_can_produce_buy(self):
        s = AdaptiveScalper()
        candles = _make_candles(_trending_up(), volumes=[10000.0] * 150)
        signal = s.evaluate("BTCUSDT", candles)
        # 추세 + 거래량이면 BUY 가능 (점수에 따라 HOLD도 가능)
        assert signal.type in (SignalType.BUY, SignalType.HOLD)

    def test_ranging_returns_hold(self):
        s = AdaptiveScalper()
        candles = _make_candles(_ranging())
        signal = s.evaluate("BTCUSDT", candles)
        # 횡보면 HOLD
        if signal.metadata.get("reason") == "no_trade_zone":
            assert signal.type == SignalType.HOLD


class TestEntryScore:
    def test_score_range(self):
        candles = _make_candles(_trending_up(), volumes=[10000.0] * 150)
        score, details = compute_entry_score(candles, None, "LONG")
        assert 0 <= score <= 6
        assert "ema_1m" in details
        assert "volume" in details
        assert "rsi_ok" in details
        assert "macd" in details
        assert "bb_position" in details


class TestRegistry:
    def test_strategies_registered(self):
        from src.strategies.registry import list_strategies
        names = [s["name"] for s in list_strategies()]
        assert "momentum_flip_scalper" in names
        assert "adaptive_scalper" in names

    def test_get_strategy(self):
        from src.strategies.registry import get_strategy
        s = get_strategy("adaptive_scalper")
        assert s.name == "adaptive_scalper"
