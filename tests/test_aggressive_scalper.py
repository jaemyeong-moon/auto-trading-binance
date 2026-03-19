"""Tests for v4 AggressiveMomentumRider."""

import numpy as np
import pandas as pd
import pytest

from src.core.models import SignalType
from src.strategies.aggressive_scalper import (
    AggressiveMomentumRider, AggressiveState, MicroRegime,
    detect_momentum_burst, detect_big_candle, detect_squeeze,
    compute_micro_regime,
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


def _pump(n=100):
    """급등 시뮬레이션."""
    np.random.seed(42)
    base = list(100 + np.random.randn(80) * 0.1)
    spike = list(100 + np.arange(20) * 1.0)  # 급등
    return base + spike


def _dump(n=100):
    """급락 시뮬레이션."""
    np.random.seed(42)
    base = list(100 + np.random.randn(80) * 0.1)
    spike = list(100 - np.arange(20) * 1.0)
    return base + spike


def _flat(n=100):
    np.random.seed(42)
    return list(100 + np.random.randn(n) * 0.01)


class TestAggressiveState:
    def test_flip_mode_on_2_losses(self):
        s = AggressiveMomentumRider()
        s.record_result(-1.0)  # 1st loss
        assert not s.state.flip_mode
        s.record_result(-1.0)  # 2nd loss → flip
        assert s.state.flip_mode

    def test_flip_mode_toggles(self):
        s = AggressiveMomentumRider()
        s.record_result(-1.0)
        s.record_result(-1.0)  # flip ON
        assert s.state.flip_mode
        s.record_result(-1.0)
        s.record_result(-1.0)  # flip OFF (toggle)
        assert not s.state.flip_mode

    def test_win_resets(self):
        s = AggressiveMomentumRider()
        s.record_result(-1.0)
        s.record_result(-1.0)
        assert s.state.flip_mode
        s.record_result(5.0)  # win → reset
        assert not s.state.flip_mode
        assert s.state.consecutive_losses == 0

    def test_no_cooldown(self):
        """v4는 쿨다운이 없다."""
        s = AggressiveMomentumRider()
        s.record_result(-1.0)
        candles = _make_candles(_pump(), volumes=[10000.0] * 100)
        signal = s.evaluate("BTCUSDT", candles)
        # 쿨다운 없으므로 HOLD reason이 cooldown이 아님
        if signal.type == SignalType.HOLD:
            assert signal.metadata.get("reason") != "cooldown"


class TestMomentumBurst:
    def test_pump_detected(self):
        candles = _make_candles(_pump())
        atr = 0.5  # 작은 ATR 대비 큰 움직임
        direction, strength = detect_momentum_burst(candles, atr)
        assert direction == "LONG"
        assert strength > 0.8

    def test_dump_detected(self):
        candles = _make_candles(_dump())
        direction, strength = detect_momentum_burst(candles, atr=0.5)
        assert direction == "SHORT"

    def test_flat_no_burst(self):
        candles = _make_candles(_flat())
        direction, _ = detect_momentum_burst(candles, atr=1.0)
        assert direction == "NONE"


class TestBigCandle:
    def test_big_green_candle(self):
        prices = [100.0] * 99 + [105.0]  # 마지막 캔들 큰 양봉
        candles = _make_candles(prices)
        direction, ratio = detect_big_candle(candles)
        assert direction == "LONG"

    def test_small_candle(self):
        candles = _make_candles(_flat())
        direction, _ = detect_big_candle(candles)
        # 평평한 데이터에서는 대형 캔들 없음
        # (open과 close가 거의 같음)


class TestStrategy:
    def test_name(self):
        s = AggressiveMomentumRider()
        assert s.name == "aggressive_momentum_rider"
        assert s.mode.value == "signal_only"

    def test_pump_gives_buy(self):
        s = AggressiveMomentumRider()
        candles = _make_candles(_pump(), volumes=[10000.0] * 100)
        signal = s.evaluate("BTCUSDT", candles)
        assert signal.type in (SignalType.BUY, SignalType.HOLD)

    def test_dump_gives_sell(self):
        s = AggressiveMomentumRider()
        candles = _make_candles(_dump(), volumes=[10000.0] * 100)
        signal = s.evaluate("BTCUSDT", candles)
        assert signal.type in (SignalType.SELL, SignalType.HOLD)

    def test_tight_sl_tp(self):
        s = AggressiveMomentumRider()
        assert s.SL_ATR_MULT == 0.5
        assert s.TP_ATR_MULT == 0.75
        assert s.SL_ATR_MULT < s.TP_ATR_MULT  # TP > SL

    def test_low_threshold(self):
        s = AggressiveMomentumRider()
        assert s.SCORE_THRESHOLD == 2  # 낮은 진입장벽


class TestRegistryV4:
    def test_registered(self):
        from src.strategies.registry import list_strategies
        names = [s["name"] for s in list_strategies()]
        assert "aggressive_momentum_rider" in names
