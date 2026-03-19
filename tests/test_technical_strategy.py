"""Tests for the technical analysis strategy."""

import numpy as np
import pandas as pd
import pytest

from src.core.models import SignalType
from src.strategies.technical import TechnicalStrategy


@pytest.fixture
def strategy():
    return TechnicalStrategy()


def _make_candles(prices: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """Helper to create a candle DataFrame from price series."""
    n = len(prices)
    if volumes is None:
        volumes = [1000.0] * n
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": volumes,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


def test_strategy_name(strategy):
    assert strategy.name == "technical"


def test_evaluate_returns_signal(strategy):
    # Generate enough data for indicators to warm up
    np.random.seed(42)
    prices = list(np.cumsum(np.random.randn(200)) + 100)
    prices = [max(p, 1.0) for p in prices]  # ensure positive
    candles = _make_candles(prices)

    signal = strategy.evaluate("BTCUSDT", candles)
    assert signal.symbol == "BTCUSDT"
    assert signal.type in (SignalType.BUY, SignalType.SELL, SignalType.HOLD)
    assert 0.0 <= signal.confidence <= 1.0
    assert signal.source == "technical"


def test_add_indicators(strategy):
    np.random.seed(42)
    prices = list(np.cumsum(np.random.randn(200)) + 100)
    prices = [max(p, 1.0) for p in prices]
    candles = _make_candles(prices)

    df = strategy._add_indicators(candles)
    assert "rsi" in df.columns
    assert "macd" in df.columns
    assert "macd_diff" in df.columns
    assert "bb_high" in df.columns
    assert "bb_low" in df.columns
