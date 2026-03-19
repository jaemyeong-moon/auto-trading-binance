"""Tests for backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtest import Backtester, BacktestResult
from src.strategies.technical import TechnicalStrategy


def _make_candles(n: int = 300, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    prices = np.cumsum(np.random.randn(n)) + 100
    prices = np.maximum(prices, 1.0)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.uniform(100, 10000, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


@pytest.fixture
def backtester():
    return Backtester(strategy=TechnicalStrategy(), initial_capital=10000.0)


def test_backtest_returns_result(backtester):
    candles = _make_candles()
    result = backtester.run("BTCUSDT", candles)
    assert isinstance(result, BacktestResult)
    assert result.strategy_name == "technical"
    assert result.symbol == "BTCUSDT"
    assert result.initial_capital == 10000.0


def test_backtest_has_equity_curve(backtester):
    candles = _make_candles()
    result = backtester.run("BTCUSDT", candles)
    assert len(result.equity_curve) > 0
    assert result.equity_curve[0] == 10000.0


def test_backtest_trade_counts_consistent(backtester):
    candles = _make_candles()
    result = backtester.run("BTCUSDT", candles)
    assert result.total_trades == result.winning_trades + result.losing_trades


def test_backtest_insufficient_data(backtester):
    candles = _make_candles(n=10)
    result = backtester.run("BTCUSDT", candles)
    assert result.total_trades == 0
    assert result.final_capital == 10000.0


def test_backtest_max_drawdown_non_negative(backtester):
    candles = _make_candles()
    result = backtester.run("BTCUSDT", candles)
    assert result.max_drawdown_pct >= 0.0
