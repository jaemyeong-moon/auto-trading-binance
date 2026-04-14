"""Shared test fixtures for auto-trader test suite."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.core.models import Signal, SignalType


# ── Candle Generators ──


def make_candles(
    prices: list[float],
    volumes: list[float] | None = None,
    freq: str = "min",
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Create OHLCV DataFrame from close prices.

    Works as both a helper function and the foundation for fixtures.
    """
    n = len(prices)
    if volumes is None:
        volumes = [5000.0] * n
    return pd.DataFrame(
        {
            "open": [prices[0]] + prices[:-1],
            "high": [p * 1.005 for p in prices],
            "low": [p * 0.995 for p in prices],
            "close": prices,
            "volume": volumes,
        },
        index=pd.date_range(start, periods=n, freq=freq),
    )


def make_candles_random(
    n: int = 300,
    seed: int = 42,
    base: float = 100.0,
    volatility: float = 0.02,
    freq: str = "min",
) -> pd.DataFrame:
    """Create random walk candles for backtesting."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, volatility, n)
    prices = base * np.cumprod(1 + returns)
    volumes = rng.uniform(1000, 10000, n).tolist()
    return make_candles(prices.tolist(), volumes, freq=freq)


def prices_trending_up(n: int = 220, seed: int = 42) -> list[float]:
    """Uptrend series suitable for EMA50 > EMA200 after warm-up."""
    np.random.seed(seed)
    return list(100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1)


def prices_trending_down(n: int = 220, seed: int = 42) -> list[float]:
    """Downtrend series: EMA50 < EMA200."""
    np.random.seed(seed)
    return list(200 - np.arange(n) * 0.5 + np.random.randn(n) * 0.1)


def prices_flat(n: int = 100, seed: int = 42) -> list[float]:
    """Flat / ranging market."""
    np.random.seed(seed)
    return list(100 + np.random.randn(n) * 0.01)


# ── Pytest Fixtures ──


@pytest.fixture
def sample_candles_1m() -> pd.DataFrame:
    """100-bar 1-minute candle set (flat market)."""
    return make_candles(prices_flat(100))


@pytest.fixture
def sample_candles_15m() -> pd.DataFrame:
    """220-bar 15-minute candle set (uptrend for HTF)."""
    return make_candles(prices_trending_up(220), freq="15min")


@pytest.fixture
def sample_candles_15m_down() -> pd.DataFrame:
    """220-bar 15-minute candle set (downtrend for HTF)."""
    return make_candles(prices_trending_down(220), freq="15min")


# ── Mock FuturesClient ──


@pytest.fixture
def mock_futures_client() -> AsyncMock:
    """Fully mocked FuturesClient with sensible defaults.

    All async methods return reasonable values. Override specific
    methods in your test as needed.
    """
    client = AsyncMock()

    # Connection
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.sync_time = AsyncMock()

    # Market data
    client.get_candles = AsyncMock(return_value=make_candles(prices_flat(200)))
    client.get_price = AsyncMock(return_value=100.0)

    # Account
    client.get_balance = AsyncMock(return_value=1000.0)
    client.get_account_summary = AsyncMock(return_value={
        "balance": 1000.0,
        "available": 950.0,
        "unrealized_pnl": 0.0,
    })

    # Position
    client.get_position = AsyncMock(return_value=None)

    # Order execution
    client.open_long = AsyncMock(return_value={"orderId": 12345})
    client.open_short = AsyncMock(return_value={"orderId": 12346})
    client.close_long = AsyncMock(return_value={"orderId": 12347})
    client.close_short = AsyncMock(return_value={"orderId": 12348})
    client.place_sl_tp_orders = AsyncMock(return_value={
        "sl_order": 99001, "tp_order": 99002,
    })
    client.cancel_open_orders = AsyncMock()

    # Leverage
    client.set_leverage = AsyncMock()

    # Fees
    client.get_recent_fees = AsyncMock(return_value=0.05)

    return client


# ── Trading Config ──


@pytest.fixture
def trading_config() -> dict:
    """Minimal trading config for tests (dict form, not Pydantic)."""
    return {
        "symbols": ["BTCUSDT"],
        "base_currency": "USDT",
        "interval": "15m",
        "max_open_positions": 3,
        "position_size_pct": 0.20,
    }


# ── Temporary Database ──


@pytest.fixture
def tmp_db(tmp_path: Path):
    """Create a temporary SQLite DB with auto-trader schema.

    Returns (engine, SessionLocal) tuple.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.core.database import Base

    db_path = tmp_path / "test_trades.db"
    url = f"sqlite:///{db_path}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return engine, SessionLocal


# ── Signal Helpers ──


@pytest.fixture
def make_signal():
    """Factory fixture for creating Signal objects."""
    def _make(
        symbol: str = "BTCUSDT",
        signal_type: SignalType = SignalType.HOLD,
        confidence: float = 0.5,
        source: str = "test_strategy",
        **metadata,
    ) -> Signal:
        return Signal(
            symbol=symbol,
            type=signal_type,
            confidence=confidence,
            source=source,
            metadata=metadata,
        )
    return _make
