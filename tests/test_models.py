"""Tests for domain models."""

from datetime import datetime

from src.core.models import Signal, SignalType, Position, Trade


def test_signal_creation():
    signal = Signal(
        symbol="BTCUSDT",
        type=SignalType.BUY,
        confidence=0.85,
        source="technical",
    )
    assert signal.symbol == "BTCUSDT"
    assert signal.type == SignalType.BUY
    assert signal.confidence == 0.85


def test_signal_hold():
    signal = Signal(
        symbol="ETHUSDT",
        type=SignalType.HOLD,
        confidence=0.0,
        source="ml",
    )
    assert signal.type == SignalType.HOLD


def test_position_creation():
    pos = Position(
        symbol="BTCUSDT",
        side="BUY",
        entry_price=50000.0,
        quantity=0.1,
        opened_at=datetime.now(),
        stop_loss=48500.0,
        take_profit=53000.0,
    )
    assert pos.stop_loss == 48500.0


def test_trade_creation():
    trade = Trade(
        symbol="BTCUSDT",
        side="BUY",
        entry_price=50000.0,
        quantity=0.1,
        opened_at=datetime.now(),
    )
    assert trade.exit_price is None
    assert trade.pnl is None
