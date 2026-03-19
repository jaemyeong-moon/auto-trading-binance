"""Shared domain models."""

from datetime import datetime
from enum import Enum

from src.utils.timezone import now_kst

from pydantic import BaseModel


class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"  # 포지션 청산 (재진입 없음)


class Signal(BaseModel):
    symbol: str
    type: SignalType
    confidence: float  # 0.0 to 1.0
    source: str  # strategy name that generated it
    timestamp: datetime = now_kst()
    metadata: dict = {}


class Position(BaseModel):
    symbol: str
    side: str  # BUY or SELL
    entry_price: float
    quantity: float
    opened_at: datetime
    stop_loss: float | None = None
    take_profit: float | None = None


class Trade(BaseModel):
    symbol: str
    side: str
    entry_price: float
    exit_price: float | None = None
    quantity: float
    pnl: float | None = None
    opened_at: datetime
    closed_at: datetime | None = None
