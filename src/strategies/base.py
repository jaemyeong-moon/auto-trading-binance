"""Base strategy interface."""

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd

from src.core.models import Signal


class ExecutionMode(str, Enum):
    ALWAYS_FLIP = "always_flip"    # 항상 포지션 보유, 방향 전환 시 플립
    SIGNAL_ONLY = "signal_only"    # 신호 있을 때만 진입, 대기 가능


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def label(self) -> str:
        return self.name

    @property
    def description(self) -> str:
        return ""

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.ALWAYS_FLIP

    @abstractmethod
    def evaluate(self, symbol: str, candles: pd.DataFrame,
                 htf_candles: pd.DataFrame | None = None) -> Signal:
        """매매 신호 평가.

        Args:
            symbol: 심볼
            candles: 1분봉 (주 타임프레임)
            htf_candles: 상위 타임프레임 캔들 (15분봉 등, 선택)
        """
        ...

    def record_result(self, pnl: float) -> None:
        pass
