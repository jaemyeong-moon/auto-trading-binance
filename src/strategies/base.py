"""Base strategy interface."""

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd

from src.core.models import Signal


class ExecutionMode(str, Enum):
    ALWAYS_FLIP = "always_flip"    # 항상 포지션 보유, 방향 전환 시 플립
    SIGNAL_ONLY = "signal_only"    # 신호 있을 때만 진입, 대기 가능


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    각 전략은 자체 매매 파라미터를 클래스 속성으로 정의한다.
    엔진은 DB 설정 대신 전략의 속성을 읽어 SL/TP/레버리지/투자비율을 결정.
    """

    # ── 매매 파라미터 (전략별 오버라이드) ──
    LEVERAGE: int = 5                   # 레버리지 배수
    POSITION_SIZE_PCT: float = 0.20     # 잔고 대비 투자 비율
    MAX_HOLD_HOURS: float = 4.0         # 최대 보유 시간 (시간)

    # ATR 기반 SL/TP (0이면 전략 자체 관리)
    SL_ATR_MULT: float = 2.0
    TP_ATR_MULT: float = 4.0
    PARTIAL_TP_ATR_MULT: float = 0.0    # 0이면 부분익절 없음
    TRAILING_ATR_MULT: float = 0.0      # 0이면 트레일링 없음
    TRAILING_DIST_ATR: float = 0.0

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
