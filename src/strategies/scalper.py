"""Momentum Flip Scalper — 항상 포지션 보유, 방향 전환 시 즉시 플립.

핵심 원리:
- EMA(3) vs EMA(8) 크로스로 방향 판단
- 거래량 급증 확인으로 가짜 신호 필터링
- 연속 손실 시 역추세 모드로 자동 전환
"""

from dataclasses import dataclass, field

import pandas as pd

from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.registry import register


@dataclass
class ScalperState:
    """트레이드 간 유지되는 상태."""
    current_direction: str = "NONE"  # LONG / SHORT / NONE
    consecutive_losses: int = 0
    contrarian_mode: bool = False    # 역추세 모드
    total_trades: int = 0
    recent_pnls: list[float] = field(default_factory=list)
    entry_atr: float = 0.0          # 진입 시점 ATR (동적 SL/TP용)

    def record_result(self, pnl: float) -> None:
        self.recent_pnls.append(pnl)
        if len(self.recent_pnls) > 20:
            self.recent_pnls.pop(0)
        self.total_trades += 1

        if pnl < 0:
            self.consecutive_losses += 1
            # 3연패 → 역추세 모드 진입
            if self.consecutive_losses >= 3 and not self.contrarian_mode:
                self.contrarian_mode = True
                self.consecutive_losses = 0
        else:
            # 수익 나면 연패 카운터 리셋
            self.consecutive_losses = 0
            # 역추세 모드에서 수익 → 정상 복귀
            if self.contrarian_mode:
                self.contrarian_mode = False


@register
class MomentumFlipScalper(Strategy):
    """
    1분봉 스캘핑 전략.

    판단 로직 (단순함):
    1. EMA(3)과 EMA(8)의 크로스 → 방향 결정
    2. 현재 거래량 > 최근 평균의 1.2배 → 신호 강도 확인
    3. 방향이 바뀌면 → 기존 포지션 청산 + 반대 방향 진입
    4. 3연패 시 → 신호를 반전 (역추세 모드)
    """

    # v1: ALWAYS_FLIP — 항상 포지션 보유, EMA 크로스로 방향 전환
    LEVERAGE = 5
    POSITION_SIZE_PCT = 0.15    # 보수적 (항상 포지션 보유이므로)
    SL_ATR_MULT = 2.5           # SL = 2.5 × ATR
    TP_ATR_MULT = 3.75          # TP = 3.75 × ATR (1:1.5 RR)
    MAX_HOLD_HOURS = 0          # ALWAYS_FLIP은 시간제한 없음

    def __init__(self, ema_fast: int = 3, ema_slow: int = 8) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.state = ScalperState()

    @property
    def name(self) -> str:
        return "momentum_flip_scalper"

    @property
    def label(self) -> str:
        return "v1. Momentum Flip Scalper"

    @property
    def description(self) -> str:
        return "EMA(3/8) 크로스 + 거래량 필터. 항상 포지션 보유, 방향 전환 시 즉시 플립. 3연패 시 역추세 전환."

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.ALWAYS_FLIP

    def record_result(self, pnl: float) -> None:
        self.state.record_result(pnl)

    def evaluate(self, symbol: str, candles: pd.DataFrame,
                 htf_candles: pd.DataFrame | None = None) -> Signal:
        from src.core.time_filter import is_tradeable_hour
        if not is_tradeable_hour():
            return Signal(symbol=symbol, type=SignalType.HOLD, confidence=0.0, source=self.name)

        if len(candles) < self.ema_slow + 5:
            return Signal(
                symbol=symbol, type=SignalType.HOLD,
                confidence=0.0, source=self.name,
            )

        df = candles.copy()
        close = df["close"]
        volume = df["volume"]

        # ── 0. ATR 계산 (동적 SL/TP용) ──
        high, low = df["high"], df["low"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_val = tr.rolling(14).mean().iloc[-1]
        if pd.notna(atr_val) and atr_val > 0:
            self.state.entry_atr = float(atr_val)

        # ── 1. EMA 크로스 방향 ──
        ema_fast = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.ema_slow, adjust=False).mean()

        current_fast = ema_fast.iloc[-1]
        current_slow = ema_slow.iloc[-1]
        prev_fast = ema_fast.iloc[-2]
        prev_slow = ema_slow.iloc[-2]

        # 크로스 발생 여부
        cross_up = prev_fast <= prev_slow and current_fast > current_slow
        cross_down = prev_fast >= prev_slow and current_fast < current_slow

        # 기존 추세 유지 (크로스 없을 때)
        trending_up = current_fast > current_slow
        trending_down = current_fast < current_slow

        # ── 2. 거래량 필터 ──
        vol_avg = volume.rolling(20).mean().iloc[-1]
        vol_now = volume.iloc[-1]
        vol_strong = vol_now > vol_avg * 1.2

        # ── 3. 방향 결정 (LONG은 거래량 확인 필수) ──
        if cross_up:
            if not vol_strong:
                return Signal(symbol=symbol, type=SignalType.HOLD,
                              confidence=0.0, source=self.name,
                              metadata={"reason": "long_needs_volume"})
            raw_direction = "LONG"
            confidence = 0.9
        elif cross_down:
            raw_direction = "SHORT"
            confidence = 0.9 if vol_strong else 0.7
        elif trending_up and vol_strong:
            raw_direction = "LONG"
            confidence = 0.6
        elif trending_down and vol_strong:
            raw_direction = "SHORT"
            confidence = 0.6
        else:
            # 추세 유지, 플립 필요 없음
            return Signal(
                symbol=symbol, type=SignalType.HOLD,
                confidence=0.0, source=self.name,
                metadata={"reason": "no_signal", "contrarian": self.state.contrarian_mode},
            )

        # ── 4. 역추세 모드: 신호 반전 ──
        if self.state.contrarian_mode:
            raw_direction = "SHORT" if raw_direction == "LONG" else "LONG"

        # ── 5. 같은 방향이면 유지 ──
        if raw_direction == self.state.current_direction:
            return Signal(
                symbol=symbol, type=SignalType.HOLD,
                confidence=0.0, source=self.name,
                metadata={"reason": "same_direction", "direction": raw_direction},
            )

        # ── 6. 방향 전환 → 플립 신호 ──
        self.state.current_direction = raw_direction
        signal_type = SignalType.BUY if raw_direction == "LONG" else SignalType.SELL

        return Signal(
            symbol=symbol,
            type=signal_type,
            confidence=confidence,
            source=self.name,
            metadata={
                "direction": raw_direction,
                "cross_up": cross_up,
                "cross_down": cross_down,
                "vol_strong": vol_strong,
                "contrarian": self.state.contrarian_mode,
                "consecutive_losses": self.state.consecutive_losses,
            },
        )
