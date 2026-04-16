"""v14. VWAP Reversion Scalper — 거래량 가중 평균가 회귀 전략.

기존 전략들(추세 기반)과 다른 시그널 소스를 제공하여 포트폴리오 다변화.
캔들 OHLCV만으로 계산 — 추가 API 호출 없음, 1코어 1GB 서버 최적화.

진입 로직:
  1. VWAP = cumsum(typical_price × volume) / cumsum(volume)
  2. 가격 < VWAP - ENTRY_ATR_MULT × ATR → BUY (과매도 회귀 기대)
  3. 가격 > VWAP + ENTRY_ATR_MULT × ATR → SELL (과매수 회귀 기대)
  4. HTF EMA 필터: HTF 추세 반대 방향 진입 차단 (역추세 방지)
  5. RSI 보조 확인: 극단 구간에서만 진입 (BUY: RSI<40, SELL: RSI>60)

청산: ATR 기반 SL/TP (엔진 관리).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import ta

from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.registry import register

KST = timezone(timedelta(hours=9))

# VWAP 이탈 진입 기준 (ATR 배수)
ENTRY_ATR_MULT = 1.0
# RSI 필터
RSI_PERIOD = 14
RSI_OVERSOLD = 40
RSI_OVERBOUGHT = 60
# EMA 기간 (HTF 추세 필터)
EMA_FAST = 9
EMA_SLOW = 21
# 매매 제한
MAX_TRADES_PER_HOUR = 3
COOLDOWN_BASE = 2  # 기본 쿨다운 (틱)
# 새벽 시간대 매매 금지 (KST)
BLOCKED_HOURS = {0, 1, 2, 3, 4, 5}


@dataclass
class V14State:
    position_side: str = "NONE"
    entry_atr: float = 0.0
    cooldown_remaining: int = 0
    consecutive_losses: int = 0
    trades_this_hour: int = 0
    last_hour: int = -1
    total_trades: int = 0
    wins: int = 0
    losses: int = 0


@register
class VWAPReversionScalper(Strategy):
    """v14 — VWAP 회귀 전략. 횡보장에 강점."""

    LEVERAGE = 5
    POSITION_SIZE_PCT = 0.15
    MAX_HOLD_HOURS = 4.0
    TIMEFRAMES = ["15m", "1h"]
    SL_ATR_MULT = 1.5
    TP_ATR_MULT = 3.0

    def __init__(self) -> None:
        self.state = V14State()

    @property
    def name(self) -> str:
        return "vwap_reversion"

    @property
    def label(self) -> str:
        return "v14. VWAP Reversion Scalper"

    @property
    def description(self) -> str:
        return (
            "VWAP(거래량 가중 평균가) 이탈 시 평균 회귀 진입. "
            "RSI + HTF EMA 필터. 횡보장에서 강점, 추세 전략과 낮은 상관관계."
        )

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.SIGNAL_ONLY

    def evaluate(
        self,
        symbol: str,
        candles: pd.DataFrame,
        htf_candles: pd.DataFrame | None = None,
    ) -> Signal:
        st = self.state
        if st.cooldown_remaining > 0:
            st.cooldown_remaining -= 1
            return self._hold(symbol, reason="cooldown")

        if len(candles) < 50:
            return self._hold(symbol, reason="insufficient_data")

        now_kst = datetime.now(KST)
        if now_kst.hour in BLOCKED_HOURS:
            return self._hold(symbol, reason="blocked_hour")

        # 시간당 매매 제한
        current_hour = now_kst.hour
        if current_hour != st.last_hour:
            st.trades_this_hour = 0
            st.last_hour = current_hour
        if st.trades_this_hour >= MAX_TRADES_PER_HOUR:
            return self._hold(symbol, reason="hourly_limit")

        close = candles["close"].values.astype(float)
        high = candles["high"].values.astype(float)
        low = candles["low"].values.astype(float)
        volume = candles["volume"].values.astype(float)
        price = close[-1]

        # ATR
        atr_series = ta.volatility.AverageTrueRange(
            candles["high"], candles["low"], candles["close"], window=14,
        ).average_true_range()
        atr = float(atr_series.iloc[-1])
        if atr <= 0 or pd.isna(atr):
            return self._hold(symbol, reason="zero_atr")

        # VWAP 계산
        typical_price = (high + low + close) / 3.0
        cum_tp_vol = np.cumsum(typical_price * volume)
        cum_vol = np.cumsum(volume)
        cum_vol[cum_vol == 0] = 1e-10  # 0 나눗셈 방지
        vwap = cum_tp_vol / cum_vol
        current_vwap = vwap[-1]

        # RSI
        rsi_series = ta.momentum.RSIIndicator(
            candles["close"], window=RSI_PERIOD,
        ).rsi()
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0

        # HTF 추세 필터
        htf_trend = self._htf_trend(htf_candles)

        # 진입 판단
        deviation = price - current_vwap
        threshold = ENTRY_ATR_MULT * atr

        direction = None
        score = 0

        if deviation < -threshold and rsi < RSI_OVERSOLD:
            # VWAP 아래 이탈 + RSI 과매도 → BUY
            if htf_trend != "SHORT":
                direction = "LONG"
                score = min(5, int(abs(deviation) / threshold) + (1 if rsi < 30 else 0)
                            + (1 if htf_trend == "LONG" else 0))
        elif deviation > threshold and rsi > RSI_OVERBOUGHT:
            # VWAP 위 이탈 + RSI 과매수 → SELL
            if htf_trend != "LONG":
                direction = "SHORT"
                score = min(5, int(abs(deviation) / threshold) + (1 if rsi > 70 else 0)
                            + (1 if htf_trend == "SHORT" else 0))

        if direction is None or score < 2:
            return self._hold(
                symbol, reason="no_signal",
                vwap=round(current_vwap, 2), deviation=round(deviation, 2),
                rsi=round(rsi, 1), htf_trend=htf_trend,
            )

        confidence = min(1.0, (score + 1) / 6.0)  # 2~5점 → 0.5~1.0

        signal_type = SignalType.BUY if direction == "LONG" else SignalType.SELL

        st.entry_atr = atr
        st.trades_this_hour += 1
        st.total_trades += 1

        return Signal(
            symbol=symbol,
            type=signal_type,
            confidence=round(confidence, 3),
            source=self.name,
            metadata={
                "direction": direction,
                "score": score,
                "vwap": round(current_vwap, 2),
                "deviation": round(deviation, 2),
                "deviation_atr": round(abs(deviation) / atr, 2),
                "rsi": round(rsi, 1),
                "htf_trend": htf_trend,
                "atr": round(atr, 4),
            },
        )

    def record_result(self, pnl: float) -> None:
        st = self.state
        if pnl >= 0:
            st.wins += 1
            st.consecutive_losses = 0
            st.cooldown_remaining = COOLDOWN_BASE
        else:
            st.losses += 1
            st.consecutive_losses += 1
            st.cooldown_remaining = COOLDOWN_BASE * (1 + st.consecutive_losses)

    def _htf_trend(self, htf_candles: pd.DataFrame | None) -> str:
        """HTF EMA 방향. LONG/SHORT/NEUTRAL."""
        if htf_candles is None or len(htf_candles) < 30:
            return "NEUTRAL"
        close = pd.Series(htf_candles["close"].values.astype(float))
        ema_fast = close.ewm(span=EMA_FAST, adjust=False).mean().iloc[-1]
        ema_slow = close.ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1]
        if ema_fast > ema_slow:
            return "LONG"
        elif ema_fast < ema_slow:
            return "SHORT"
        return "NEUTRAL"

    def _hold(self, symbol: str, reason: str = "", **kwargs) -> Signal:
        return Signal(
            symbol=symbol,
            type=SignalType.HOLD,
            confidence=0.0,
            source=self.name,
            metadata={"reason": reason, **kwargs},
        )
