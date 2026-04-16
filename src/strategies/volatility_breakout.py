"""v15. Volatility Breakout — 변동성 돌파 전략.

래리 윌리엄스 변동성 돌파 전략을 선물 양방향으로 확장.
전일 ATR 기반 돌파 구간을 계산하고, 돌파 시 추세 방향으로 진입.
극도로 단순한 로직으로 1코어 1GB 서버에 최적화.

진입 로직:
  1. 전일 range = 전일 고가 - 전일 저가
  2. 상단 돌파: 당일 시가 + K × 전일range 돌파 → BUY
  3. 하단 돌파: 당일 시가 - K × 전일range 돌파 → SELL
  4. K = 0.5 (래리 윌리엄스 기본값)
  5. 거래량 확인: 돌파 시 거래량이 20기간 평균보다 높을 것

청산: ATR 기반 SL/TP (엔진 관리) + 최대 보유시간.
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

# 돌파 계수
K_FACTOR = 0.5
# 거래량 필터 (평균 대비 배수)
VOLUME_THRESHOLD = 1.2
# 매매 제한
MAX_TRADES_PER_HOUR = 2
COOLDOWN_BASE = 3
# 새벽 시간대 매매 금지 (KST)
BLOCKED_HOURS = {0, 1, 2, 3, 4, 5}


@dataclass
class V15State:
    position_side: str = "NONE"
    entry_atr: float = 0.0
    cooldown_remaining: int = 0
    consecutive_losses: int = 0
    trades_this_hour: int = 0
    last_hour: int = -1
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    # 당일 돌파 여부 추적 (동일 방향 중복 진입 방지)
    today_breakout_long: bool = False
    today_breakout_short: bool = False
    last_date: str = ""


@register
class VolatilityBreakoutScalper(Strategy):
    """v15 — 변동성 돌파 전략. 추세 시작 포착에 강점."""

    LEVERAGE = 5
    POSITION_SIZE_PCT = 0.15
    MAX_HOLD_HOURS = 6.0
    TIMEFRAMES = ["15m", "1h"]
    SL_ATR_MULT = 2.0
    TP_ATR_MULT = 3.0

    def __init__(self) -> None:
        self.state = V15State()

    @property
    def name(self) -> str:
        return "volatility_breakout"

    @property
    def label(self) -> str:
        return "v15. Volatility Breakout"

    @property
    def description(self) -> str:
        return (
            "래리 윌리엄스 변동성 돌파(K=0.5) 선물 확장. "
            "전일 range 기반 상단/하단 돌파 시 추세 방향 진입. 극도로 단순, 추세 시작 포착."
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

        if len(candles) < 100:
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

        # 날짜 변경 시 돌파 추적 리셋
        today_str = now_kst.strftime("%Y-%m-%d")
        if today_str != st.last_date:
            st.today_breakout_long = False
            st.today_breakout_short = False
            st.last_date = today_str

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

        # 전일 range 계산 (15분봉 기준: ~96개 = 24시간)
        # 직전 96개 캔들의 high/low로 전일 range 근사
        lookback = min(96, len(candles) - 1)
        prev_high = float(np.max(high[-lookback - 1:-1]))
        prev_low = float(np.min(low[-lookback - 1:-1]))
        prev_range = prev_high - prev_low

        if prev_range <= 0:
            return self._hold(symbol, reason="zero_range")

        # 당일 시가 근사 (최근 구간의 첫 캔들)
        # 15분봉 기준 당일 KST 0시 이후 첫 캔들의 open
        day_open = float(candles["open"].iloc[-lookback])

        # 돌파 레벨
        upper_break = day_open + K_FACTOR * prev_range
        lower_break = day_open - K_FACTOR * prev_range

        # 거래량 확인
        vol_avg = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
        current_vol = volume[-1]
        vol_ok = current_vol >= vol_avg * VOLUME_THRESHOLD

        # HTF 추세 보조 (선택적 — 역추세 차단)
        htf_trend = self._htf_trend(htf_candles)

        direction = None
        score = 0

        if price > upper_break and not st.today_breakout_long:
            # 상단 돌파 → BUY
            if htf_trend != "SHORT":
                direction = "LONG"
                score = 2 + (1 if vol_ok else 0) + (1 if htf_trend == "LONG" else 0)
        elif price < lower_break and not st.today_breakout_short:
            # 하단 돌파 → SELL
            if htf_trend != "LONG":
                direction = "SHORT"
                score = 2 + (1 if vol_ok else 0) + (1 if htf_trend == "SHORT" else 0)

        if direction is None or score < 2:
            return self._hold(
                symbol, reason="no_breakout",
                upper=round(upper_break, 2), lower=round(lower_break, 2),
                price=round(price, 2), vol_ok=vol_ok,
            )

        confidence = min(1.0, (score + 1) / 5.0)  # 2~4점 → 0.6~1.0

        signal_type = SignalType.BUY if direction == "LONG" else SignalType.SELL

        # 돌파 기록
        if direction == "LONG":
            st.today_breakout_long = True
        else:
            st.today_breakout_short = True

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
                "upper_break": round(upper_break, 2),
                "lower_break": round(lower_break, 2),
                "day_open": round(day_open, 2),
                "prev_range": round(prev_range, 2),
                "vol_ratio": round(current_vol / vol_avg, 2) if vol_avg > 0 else 0,
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
        ema_fast = close.ewm(span=9, adjust=False).mean().iloc[-1]
        ema_slow = close.ewm(span=21, adjust=False).mean().iloc[-1]
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
