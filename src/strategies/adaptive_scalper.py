"""v2. Adaptive Scalper — 멀티 타임프레임 적응형 스캘핑.

핵심 원리:
- 15분봉으로 큰 추세/시장 상태 판단
- 1분봉 300개로 정밀 진입 타이밍
- 횡보면 진입 금지, 추세장에서만 매매
- 점수 기반 진입, 쿨다운, 부분익절 + 트레일링
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
import ta

from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.registry import register


class MarketState(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class AdaptiveState:
    cooldown_remaining: int = 0
    position_side: str = "NONE"
    partial_tp_taken: bool = False
    trailing_stop_price: float | None = None
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float("inf")
    total_trades: int = 0
    wins: int = 0
    losses: int = 0

    def trigger_cooldown(self, candles: int = 4) -> None:
        self.cooldown_remaining = candles

    def tick_cooldown(self) -> bool:
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return True
        return False

    def open(self, side: str, price: float) -> None:
        self.position_side = side
        self.partial_tp_taken = False
        self.trailing_stop_price = None
        self.highest_since_entry = price
        self.lowest_since_entry = price

    def close(self) -> None:
        self.position_side = "NONE"
        self.partial_tp_taken = False
        self.trailing_stop_price = None

    def update_price(self, price: float) -> None:
        self.highest_since_entry = max(self.highest_since_entry, price)
        self.lowest_since_entry = min(self.lowest_since_entry, price)


# ─── 시장 상태 판단 (15분봉 기반) ──────────────────────────

def detect_market_state_htf(df: pd.DataFrame) -> MarketState:
    """15분봉으로 큰 추세를 판단한다."""
    close = df["close"]

    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    e8, e21, e50 = ema8.iloc[-1], ema21.iloc[-1], ema50.iloc[-1]
    aligned_up = e8 > e21 > e50
    aligned_down = e8 < e21 < e50

    # ATR 변동성
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], close, window=14).average_true_range()
    atr_now = atr.iloc[-1]
    atr_avg = atr.rolling(20).mean().iloc[-1]
    atr_ratio = atr_now / atr_avg if atr_avg > 0 else 1.0

    # BB 밴드폭
    bb = ta.volatility.BollingerBands(close, window=20)
    bb_mavg = bb.bollinger_mavg()
    bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb_mavg.replace(0, np.nan)
    bw_now = bb_width.iloc[-1]
    bw_avg = bb_width.rolling(20).mean().iloc[-1]

    if atr_ratio > 1.5:
        return MarketState.VOLATILE
    if aligned_up and bw_now > bw_avg * 0.8:
        return MarketState.TRENDING_UP
    if aligned_down and bw_now > bw_avg * 0.8:
        return MarketState.TRENDING_DOWN
    return MarketState.RANGING


def detect_market_state_1m(df: pd.DataFrame) -> MarketState:
    """1분봉으로 단기 추세를 보조 판단한다."""
    close = df["close"]
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    if ema8.iloc[-1] > ema21.iloc[-1]:
        return MarketState.TRENDING_UP
    elif ema8.iloc[-1] < ema21.iloc[-1]:
        return MarketState.TRENDING_DOWN
    return MarketState.RANGING


# ─── 진입 점수 시스템 (1분봉 기반, 6점 만점) ───────────────

def compute_entry_score(df_1m: pd.DataFrame, df_htf: pd.DataFrame | None,
                        direction: str) -> tuple[int, dict]:
    """6개 지표 기반 진입 점수."""
    close = df_1m["close"]
    score = 0
    details = {}

    # 1. 1분봉 EMA 정렬 (+1)
    ema8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    if (direction == "LONG" and ema8 > ema21) or (direction == "SHORT" and ema8 < ema21):
        score += 1
        details["ema_1m"] = True
    else:
        details["ema_1m"] = False

    # 2. 15분봉 추세 동의 (+1) — 핵심: 상위 프레임과 같은 방향이면 가산
    if df_htf is not None and len(df_htf) > 50:
        htf_state = detect_market_state_htf(df_htf)
        htf_agrees = (
            (direction == "LONG" and htf_state == MarketState.TRENDING_UP) or
            (direction == "SHORT" and htf_state == MarketState.TRENDING_DOWN)
        )
        if htf_agrees:
            score += 1
            details["htf_agree"] = True
        else:
            details["htf_agree"] = False
        details["htf_state"] = htf_state.value
    else:
        details["htf_agree"] = False
        details["htf_state"] = "unavailable"

    # 3. 거래량 (+1)
    vol = df_1m["volume"]
    vol_avg = vol.rolling(20).mean().iloc[-1]
    vol_ratio = vol.iloc[-1] / vol_avg if vol_avg > 0 else 0
    if vol_ratio > 1.2:
        score += 1
        details["volume"] = True
    else:
        details["volume"] = False
    details["vol_ratio"] = round(vol_ratio, 2)

    # 4. RSI 영역 (+1)
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    if 30 < rsi < 70:
        score += 1
        details["rsi_ok"] = True
    else:
        details["rsi_ok"] = False
    details["rsi"] = round(rsi, 1)

    # 5. MACD 방향 (+1)
    macd = ta.trend.MACD(close)
    hist = macd.macd_diff().iloc[-1]
    if (direction == "LONG" and hist > 0) or (direction == "SHORT" and hist < 0):
        score += 1
        details["macd"] = True
    else:
        details["macd"] = False

    # 6. BB 위치 (+1)
    bb = ta.volatility.BollingerBands(close, window=20)
    bb_pct = (close.iloc[-1] - bb.bollinger_lband().iloc[-1]) / \
             (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1] + 1e-10)
    if (direction == "LONG" and bb_pct < 0.5) or (direction == "SHORT" and bb_pct > 0.5):
        score += 1
        details["bb_position"] = True
    else:
        details["bb_position"] = False
    details["bb_pct"] = round(bb_pct, 2)

    return score, details


# ─── 전략 클래스 ───────────────────────────────────────────

@register
class AdaptiveScalper(Strategy):
    """v2 멀티 타임프레임 적응형 스캘핑."""

    SCORE_THRESHOLD = 4   # 6점 만점 중 4점 이상 (15분봉 추가로 기준 상향)
    COOLDOWN_CANDLES = 4

    PARTIAL_TP_PCT = 0.005
    FULL_TP_PCT = 0.012
    TRAILING_ACTIVATE_PCT = 0.008
    TRAILING_DISTANCE_PCT = 0.003
    SL_PCT = 0.005

    def __init__(self) -> None:
        self.state = AdaptiveState()

    @property
    def name(self) -> str:
        return "adaptive_scalper"

    @property
    def label(self) -> str:
        return "v2. Adaptive Scalper"

    @property
    def description(self) -> str:
        return (
            "멀티 타임프레임: 15분봉 추세 + 1분봉 진입. "
            "횡보 진입금지, 점수 기반 진입 (4/6점). "
            "쿨다운, 부분익절 + 트레일링."
        )

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.SIGNAL_ONLY

    def record_result(self, pnl: float) -> None:
        self.state.total_trades += 1
        if pnl >= 0:
            self.state.wins += 1
        else:
            self.state.losses += 1
            self.state.trigger_cooldown(self.COOLDOWN_CANDLES)
        self.state.close()

    def evaluate(self, symbol: str, candles: pd.DataFrame,
                 htf_candles: pd.DataFrame | None = None) -> Signal:
        from src.core.time_filter import is_tradeable_hour
        if not is_tradeable_hour():
            return self._hold(symbol, reason="blocked_hour")

        if len(candles) < 100:
            return self._hold(symbol, reason="insufficient_data")

        df = candles.copy()
        price = df["close"].iloc[-1]

        # ── 쿨다운 ──
        if self.state.tick_cooldown():
            return self._hold(symbol, reason="cooldown",
                              cooldown=self.state.cooldown_remaining)

        # ── 시장 상태: 15분봉 우선, 없으면 1분봉 ──
        if htf_candles is not None and len(htf_candles) > 50:
            market = detect_market_state_htf(htf_candles)
        else:
            market = detect_market_state_1m(df)

        # ── No Trade Zone ──
        if market == MarketState.RANGING:
            return self._hold(symbol, reason="no_trade_zone", market=market.value)

        # ── 포지션 보유 중: 청산 판단 ──
        if self.state.position_side != "NONE":
            return self._evaluate_exit(symbol, df, price, market, htf_candles)

        # ── 방향 결정 ──
        if market == MarketState.TRENDING_UP:
            direction = "LONG"
        elif market == MarketState.TRENDING_DOWN:
            direction = "SHORT"
        elif market == MarketState.VOLATILE:
            short_state = detect_market_state_1m(df)
            direction = "LONG" if short_state == MarketState.TRENDING_UP else "SHORT"
        else:
            return self._hold(symbol, reason="no_direction", market=market.value)

        # ── 점수 계산 (1분봉 + 15분봉) ──
        score, details = compute_entry_score(df, htf_candles, direction)

        if score < self.SCORE_THRESHOLD:
            return self._hold(symbol, reason="low_score", score=score,
                              threshold=self.SCORE_THRESHOLD, details=details,
                              market=market.value)

        # ── 진입 ──
        self.state.open(direction, price)
        signal_type = SignalType.BUY if direction == "LONG" else SignalType.SELL
        confidence = score / 6.0

        return Signal(
            symbol=symbol, type=signal_type, confidence=confidence,
            source=self.name,
            metadata={
                "direction": direction, "market": market.value,
                "score": score, "max_score": 6, "details": details,
                "partial_tp_pct": self.PARTIAL_TP_PCT,
                "full_tp_pct": self.FULL_TP_PCT,
                "trailing_activate_pct": self.TRAILING_ACTIVATE_PCT,
                "trailing_distance_pct": self.TRAILING_DISTANCE_PCT,
                "sl_pct": self.SL_PCT,
            },
        )

    def _evaluate_exit(self, symbol: str, df: pd.DataFrame, price: float,
                       market: MarketState, htf_candles: pd.DataFrame | None) -> Signal:
        self.state.update_price(price)
        side = self.state.position_side

        # 15분봉 추세 역전 시 청산
        if side == "LONG" and market == MarketState.TRENDING_DOWN:
            return Signal(symbol=symbol, type=SignalType.CLOSE, confidence=0.8,
                          source=self.name, metadata={"reason": "trend_reversal_htf"})
        if side == "SHORT" and market == MarketState.TRENDING_UP:
            return Signal(symbol=symbol, type=SignalType.CLOSE, confidence=0.8,
                          source=self.name, metadata={"reason": "trend_reversal_htf"})

        # 1분봉 단기 추세도 역전이면 추가 확인
        short_state = detect_market_state_1m(df)
        if side == "LONG" and short_state == MarketState.TRENDING_DOWN and market != MarketState.TRENDING_UP:
            return Signal(symbol=symbol, type=SignalType.CLOSE, confidence=0.6,
                          source=self.name, metadata={"reason": "trend_reversal_1m"})
        if side == "SHORT" and short_state == MarketState.TRENDING_UP and market != MarketState.TRENDING_DOWN:
            return Signal(symbol=symbol, type=SignalType.CLOSE, confidence=0.6,
                          source=self.name, metadata={"reason": "trend_reversal_1m"})

        # 트레일링 스탑
        if self.state.trailing_stop_price is not None:
            if side == "LONG" and price <= self.state.trailing_stop_price:
                return Signal(symbol=symbol, type=SignalType.CLOSE, confidence=0.9,
                              source=self.name, metadata={"reason": "trailing_stop"})
            if side == "SHORT" and price >= self.state.trailing_stop_price:
                return Signal(symbol=symbol, type=SignalType.CLOSE, confidence=0.9,
                              source=self.name, metadata={"reason": "trailing_stop"})

            if side == "LONG":
                new_stop = self.state.highest_since_entry * (1 - self.TRAILING_DISTANCE_PCT)
                if new_stop > self.state.trailing_stop_price:
                    self.state.trailing_stop_price = new_stop
            else:
                new_stop = self.state.lowest_since_entry * (1 + self.TRAILING_DISTANCE_PCT)
                if new_stop < self.state.trailing_stop_price:
                    self.state.trailing_stop_price = new_stop

        return self._hold(symbol, reason="hold_position", market=market.value, side=side)

    def _hold(self, symbol: str, reason: str = "", **kwargs) -> Signal:
        return Signal(symbol=symbol, type=SignalType.HOLD, confidence=0.0,
                      source=self.name, metadata={"reason": reason, **kwargs})
