"""v11. Data-Driven Scalper — 가상매매 데이터 기반 전략.

v1~v10 가상매매 분석에서 발견된 패턴:
1. 새벽~오전(0-9시 KST) 승률 극히 낮음 (10%) → 매매 금지
2. LONG이 SHORT보다 손실 큼 (avg -1.83 vs -0.34) → LONG 진입 기준 강화
3. 연속 손실 4회가 빈번 → 적응형 쿨다운
4. v3의 ATR+RR 검증 + v2의 멀티TF가 효과적 → 결합
5. v1의 EMA(3,8)은 너무 민감 → EMA(8,21) 사용

설계 원칙:
- 데이터가 말한 것만 반영 (과적합 금지)
- 매매 안 하는 것도 전략 (No Trade = No Loss)
- 수수료+슬리피지 사전 차감
- 최소 1.5:1 리스크-리워드
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import ta

from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.registry import register


# ─── 상수 ──────────────────────────────────────────────────
KST = timezone(timedelta(hours=9))

EXECUTION_COST = 0.0006   # 왕복 수수료+슬리피지 0.06%
MIN_RR_RATIO = 1.5        # 최소 리스크-리워드
SL_ATR_MULT = 1.5         # SL = 1.5 ATR (타이트)
TP_ATR_MULT = 3.0         # TP = 3.0 ATR (넉넉)

# 매매 금지 시간대 (KST) — 데이터 기반: 0~9시 승률 10%
NO_TRADE_HOURS_KST = set(range(0, 10))  # 0, 1, 2, ..., 9

# 진입 점수 임계값
SCORE_THRESHOLD = 5        # 8점 만점 중 5점
SCORE_THRESHOLD_LONG = 6   # LONG은 더 엄격 (데이터: LONG 약세)

MAX_TRADES_PER_HOUR = 3
COOLDOWN_BASE = 3          # 기본 쿨다운 (틱 수)


@dataclass
class V11State:
    position_side: str = "NONE"
    entry_atr: float = 0.0
    partial_tp_taken: bool = False
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float("inf")
    trailing_stop_price: float | None = None

    cooldown_remaining: int = 0
    consecutive_losses: int = 0
    trades_this_hour: int = 0
    last_hour: int = -1
    total_trades: int = 0
    wins: int = 0
    losses: int = 0

    def open(self, side: str, price: float, atr: float) -> None:
        self.position_side = side
        self.entry_atr = atr
        self.partial_tp_taken = False
        self.highest_since_entry = price
        self.lowest_since_entry = price
        self.trailing_stop_price = None

    def close(self) -> None:
        self.position_side = "NONE"
        self.entry_atr = 0.0
        self.trailing_stop_price = None

    def update_price(self, price: float) -> None:
        self.highest_since_entry = max(self.highest_since_entry, price)
        self.lowest_since_entry = min(self.lowest_since_entry, price)

    def check_trade_limit(self, current_hour: int) -> bool:
        if current_hour != self.last_hour:
            self.last_hour = current_hour
            self.trades_this_hour = 0
        return self.trades_this_hour < MAX_TRADES_PER_HOUR


# ─── 분석 함수 ────────────────────────────────────────────

def get_kst_hour() -> int:
    """현재 KST 시간 반환."""
    return datetime.now(KST).hour


def compute_trend(df: pd.DataFrame) -> dict:
    """EMA + ADX 기반 추세 분석."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    # ADX
    adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
    adx = adx_ind.adx().iloc[-1]
    plus_di = adx_ind.adx_pos().iloc[-1]
    minus_di = adx_ind.adx_neg().iloc[-1]

    # EMA 정렬
    aligned_up = ema8.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]
    aligned_down = ema8.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]

    # ATR
    atr_ind = ta.volatility.AverageTrueRange(high, low, close, window=14)
    atr = atr_ind.average_true_range()
    atr_now = atr.iloc[-1]
    atr_avg = atr.rolling(20).mean().iloc[-1]
    atr_ratio = atr_now / atr_avg if atr_avg > 0 else 1.0

    return {
        "ema8": ema8.iloc[-1],
        "ema21": ema21.iloc[-1],
        "ema50": ema50.iloc[-1],
        "aligned_up": aligned_up,
        "aligned_down": aligned_down,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "atr": atr_now,
        "atr_ratio": atr_ratio,
    }


def compute_entry_score(
    df_1m: pd.DataFrame,
    df_htf: pd.DataFrame | None,
    direction: str,
    trend_1m: dict,
    trend_htf: dict | None,
) -> tuple[int, int, dict]:
    """8개 지표 기반 진입 점수. (점수, 만점, 상세) 반환."""
    close = df_1m["close"]
    volume = df_1m["volume"]
    price = close.iloc[-1]
    score = 0
    max_score = 8
    details = {}

    # 1. 1분봉 EMA 정렬 (+1)
    if direction == "LONG" and trend_1m["aligned_up"]:
        score += 1
        details["ema_align"] = True
    elif direction == "SHORT" and trend_1m["aligned_down"]:
        score += 1
        details["ema_align"] = True
    else:
        details["ema_align"] = False

    # 2. ADX 방향 확인 (+1)
    if direction == "LONG" and trend_1m["plus_di"] > trend_1m["minus_di"]:
        score += 1
        details["adx_dir"] = True
    elif direction == "SHORT" and trend_1m["minus_di"] > trend_1m["plus_di"]:
        score += 1
        details["adx_dir"] = True
    else:
        details["adx_dir"] = False

    # 3. ADX 강도 (+1) — 진짜 추세만
    if trend_1m["adx"] > 25:
        score += 1
        details["adx_strong"] = True
    else:
        details["adx_strong"] = False

    # 4. 15분봉 추세 동의 (+1)
    htf_agrees = False
    if trend_htf is not None:
        if direction == "LONG" and trend_htf["aligned_up"]:
            htf_agrees = True
        elif direction == "SHORT" and trend_htf["aligned_down"]:
            htf_agrees = True
    if htf_agrees:
        score += 1
    details["htf_agree"] = htf_agrees

    # 5. 거래량 확인 (+1) — 1.5배 이상
    vol_avg = volume.rolling(20).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 0
    if vol_ratio > 1.5:
        score += 1
        details["volume"] = True
    else:
        details["volume"] = False
    details["vol_ratio"] = round(vol_ratio, 2)

    # 6. RSI 유리 영역 (+1) — 추세 방향 확인
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    if direction == "LONG" and 35 < rsi < 60:
        score += 1
        details["rsi_ok"] = True
    elif direction == "SHORT" and 40 < rsi < 65:
        score += 1
        details["rsi_ok"] = True
    else:
        details["rsi_ok"] = False
    details["rsi"] = round(rsi, 1)

    # 7. MACD 히스토그램 성장 (+1)
    macd = ta.trend.MACD(close)
    hist = macd.macd_diff()
    hist_now = hist.iloc[-1]
    hist_prev = hist.iloc[-2]
    if direction == "LONG" and hist_now > hist_prev and hist_now > 0:
        score += 1
        details["macd_grow"] = True
    elif direction == "SHORT" and hist_now < hist_prev and hist_now < 0:
        score += 1
        details["macd_grow"] = True
    else:
        details["macd_grow"] = False

    # 8. BB 위치 — 유리한 가격대 (+1)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_high = bb.bollinger_hband().iloc[-1]
    bb_low = bb.bollinger_lband().iloc[-1]
    bb_mid = bb.bollinger_mavg().iloc[-1]
    bb_range = bb_high - bb_low
    if bb_range > 0:
        bb_pct = (price - bb_low) / bb_range
        if direction == "LONG" and bb_pct < 0.4:
            score += 1
            details["bb_pos"] = True
        elif direction == "SHORT" and bb_pct > 0.6:
            score += 1
            details["bb_pos"] = True
        else:
            details["bb_pos"] = False
        details["bb_pct"] = round(bb_pct, 2)
    else:
        details["bb_pos"] = False

    return score, max_score, details


# ─── 전략 클래스 ──────────────────────────────────────────

@register
class DataDrivenScalper(Strategy):
    """v11 — 가상매매 데이터 기반 전략.

    핵심 차별점:
    1. 시간대 필터 (새벽 매매 금지 — 데이터 기반)
    2. LONG 진입 기준 강화 (데이터: LONG 약세)
    3. 8점 진입 시스템 (v2+v3 결합)
    4. ATR 기반 동적 SL/TP (최소 1.5:1 RR)
    5. 적응형 쿨다운 (연패 시 증가)
    """

    SL_ATR_MULT = SL_ATR_MULT
    TP_ATR_MULT = TP_ATR_MULT
    PARTIAL_TP_ATR_MULT = round(TP_ATR_MULT * 0.5, 2)
    TRAILING_ATR_MULT = round(TP_ATR_MULT * 0.7, 2)
    TRAILING_DIST_ATR = 0.5

    def __init__(self) -> None:
        self.state = V11State()

    @property
    def name(self) -> str:
        return "data_driven_scalper"

    @property
    def label(self) -> str:
        return "v11. Data-Driven Scalper"

    @property
    def description(self) -> str:
        return (
            "가상매매 데이터 기반. 새벽 매매 금지, LONG 기준 강화, "
            "8점 진입 시스템, ATR 동적 SL/TP, 적응형 쿨다운."
        )

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.SIGNAL_ONLY

    def record_result(self, pnl: float) -> None:
        self.state.total_trades += 1
        if pnl >= 0:
            self.state.wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.losses += 1
            self.state.consecutive_losses += 1
            cooldown = COOLDOWN_BASE + (self.state.consecutive_losses * 3)
            self.state.cooldown_remaining = min(cooldown, 30)
        self.state.close()

    def evaluate(self, symbol: str, candles: pd.DataFrame,
                 htf_candles: pd.DataFrame | None = None) -> Signal:
        if len(candles) < 100:
            return self._hold(symbol, reason="insufficient_data")

        df = candles.copy()
        close = df["close"]
        price = close.iloc[-1]

        # ── 시간대 필터 (KST) ──
        kst_hour = get_kst_hour()
        if kst_hour in NO_TRADE_HOURS_KST:
            return self._hold(symbol, reason="no_trade_hour", kst_hour=kst_hour)

        # ── 쿨다운 ──
        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            return self._hold(symbol, reason="cooldown",
                              remaining=self.state.cooldown_remaining)

        # ── 시간당 매매 제한 ──
        if not self.state.check_trade_limit(kst_hour):
            return self._hold(symbol, reason="trade_limit")

        # ── 포지션 보유 중 ──
        if self.state.position_side != "NONE":
            self.state.update_price(price)
            return self._evaluate_exit(symbol, df, htf_candles, price)

        # ── 1분봉 추세 분석 ──
        trend_1m = compute_trend(df)
        atr = trend_1m["atr"]

        if atr <= 0 or pd.isna(atr):
            return self._hold(symbol, reason="zero_atr")

        # ATR이 너무 작으면 수수료 대비 수익 불가
        atr_pct = atr / price
        if atr_pct < EXECUTION_COST * 3:
            return self._hold(symbol, reason="atr_too_small")

        # ── 15분봉 추세 분석 ──
        trend_htf = None
        if htf_candles is not None and len(htf_candles) > 50:
            trend_htf = compute_trend(htf_candles)

        # ── ADX < 20 → 횡보 → 매매 안 함 ──
        if trend_1m["adx"] < 20:
            return self._hold(symbol, reason="ranging",
                              adx=round(trend_1m["adx"], 1))

        # ── 변동성 급등 → 대기 ──
        if trend_1m["atr_ratio"] > 2.0:
            return self._hold(symbol, reason="volatile",
                              atr_ratio=round(trend_1m["atr_ratio"], 2))

        # ── 방향 결정 ──
        if trend_1m["plus_di"] > trend_1m["minus_di"]:
            direction = "LONG"
        else:
            direction = "SHORT"

        # ── 진입 점수 ──
        score, max_score, details = compute_entry_score(
            df, htf_candles, direction, trend_1m, trend_htf)

        # LONG은 더 엄격한 기준 적용
        threshold = SCORE_THRESHOLD_LONG if direction == "LONG" else SCORE_THRESHOLD

        if score < threshold:
            return self._hold(symbol, reason="low_score",
                              score=score, threshold=threshold,
                              direction=direction, details=details)

        # ── 리스크-리워드 사전 검증 ──
        expected_sl = atr * SL_ATR_MULT
        expected_tp = atr * TP_ATR_MULT
        real_tp = expected_tp - (price * EXECUTION_COST)
        real_sl = expected_sl + (price * EXECUTION_COST)
        real_rr = real_tp / real_sl if real_sl > 0 else 0

        if real_rr < MIN_RR_RATIO:
            return self._hold(symbol, reason="poor_rr",
                              real_rr=round(real_rr, 2))

        # ── 진입 ──
        self.state.open(direction, price, atr)
        self.state.trades_this_hour += 1
        signal_type = SignalType.BUY if direction == "LONG" else SignalType.SELL
        confidence = score / max_score

        return Signal(
            symbol=symbol, type=signal_type, confidence=confidence,
            source=self.name,
            metadata={
                "direction": direction,
                "score": score, "max_score": max_score,
                "threshold": threshold,
                "details": details,
                "atr": round(atr, 2),
                "real_rr": round(real_rr, 2),
                "adx": round(trend_1m["adx"], 1),
                "kst_hour": kst_hour,
            },
        )

    def _evaluate_exit(self, symbol: str, df: pd.DataFrame,
                       htf_candles: pd.DataFrame | None,
                       price: float) -> Signal:
        """추세 역전 감지 시 청산 신호."""
        side = self.state.position_side

        # 15분봉 추세 역전 체크
        if htf_candles is not None and len(htf_candles) > 50:
            trend_htf = compute_trend(htf_candles)
            if side == "LONG" and trend_htf["aligned_down"]:
                return Signal(symbol=symbol, type=SignalType.CLOSE,
                              confidence=0.8, source=self.name,
                              metadata={"reason": "htf_reversal"})
            if side == "SHORT" and trend_htf["aligned_up"]:
                return Signal(symbol=symbol, type=SignalType.CLOSE,
                              confidence=0.8, source=self.name,
                              metadata={"reason": "htf_reversal"})

        return self._hold(symbol, reason="hold_position", side=side)

    def _hold(self, symbol: str, reason: str = "", **kwargs) -> Signal:
        return Signal(symbol=symbol, type=SignalType.HOLD, confidence=0.0,
                      source=self.name, metadata={"reason": reason, **kwargs})
