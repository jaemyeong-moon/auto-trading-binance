"""v12. Pattern Scalper — 쌍바닥/쌍봉 패턴 인식 전략.

차트 패턴 기반 진입:
1. 쌍바닥 (Double Bottom) → LONG 진입
2. 쌍봉 (Double Top) → SHORT 진입

패턴 조건:
- 두 저점/고점이 가격 대비 0.3% 이내 근접
- 두 저점/고점 사이에 반대 극점(넥라인) 존재
- 넥라인 돌파 + 거래량 확인 시 진입
- ATR 기반 동적 SL/TP (패턴 높이 활용)

기존 전략 대비 차이점:
- 추세/점수 시스템 대신 **가격 구조(패턴)**로 판단
- 횡보장에서도 거래 가능 (패턴은 횡보에서 자주 출현)
- 진입 조건이 상대적으로 관대 → 거래 빈도 높음
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import ta

from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.registry import register


# ─── 상수 ──────────────────────────────────────────────────
KST = timezone(timedelta(hours=9))

EXECUTION_COST = 0.0006       # 왕복 수수료+슬리피지
MIN_RR_RATIO = 1.0            # 최소 리스크-리워드 (1:1부터 허용)
BOTTOM_PROXIMITY_PCT = 0.005  # 두 저점/고점 근접도 0.5% (완화)
MIN_PATTERN_HEIGHT_PCT = 0.001  # 패턴 최소 높이 0.1% (완화)
NECKLINE_BUFFER_PCT = 0.0002  # 넥라인 돌파 버퍼 0.02%
VOLUME_CONFIRM_RATIO = 1.0    # 돌파 시 거래량 평균 대비 (1.0 = 사실상 필터 없음)

MAX_TRADES_PER_HOUR = 6
COOLDOWN_BASE = 1             # 기본 쿨다운 (틱)
LOOKBACK_PERIOD = 120         # 패턴 탐색 캔들 수 (2시간)
MIN_TROUGH_DISTANCE = 5       # 두 저점/고점 최소 간격 (5분)
LOCAL_EXTREMA_WINDOW = 2      # 극점 탐색 윈도우 (2분)


@dataclass
class PatternInfo:
    """감지된 패턴 정보."""
    pattern_type: str = ""    # "double_bottom" | "double_top"
    first_price: float = 0.0  # 첫 번째 저점/고점
    second_price: float = 0.0  # 두 번째 저점/고점
    neckline: float = 0.0     # 넥라인 (두 극점 사이의 반대 극점)
    pattern_height: float = 0.0  # 패턴 높이 (넥라인 - 저점 or 고점 - 넥라인)
    strength: float = 0.0      # 패턴 강도 0~1
    first_idx: int = 0
    second_idx: int = 0
    neckline_idx: int = 0


@dataclass
class V12State:
    position_side: str = "NONE"
    entry_price: float = 0.0
    entry_atr: float = 0.0

    cooldown_remaining: int = 0
    consecutive_losses: int = 0
    trades_this_hour: int = 0
    last_hour: int = -1
    total_trades: int = 0
    wins: int = 0
    losses: int = 0

    # 마지막 감지 패턴 (중복 진입 방지)
    last_pattern_idx: int = -1

    def check_trade_limit(self, current_hour: int) -> bool:
        if current_hour != self.last_hour:
            self.last_hour = current_hour
            self.trades_this_hour = 0
        return self.trades_this_hour < MAX_TRADES_PER_HOUR


# ─── 패턴 인식 함수 ─────────────────────────────────────────

def find_local_minima(values: np.ndarray, window: int = LOCAL_EXTREMA_WINDOW) -> list[int]:
    """로컬 최저점 인덱스 반환."""
    minima = []
    for i in range(window, len(values) - window):
        local_slice = values[max(0, i - window):i + window + 1]
        if values[i] == local_slice.min():
            minima.append(i)
    return minima


def find_local_maxima(values: np.ndarray, window: int = LOCAL_EXTREMA_WINDOW) -> list[int]:
    """로컬 최고점 인덱스 반환."""
    maxima = []
    for i in range(window, len(values) - window):
        local_slice = values[max(0, i - window):i + window + 1]
        if values[i] == local_slice.max():
            maxima.append(i)
    return maxima


def detect_double_bottom(
    lows: np.ndarray, highs: np.ndarray, close: np.ndarray
) -> PatternInfo | None:
    """쌍바닥 패턴 감지.

    조건:
    1. 두 개의 로컬 최저점이 가격 근접 (BOTTOM_PROXIMITY_PCT 이내)
    2. 두 저점 사이에 로컬 최고점(넥라인) 존재
    3. 패턴 높이가 최소 기준 충족
    4. 현재 가격이 아직 넥라인 근처 (돌파 직전/직후)
    """
    window = min(LOOKBACK_PERIOD, len(lows))
    recent_lows = lows[-window:]
    recent_highs = highs[-window:]
    recent_close = close[-window:]

    min_indices = find_local_minima(recent_lows)
    max_indices = find_local_maxima(recent_highs)

    if len(min_indices) < 2:
        return None

    # 가장 최근 2개 저점 조합을 역순으로 탐색
    for i in range(len(min_indices) - 1, 0, -1):
        idx2 = min_indices[i]
        idx1 = min_indices[i - 1]

        # 최소 간격 확인
        if idx2 - idx1 < MIN_TROUGH_DISTANCE:
            continue

        low1 = recent_lows[idx1]
        low2 = recent_lows[idx2]

        # 근접도 확인
        avg_low = (low1 + low2) / 2
        if abs(low1 - low2) / avg_low > BOTTOM_PROXIMITY_PCT:
            continue

        # 두 저점 사이의 최고점 (넥라인)
        peaks_between = [m for m in max_indices if idx1 < m < idx2]
        if not peaks_between:
            continue

        neckline_idx = max(peaks_between, key=lambda m: recent_highs[m])
        neckline = recent_highs[neckline_idx]

        # 패턴 높이 확인
        pattern_height = neckline - avg_low
        if pattern_height / avg_low < MIN_PATTERN_HEIGHT_PCT:
            continue

        # 현재 가격이 넥라인 부근인지 (아래에서 접근 or 돌파)
        current_price = recent_close[-1]
        distance_to_neckline = (current_price - neckline) / neckline
        # 넥라인 아래 2%에서 위 1% 사이 (완화)
        if distance_to_neckline < -0.02 or distance_to_neckline > 0.01:
            continue

        # 강도 계산: 두 저점이 가까울수록, 패턴이 클수록 강함
        proximity_score = 1 - abs(low1 - low2) / avg_low / BOTTOM_PROXIMITY_PCT
        height_score = min(1.0, pattern_height / avg_low / 0.01)
        strength = (proximity_score + height_score) / 2

        return PatternInfo(
            pattern_type="double_bottom",
            first_price=low1, second_price=low2,
            neckline=neckline, pattern_height=pattern_height,
            strength=strength,
            first_idx=len(lows) - window + idx1,
            second_idx=len(lows) - window + idx2,
            neckline_idx=len(lows) - window + neckline_idx,
        )

    return None


def detect_double_top(
    lows: np.ndarray, highs: np.ndarray, close: np.ndarray
) -> PatternInfo | None:
    """쌍봉 패턴 감지.

    조건:
    1. 두 개의 로컬 최고점이 가격 근접
    2. 두 고점 사이에 로컬 최저점(넥라인) 존재
    3. 현재 가격이 넥라인 근처 (위에서 접근)
    """
    window = min(LOOKBACK_PERIOD, len(highs))
    recent_lows = lows[-window:]
    recent_highs = highs[-window:]
    recent_close = close[-window:]

    max_indices = find_local_maxima(recent_highs)
    min_indices = find_local_minima(recent_lows)

    if len(max_indices) < 2:
        return None

    for i in range(len(max_indices) - 1, 0, -1):
        idx2 = max_indices[i]
        idx1 = max_indices[i - 1]

        if idx2 - idx1 < MIN_TROUGH_DISTANCE:
            continue

        high1 = recent_highs[idx1]
        high2 = recent_highs[idx2]

        avg_high = (high1 + high2) / 2
        if abs(high1 - high2) / avg_high > BOTTOM_PROXIMITY_PCT:
            continue

        troughs_between = [m for m in min_indices if idx1 < m < idx2]
        if not troughs_between:
            continue

        neckline_idx = min(troughs_between, key=lambda m: recent_lows[m])
        neckline = recent_lows[neckline_idx]

        pattern_height = avg_high - neckline
        if pattern_height / avg_high < MIN_PATTERN_HEIGHT_PCT:
            continue

        current_price = recent_close[-1]
        distance_to_neckline = (neckline - current_price) / neckline
        # 넥라인 위 2%에서 아래 1% 사이 (완화)
        if distance_to_neckline < -0.02 or distance_to_neckline > 0.01:
            continue

        proximity_score = 1 - abs(high1 - high2) / avg_high / BOTTOM_PROXIMITY_PCT
        height_score = min(1.0, pattern_height / avg_high / 0.01)
        strength = (proximity_score + height_score) / 2

        return PatternInfo(
            pattern_type="double_top",
            first_price=high1, second_price=high2,
            neckline=neckline, pattern_height=pattern_height,
            strength=strength,
            first_idx=len(highs) - window + idx1,
            second_idx=len(highs) - window + idx2,
            neckline_idx=len(highs) - window + neckline_idx,
        )

    return None


# ─── 전략 클래스 ──────────────────────────────────────────

@register
class PatternScalper(Strategy):
    """v12 — 쌍바닥/쌍봉 패턴 인식 전략.

    핵심:
    1. 쌍바닥 감지 → 넥라인 돌파 + 거래량 확인 → LONG
    2. 쌍봉 감지 → 넥라인 하향 돌파 + 거래량 확인 → SHORT
    3. SL = 패턴 저점/고점 아래/위, TP = 패턴 높이만큼
    4. 횡보장에서도 작동 (ADX 필터 없음)
    """

    # ATR 기반 SL/TP (패턴 기반과 비교하여 더 작은 값 사용)
    SL_ATR_MULT = 1.5
    TP_ATR_MULT = 2.5

    def __init__(self) -> None:
        self.state = V12State()

    @property
    def name(self) -> str:
        return "pattern_scalper"

    @property
    def label(self) -> str:
        return "v12. Pattern Scalper (쌍바닥/쌍봉)"

    @property
    def description(self) -> str:
        return (
            "쌍바닥/쌍봉 차트 패턴 인식. 넥라인 돌파 + 거래량 확인으로 진입. "
            "횡보장에서도 거래 가능, ADX 필터 없음."
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
            cooldown = COOLDOWN_BASE + self.state.consecutive_losses * 2
            self.state.cooldown_remaining = min(cooldown, 15)
        self.state.position_side = "NONE"

    def evaluate(self, symbol: str, candles: pd.DataFrame,
                 htf_candles: pd.DataFrame | None = None) -> Signal:
        if len(candles) < LOOKBACK_PERIOD + 10:
            return self._hold(symbol, reason="insufficient_data")

        close = candles["close"].values.astype(float)
        high = candles["high"].values.astype(float)
        low = candles["low"].values.astype(float)
        volume = candles["volume"].values.astype(float)
        price = close[-1]

        # ── 쿨다운 ──
        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            return self._hold(symbol, reason="cooldown",
                              remaining=self.state.cooldown_remaining)

        # ── 시간당 매매 제한 ──
        kst_hour = datetime.now(KST).hour
        if not self.state.check_trade_limit(kst_hour):
            return self._hold(symbol, reason="trade_limit")

        # ── 포지션 보유 중: 청산 판단 ──
        if self.state.position_side != "NONE":
            return self._evaluate_exit(symbol, candles, htf_candles, price)

        # ── ATR 계산 ──
        df = candles
        atr_series = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], window=14
        ).average_true_range()
        atr = float(atr_series.iloc[-1])

        if atr <= 0 or pd.isna(atr):
            return self._hold(symbol, reason="zero_atr")

        # ATR이 수수료 대비 너무 작으면 스킵 (극단적 횡보만 차단)
        atr_pct = atr / price
        if atr_pct < EXECUTION_COST * 0.5:
            return self._hold(symbol, reason="atr_too_small")

        # ── 거래량 확인 ──
        vol_avg = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
        vol_ratio = volume[-1] / vol_avg if vol_avg > 0 else 0

        # ── 멀티 타임프레임 패턴 감지 ──
        # 5분봉이 있으면 5분봉에서 패턴 탐색 (더 안정적)
        # 없으면 기본 캔들(1분봉 or 전달된 타임프레임)에서 탐색
        candles_5m = candles.attrs.get("candles_5m") if hasattr(candles, "attrs") else None
        if candles_5m is not None and not candles_5m.empty and len(candles_5m) > LOOKBACK_PERIOD:
            pattern_close = candles_5m["close"].values.astype(float)
            pattern_high = candles_5m["high"].values.astype(float)
            pattern_low = candles_5m["low"].values.astype(float)
            pattern_src = "5m"
        else:
            pattern_close = close
            pattern_high = high
            pattern_low = low
            pattern_src = "base"

        # 15분봉 추세 확인 (패턴 방향 필터)
        htf_bias = None  # None = 필터 없음, "UP" = 상승, "DOWN" = 하락
        if htf_candles is not None and len(htf_candles) > 50:
            htf_close = htf_candles["close"]
            ema8 = htf_close.ewm(span=8, adjust=False).mean().iloc[-1]
            ema21 = htf_close.ewm(span=21, adjust=False).mean().iloc[-1]
            if ema8 > ema21:
                htf_bias = "UP"
            elif ema8 < ema21:
                htf_bias = "DOWN"

        # ── 쌍바닥 감지 → LONG ──
        db_pattern = detect_double_bottom(pattern_low, pattern_high, pattern_close)
        if db_pattern and db_pattern.second_idx != self.state.last_pattern_idx:
            # 넥라인 돌파 확인 (1분봉 가격 기준)
            if price > db_pattern.neckline * (1 + NECKLINE_BUFFER_PCT):
                vol_ok = vol_ratio >= VOLUME_CONFIRM_RATIO
                # 15분봉 추세 일치 시 confidence 보너스
                htf_bonus = 1.2 if htf_bias == "UP" else (0.8 if htf_bias == "DOWN" else 1.0)
                confidence = db_pattern.strength * (0.9 if vol_ok else 0.6) * htf_bonus

                if confidence >= 0.15:
                    sl_price = min(db_pattern.first_price, db_pattern.second_price) - atr * 0.3
                    tp_price = db_pattern.neckline + db_pattern.pattern_height
                    real_rr = (tp_price - price) / (price - sl_price) if price > sl_price else 0

                    if real_rr > 0:  # RR > 0이면 진입 (TP가 현재가 위)
                        self.state.position_side = "LONG"
                        self.state.entry_price = price
                        self.state.entry_atr = atr
                        self.state.trades_this_hour += 1
                        self.state.last_pattern_idx = db_pattern.second_idx

                        return Signal(
                            symbol=symbol, type=SignalType.BUY,
                            confidence=min(confidence, 1.0),
                            source=self.name,
                            metadata={
                                "pattern": "double_bottom",
                                "pattern_src": pattern_src,
                                "htf_bias": htf_bias or "NONE",
                                "first_bottom": round(db_pattern.first_price, 2),
                                "second_bottom": round(db_pattern.second_price, 2),
                                "neckline": round(db_pattern.neckline, 2),
                                "pattern_height": round(db_pattern.pattern_height, 2),
                                "sl_price": round(sl_price, 2),
                                "tp_price": round(tp_price, 2),
                                "real_rr": round(real_rr, 2),
                                "vol_ratio": round(vol_ratio, 2),
                                "strength": round(db_pattern.strength, 2),
                                "atr": round(atr, 2),
                            },
                        )

        # ── 쌍봉 감지 → SHORT ──
        dt_pattern = detect_double_top(pattern_low, pattern_high, pattern_close)
        if dt_pattern and dt_pattern.second_idx != self.state.last_pattern_idx:
            # 넥라인 하향 돌파 확인
            if price < dt_pattern.neckline * (1 - NECKLINE_BUFFER_PCT):
                vol_ok = vol_ratio >= VOLUME_CONFIRM_RATIO
                htf_bonus = 1.2 if htf_bias == "DOWN" else (0.8 if htf_bias == "UP" else 1.0)
                confidence = dt_pattern.strength * (0.9 if vol_ok else 0.6) * htf_bonus

                if confidence >= 0.15:  # 최소 신뢰도 (관대)
                    sl_price = max(dt_pattern.first_price, dt_pattern.second_price) + atr * 0.3
                    tp_price = dt_pattern.neckline - dt_pattern.pattern_height
                    real_rr = (price - tp_price) / (sl_price - price) if sl_price > price else 0

                    if real_rr > 0:  # RR > 0이면 진입
                        self.state.position_side = "SHORT"
                        self.state.entry_price = price
                        self.state.entry_atr = atr
                        self.state.trades_this_hour += 1
                        self.state.last_pattern_idx = dt_pattern.second_idx

                        return Signal(
                            symbol=symbol, type=SignalType.SELL,
                            confidence=min(confidence, 1.0),
                            source=self.name,
                            metadata={
                                "pattern": "double_top",
                                "pattern_src": pattern_src,
                                "htf_bias": htf_bias or "NONE",
                                "first_top": round(dt_pattern.first_price, 2),
                                "second_top": round(dt_pattern.second_price, 2),
                                "neckline": round(dt_pattern.neckline, 2),
                                "pattern_height": round(dt_pattern.pattern_height, 2),
                                "sl_price": round(sl_price, 2),
                                "tp_price": round(tp_price, 2),
                                "real_rr": round(real_rr, 2),
                                "vol_ratio": round(vol_ratio, 2),
                                "strength": round(dt_pattern.strength, 2),
                                "atr": round(atr, 2),
                            },
                        )

        return self._hold(symbol, reason="no_pattern")

    def _evaluate_exit(self, symbol: str, candles: pd.DataFrame,
                       htf_candles: pd.DataFrame | None,
                       price: float) -> Signal:
        """반전 패턴 감지 시 청산."""
        side = self.state.position_side
        close = candles["close"].values.astype(float)
        high = candles["high"].values.astype(float)
        low = candles["low"].values.astype(float)

        # LONG 보유 중 쌍봉 감지 → 청산
        if side == "LONG":
            dt = detect_double_top(low, high, close)
            if dt and price < dt.neckline:
                return Signal(symbol=symbol, type=SignalType.CLOSE,
                              confidence=0.8, source=self.name,
                              metadata={"reason": "double_top_exit"})

        # SHORT 보유 중 쌍바닥 감지 → 청산
        if side == "SHORT":
            db = detect_double_bottom(low, high, close)
            if db and price > db.neckline:
                return Signal(symbol=symbol, type=SignalType.CLOSE,
                              confidence=0.8, source=self.name,
                              metadata={"reason": "double_bottom_exit"})

        # 15분봉 추세 역전 체크
        if htf_candles is not None and len(htf_candles) > 50:
            htf_close = htf_candles["close"]
            ema8 = htf_close.ewm(span=8, adjust=False).mean()
            ema21 = htf_close.ewm(span=21, adjust=False).mean()
            ema50 = htf_close.ewm(span=50, adjust=False).mean()

            if side == "LONG" and ema8.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]:
                return Signal(symbol=symbol, type=SignalType.CLOSE,
                              confidence=0.7, source=self.name,
                              metadata={"reason": "htf_reversal"})
            if side == "SHORT" and ema8.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]:
                return Signal(symbol=symbol, type=SignalType.CLOSE,
                              confidence=0.7, source=self.name,
                              metadata={"reason": "htf_reversal"})

        return self._hold(symbol, reason="hold_position", side=side)

    def _hold(self, symbol: str, reason: str = "", **kwargs) -> Signal:
        return Signal(symbol=symbol, type=SignalType.HOLD, confidence=0.0,
                      source=self.name, metadata={"reason": reason, **kwargs})
