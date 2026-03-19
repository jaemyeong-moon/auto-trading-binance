"""v3. Smart Momentum Scalper — 실전 괴리 최소화 전략.

v1/v2의 실전 성능 저하 원인 분석:
1. 고정 % TP/SL → 변동성과 무관하게 같은 기준 적용
2. 슬리피지/스프레드 미반영 → 백테스트에서 이상적 체결 가정
3. 지표 후행성 → 신호 시점에 이미 움직임 끝남
4. 과다매매 → 수수료가 수익을 갉아먹음

v3 핵심 설계 원칙:
- ATR 기반 동적 TP/SL (시장 호흡에 맞춤)
- 최소 2:1 리스크-리워드 보장 (기대값 양수)
- 슬리피지 버퍼 내장 (진입 전 수익성 사전 검증)
- 매매 빈도 제한 (수수료 절감)
- ADX로 진짜 추세만 매매 (가짜 신호 필터)
- VWAP 앵커 (기관 매매 수준 참고)
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
import ta

from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.registry import register


# ─── 상수 ──────────────────────────────────────────────────

COMMISSION_PCT = 0.0004    # 왕복 수수료 0.04%
SLIPPAGE_PCT = 0.0002     # 예상 슬리피지 0.02%
EXECUTION_COST = COMMISSION_PCT + SLIPPAGE_PCT  # 총 실행 비용 0.06%
MIN_RR_RATIO = 2.0        # 최소 리스크-리워드 비율


class Regime(str, Enum):
    STRONG_TREND = "strong_trend"     # ADX > 30, EMA 정렬
    WEAK_TREND = "weak_trend"         # ADX 20-30
    RANGING = "ranging"               # ADX < 20
    VOLATILE = "volatile"             # ATR 급등


@dataclass
class SmartState:
    position_side: str = "NONE"
    cooldown_remaining: int = 0
    consecutive_losses: int = 0
    partial_tp_taken: bool = False
    trailing_stop_price: float | None = None
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float("inf")
    entry_atr: float = 0.0            # 진입 시점 ATR (TP/SL 기준)
    trades_this_hour: int = 0          # 시간당 매매 제한
    last_hour: int = -1
    total_trades: int = 0
    wins: int = 0
    losses: int = 0

    def tick_cooldown(self) -> bool:
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return True
        return False

    def open(self, side: str, price: float, atr: float) -> None:
        self.position_side = side
        self.partial_tp_taken = False
        self.trailing_stop_price = None
        self.highest_since_entry = price
        self.lowest_since_entry = price
        self.entry_atr = atr

    def close(self) -> None:
        self.position_side = "NONE"
        self.partial_tp_taken = False
        self.trailing_stop_price = None
        self.entry_atr = 0.0

    def update_price(self, price: float) -> None:
        self.highest_since_entry = max(self.highest_since_entry, price)
        self.lowest_since_entry = min(self.lowest_since_entry, price)

    def check_trade_limit(self, current_hour: int, max_per_hour: int = 4) -> bool:
        """시간당 매매 횟수 제한. True면 매매 가능."""
        if current_hour != self.last_hour:
            self.last_hour = current_hour
            self.trades_this_hour = 0
        return self.trades_this_hour < max_per_hour

    def record_trade(self) -> None:
        self.trades_this_hour += 1


# ─── 시장 분석 함수들 ─────────────────────────────────────

def compute_regime(df_htf: pd.DataFrame) -> tuple[Regime, dict]:
    """15분봉으로 시장 체제 판단. ADX 중심."""
    close = df_htf["close"]
    high = df_htf["high"]
    low = df_htf["low"]

    # ADX — 추세의 '강도'를 측정 (방향 아님)
    adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
    adx = adx_ind.adx().iloc[-1]
    plus_di = adx_ind.adx_pos().iloc[-1]
    minus_di = adx_ind.adx_neg().iloc[-1]

    # ATR 변동성
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    atr_now = atr.iloc[-1]
    atr_avg = atr.rolling(20).mean().iloc[-1]
    atr_ratio = atr_now / atr_avg if atr_avg > 0 else 1.0

    # EMA 정렬
    ema8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]

    info = {
        "adx": round(adx, 1),
        "plus_di": round(plus_di, 1),
        "minus_di": round(minus_di, 1),
        "atr": round(atr_now, 2),
        "atr_ratio": round(atr_ratio, 2),
        "ema8": round(ema8, 2),
        "ema21": round(ema21, 2),
    }

    if atr_ratio > 2.0:
        return Regime.VOLATILE, info
    if adx > 30:
        return Regime.STRONG_TREND, info
    if adx > 20:
        return Regime.WEAK_TREND, info
    return Regime.RANGING, info


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP 계산 — 기관 가격 앵커."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumvol = df["volume"].cumsum()
    cumtp = (typical_price * df["volume"]).cumsum()
    return cumtp / cumvol


def compute_momentum_acceleration(close: pd.Series, period: int = 5) -> float:
    """모멘텀 가속도: 최근 변화율의 변화율.
    양수 = 가속 상승, 음수 = 가속 하락."""
    roc = close.pct_change(period)
    roc_change = roc.diff()
    return roc_change.iloc[-1]


def compute_entry_score_v3(
    df_1m: pd.DataFrame,
    df_htf: pd.DataFrame | None,
    direction: str,
    regime: Regime,
    regime_info: dict,
) -> tuple[int, int, dict]:
    """7개 지표 기반 점수. (점수, 만점, 상세) 반환."""
    close = df_1m["close"]
    volume = df_1m["volume"]
    score = 0
    max_score = 7
    details = {}

    # 1. ADX 방향 확인 (+1) — DI+와 DI- 비교
    if direction == "LONG" and regime_info["plus_di"] > regime_info["minus_di"]:
        score += 1
        details["adx_direction"] = True
    elif direction == "SHORT" and regime_info["minus_di"] > regime_info["plus_di"]:
        score += 1
        details["adx_direction"] = True
    else:
        details["adx_direction"] = False

    # 2. 15분봉 추세 동의 (+1)
    htf_agrees = False
    if df_htf is not None and len(df_htf) > 30:
        htf_ema8 = df_htf["close"].ewm(span=8, adjust=False).mean().iloc[-1]
        htf_ema21 = df_htf["close"].ewm(span=21, adjust=False).mean().iloc[-1]
        if (direction == "LONG" and htf_ema8 > htf_ema21) or \
           (direction == "SHORT" and htf_ema8 < htf_ema21):
            score += 1
            htf_agrees = True
    details["htf_agree"] = htf_agrees

    # 3. VWAP 위치 (+1) — LONG이면 VWAP 아래서 진입, SHORT이면 위에서
    vwap = compute_vwap(df_1m)
    vwap_val = vwap.iloc[-1]
    price = close.iloc[-1]
    if (direction == "LONG" and price < vwap_val) or \
       (direction == "SHORT" and price > vwap_val):
        score += 1
        details["vwap_favorable"] = True
    else:
        details["vwap_favorable"] = False
    details["vwap"] = round(vwap_val, 2)

    # 4. 거래량 폭발 (+1) — 1.5x 이상 (v2보다 엄격)
    vol_avg = volume.rolling(20).mean().iloc[-1]
    vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 0
    if vol_ratio > 1.5:
        score += 1
        details["volume_surge"] = True
    else:
        details["volume_surge"] = False
    details["vol_ratio"] = round(vol_ratio, 2)

    # 5. 모멘텀 가속 (+1) — 단순 방향이 아닌 '가속'
    accel = compute_momentum_acceleration(close)
    if (direction == "LONG" and accel > 0) or (direction == "SHORT" and accel < 0):
        score += 1
        details["momentum_accel"] = True
    else:
        details["momentum_accel"] = False
    details["accel"] = round(accel * 10000, 2)  # bps 단위

    # 6. RSI 유리 영역 (+1) — 추세 매매: 과매도/과매수 방향으로
    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    if direction == "LONG" and 35 < rsi < 65:
        score += 1
        details["rsi_zone"] = True
    elif direction == "SHORT" and 35 < rsi < 65:
        score += 1
        details["rsi_zone"] = True
    else:
        details["rsi_zone"] = False
    details["rsi"] = round(rsi, 1)

    # 7. MACD 히스토그램 강화 (+1) — 히스토그램이 커지는 중
    macd = ta.trend.MACD(close)
    hist = macd.macd_diff()
    hist_now = hist.iloc[-1]
    hist_prev = hist.iloc[-2]
    hist_growing = (direction == "LONG" and hist_now > hist_prev > 0) or \
                   (direction == "SHORT" and hist_now < hist_prev < 0)
    if hist_growing:
        score += 1
        details["macd_growing"] = True
    else:
        details["macd_growing"] = False

    return score, max_score, details


# ─── v3 전략 클래스 ────────────────────────────────────────

@register
class SmartMomentumScalper(Strategy):
    """v3 — 실전 괴리를 최소화한 스마트 스캘핑.

    핵심 차별점:
    1. ATR 기반 동적 TP/SL (시장 호흡 반영)
    2. 리스크-리워드 2:1 미만이면 진입 안함
    3. 실행 비용(수수료+슬리피지) 사전 차감
    4. 시간당 매매 횟수 제한 (과다매매 방지)
    5. ADX 기반 진짜 추세만 매매
    6. VWAP 앵커로 유리한 가격대에서만 진입
    7. 모멘텀 '가속도' 확인 (후행성 완화)
    """

    # ATR 배수 기반 TP/SL
    SL_ATR_MULT = 1.0       # 손절 = 1 × ATR
    TP_ATR_MULT = 2.0       # 익절 = 2 × ATR (최소 2:1 RR)
    TRAILING_ATR_MULT = 1.5  # 트레일링 활성화 = 1.5 × ATR
    TRAILING_DIST_ATR = 0.5  # 트레일링 거리 = 0.5 × ATR
    PARTIAL_TP_ATR_MULT = 1.0  # 부분익절 = 1 × ATR

    SCORE_THRESHOLD = 5      # 7점 만점 중 5점
    MAX_TRADES_PER_HOUR = 4  # 시간당 최대 매매
    COOLDOWN_BASE = 5        # 기본 쿨다운 봉 수

    def __init__(self) -> None:
        self.state = SmartState()

    @property
    def name(self) -> str:
        return "smart_momentum_scalper"

    @property
    def label(self) -> str:
        return "v3. Smart Momentum Scalper"

    @property
    def description(self) -> str:
        return (
            "ATR 동적 TP/SL + 리스크-리워드 2:1 보장. "
            "ADX 추세 강도 + VWAP 가격 앵커 + 모멘텀 가속도. "
            "실행비용 사전 차감, 시간당 매매 제한, 과다매매 방지."
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
            # 연패가 심할수록 쿨다운 증가
            cooldown = self.COOLDOWN_BASE + (self.state.consecutive_losses * 2)
            self.state.cooldown_remaining = min(cooldown, 20)
        self.state.close()

    def evaluate(self, symbol: str, candles: pd.DataFrame,
                 htf_candles: pd.DataFrame | None = None) -> Signal:
        if len(candles) < 100:
            return self._hold(symbol, reason="insufficient_data")

        df = candles.copy()
        close = df["close"]
        price = close.iloc[-1]
        current_hour = df.index[-1].hour if hasattr(df.index[-1], "hour") else 0

        # ── 쿨다운 ──
        if self.state.tick_cooldown():
            return self._hold(symbol, reason="cooldown",
                              remaining=self.state.cooldown_remaining)

        # ── 시간당 매매 제한 ──
        if not self.state.check_trade_limit(current_hour, self.MAX_TRADES_PER_HOUR):
            return self._hold(symbol, reason="trade_limit_reached",
                              trades_this_hour=self.state.trades_this_hour)

        # ── ATR 계산 ──
        atr = ta.volatility.AverageTrueRange(
            df["high"], df["low"], close, window=14
        ).average_true_range().iloc[-1]

        if atr <= 0:
            return self._hold(symbol, reason="zero_atr")

        # ATR이 너무 작으면 수수료 대비 수익 불가능
        atr_pct = atr / price
        min_profitable_move = EXECUTION_COST * 3  # 수수료의 3배는 움직여야 의미
        if atr_pct < min_profitable_move:
            return self._hold(symbol, reason="atr_too_small",
                              atr_pct=f"{atr_pct:.4%}",
                              min_needed=f"{min_profitable_move:.4%}")

        # ── 시장 체제 판단 (15분봉) ──
        if htf_candles is not None and len(htf_candles) > 40:
            regime, regime_info = compute_regime(htf_candles)
        else:
            # 15분봉 없으면 1분봉으로 대체
            regime, regime_info = compute_regime(df)

        # ── 매매 금지 구간 ──
        if regime == Regime.RANGING:
            return self._hold(symbol, reason="ranging_market", regime_info=regime_info)

        if regime == Regime.VOLATILE:
            return self._hold(symbol, reason="too_volatile", regime_info=regime_info)

        # ── 포지션 보유 중 ──
        if self.state.position_side != "NONE":
            return self._evaluate_exit(symbol, df, price, atr, regime, htf_candles)

        # ── 방향 결정 — ADX DI 기반 ──
        if regime_info["plus_di"] > regime_info["minus_di"]:
            direction = "LONG"
        else:
            direction = "SHORT"

        # ── 진입 점수 ──
        score, max_score, details = compute_entry_score_v3(
            df, htf_candles, direction, regime, regime_info
        )

        if score < self.SCORE_THRESHOLD:
            return self._hold(symbol, reason="low_score",
                              score=score, max_score=max_score,
                              threshold=self.SCORE_THRESHOLD,
                              details=details, regime=regime.value)

        # ── 리스크-리워드 사전 검증 ──
        expected_sl = atr * self.SL_ATR_MULT
        expected_tp = atr * self.TP_ATR_MULT
        # 실행 비용 차감 후 실질 기대 수익
        real_tp = expected_tp - (price * EXECUTION_COST)
        real_sl = expected_sl + (price * EXECUTION_COST)
        real_rr = real_tp / real_sl if real_sl > 0 else 0

        if real_rr < MIN_RR_RATIO:
            return self._hold(symbol, reason="poor_risk_reward",
                              real_rr=round(real_rr, 2),
                              expected_tp=round(expected_tp, 2),
                              expected_sl=round(expected_sl, 2))

        # ── 진입 ──
        self.state.open(direction, price, atr)
        self.state.record_trade()
        signal_type = SignalType.BUY if direction == "LONG" else SignalType.SELL
        confidence = score / max_score

        return Signal(
            symbol=symbol, type=signal_type, confidence=confidence,
            source=self.name,
            metadata={
                "direction": direction,
                "regime": regime.value,
                "score": score, "max_score": max_score,
                "details": details,
                "atr": round(atr, 2),
                "atr_pct": f"{atr_pct:.4%}",
                "dynamic_sl": round(expected_sl, 2),
                "dynamic_tp": round(expected_tp, 2),
                "real_rr": round(real_rr, 2),
                "sl_atr_mult": self.SL_ATR_MULT,
                "tp_atr_mult": self.TP_ATR_MULT,
                "partial_tp_atr_mult": self.PARTIAL_TP_ATR_MULT,
                "trailing_activate_atr_mult": self.TRAILING_ATR_MULT,
                "trailing_distance_atr_mult": self.TRAILING_DIST_ATR,
            },
        )

    def _evaluate_exit(self, symbol: str, df: pd.DataFrame, price: float,
                       atr: float, regime: Regime,
                       htf_candles: pd.DataFrame | None) -> Signal:
        """ATR 기반 동적 TP/SL + 트레일링."""
        self.state.update_price(price)
        side = self.state.position_side
        entry_atr = self.state.entry_atr if self.state.entry_atr > 0 else atr

        # 진입 시점 ATR 기준으로 TP/SL 계산 (진입 후 변하지 않음)
        sl_distance = entry_atr * self.SL_ATR_MULT
        tp_distance = entry_atr * self.TP_ATR_MULT
        partial_tp_distance = entry_atr * self.PARTIAL_TP_ATR_MULT
        trailing_activate = entry_atr * self.TRAILING_ATR_MULT
        trailing_dist = entry_atr * self.TRAILING_DIST_ATR

        # 현재 손익 (절대값)
        if side == "LONG":
            pnl_distance = price - self.state.lowest_since_entry
        else:
            pnl_distance = self.state.highest_since_entry - price

        # ── 추세 역전 (15분봉 기준) ──
        if htf_candles is not None and len(htf_candles) > 30:
            htf_regime, _ = compute_regime(htf_candles)
            if side == "LONG" and htf_regime == Regime.RANGING:
                return Signal(symbol=symbol, type=SignalType.CLOSE, confidence=0.7,
                              source=self.name, metadata={"reason": "regime_shift_ranging"})
            if side == "SHORT" and htf_regime == Regime.RANGING:
                return Signal(symbol=symbol, type=SignalType.CLOSE, confidence=0.7,
                              source=self.name, metadata={"reason": "regime_shift_ranging"})

        # ── 트레일링 스탑 ──
        if self.state.trailing_stop_price is not None:
            if side == "LONG" and price <= self.state.trailing_stop_price:
                return Signal(symbol=symbol, type=SignalType.CLOSE, confidence=0.9,
                              source=self.name, metadata={"reason": "trailing_stop"})
            if side == "SHORT" and price >= self.state.trailing_stop_price:
                return Signal(symbol=symbol, type=SignalType.CLOSE, confidence=0.9,
                              source=self.name, metadata={"reason": "trailing_stop"})

            # 트레일링 갱신
            if side == "LONG":
                new_stop = self.state.highest_since_entry - trailing_dist
                if new_stop > self.state.trailing_stop_price:
                    self.state.trailing_stop_price = new_stop
            else:
                new_stop = self.state.lowest_since_entry + trailing_dist
                if new_stop < self.state.trailing_stop_price:
                    self.state.trailing_stop_price = new_stop

        # ── 트레일링 활성화 ──
        profit = self.state.highest_since_entry - price if side == "SHORT" else \
                 price - self.state.lowest_since_entry
        # 이건 정확하지 않으므로 mark price 기반으로 재계산
        # 엔진에서 entry_price를 기준으로 체크하므로 여기서는 시장 상태 기반 판단만

        if self.state.trailing_stop_price is None and profit > trailing_activate:
            if side == "LONG":
                self.state.trailing_stop_price = self.state.highest_since_entry - trailing_dist
            else:
                self.state.trailing_stop_price = self.state.lowest_since_entry + trailing_dist

        return self._hold(symbol, reason="hold_position",
                          side=side, regime=regime.value,
                          entry_atr=round(entry_atr, 2))

    def _hold(self, symbol: str, reason: str = "", **kwargs) -> Signal:
        return Signal(symbol=symbol, type=SignalType.HOLD, confidence=0.0,
                      source=self.name, metadata={"reason": reason, **kwargs})
