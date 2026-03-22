"""v4. Aggressive Momentum Rider — 적극적 초단타 스캘핑.

v1~v3의 문제: 너무 보수적 → 거래 자체를 거의 안함
v4 설계 원칙:
- 모멘텀 폭발 감지 즉시 진입 (망설이면 늦음)
- 타이트한 TP/SL로 빠르게 먹고 빠짐
- 횡보장에서도 매매 (역추세 평균회귀)
- 연패 시 방향 전환 (시장이 맞고 내가 틀림)
- 거래 빈도 높음 → 소액 수익 누적
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
import ta

from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.registry import register


class MicroRegime(str, Enum):
    MOMENTUM_UP = "momentum_up"
    MOMENTUM_DOWN = "momentum_down"
    SQUEEZE = "squeeze"        # 횡보 압축 → 곧 터짐
    CHOPPY = "choppy"          # 무질서


@dataclass
class AggressiveState:
    position_side: str = "NONE"
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    cooldown_remaining: int = 0
    flip_mode: bool = False         # 연패 시 방향 반전
    ticks_in_position: int = 0      # 포지션 보유 틱 수
    partial_tp_taken: bool = False
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float("inf")
    entry_atr: float = 0.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0

    def open(self, side: str, price: float, atr: float) -> None:
        self.position_side = side
        self.ticks_in_position = 0
        self.partial_tp_taken = False
        self.highest_since_entry = price
        self.lowest_since_entry = price
        self.entry_atr = atr

    def close(self) -> None:
        self.position_side = "NONE"
        self.partial_tp_taken = False
        self.entry_atr = 0.0

    def update_price(self, price: float) -> None:
        self.highest_since_entry = max(self.highest_since_entry, price)
        self.lowest_since_entry = min(self.lowest_since_entry, price)


# ─── 분석 함수 ────────────────────────────────────────────

def detect_momentum_burst(df: pd.DataFrame, atr: float) -> tuple[str, float]:
    """최근 3봉에서 모멘텀 폭발 감지.
    Returns: (direction, strength) — direction이 NONE이면 폭발 없음."""
    if len(df) < 5 or atr <= 0:
        return "NONE", 0.0

    close = df["close"].values
    # 최근 3봉 가격 변화
    move = close[-1] - close[-4]
    move_atr = abs(move) / atr

    if move_atr >= 0.5:  # 3봉 동안 0.5 ATR 이상 움직임
        direction = "LONG" if move > 0 else "SHORT"
        return direction, move_atr
    return "NONE", move_atr


def detect_big_candle(df: pd.DataFrame) -> tuple[str, float]:
    """직전 캔들이 대형 캔들인지 감지.
    몸통이 전체 범위의 70% 이상이면 강한 확신 캔들."""
    last = df.iloc[-1]
    body = abs(last["close"] - last["open"])
    wick = last["high"] - last["low"]

    if wick <= 0:
        return "NONE", 0.0

    body_ratio = body / wick

    if body_ratio >= 0.5:
        direction = "LONG" if last["close"] > last["open"] else "SHORT"
        return direction, body_ratio
    return "NONE", body_ratio


def detect_squeeze(df: pd.DataFrame) -> bool:
    """볼린저 밴드 스퀴즈 감지 — 밴드폭이 좁아지면 곧 큰 움직임."""
    close = df["close"]
    bb = ta.volatility.BollingerBands(close, window=20)
    bb_mavg = bb.bollinger_mavg()
    width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb_mavg.replace(0, np.nan)

    if len(width.dropna()) < 20:
        return False

    current_width = width.iloc[-1]
    avg_width = width.rolling(50).mean().iloc[-1]

    return current_width < avg_width * 0.6  # 평균 대비 60% 이하면 스퀴즈


def compute_micro_regime(df: pd.DataFrame, atr: float) -> MicroRegime:
    """초단기 시장 상태 판단."""
    burst_dir, burst_str = detect_momentum_burst(df, atr)

    if burst_dir != "NONE" and burst_str >= 1.0:
        return MicroRegime.MOMENTUM_UP if burst_dir == "LONG" else MicroRegime.MOMENTUM_DOWN

    if detect_squeeze(df):
        return MicroRegime.SQUEEZE

    return MicroRegime.CHOPPY


def compute_aggressive_score(
    df: pd.DataFrame, htf: pd.DataFrame | None,
    direction: str, atr: float,
) -> tuple[int, dict]:
    """4개 지표, 2점 이상이면 진입. 빠르고 간결."""
    close = df["close"]
    volume = df["volume"]
    score = 0
    details = {}

    # 1. 초고속 EMA 크로스 — EMA(3) vs EMA(5)
    ema3 = close.ewm(span=3, adjust=False).mean().iloc[-1]
    ema5 = close.ewm(span=5, adjust=False).mean().iloc[-1]
    if (direction == "LONG" and ema3 > ema5) or (direction == "SHORT" and ema3 < ema5):
        score += 1
        details["micro_ema"] = True
    else:
        details["micro_ema"] = False

    # 2. 거래량 스파이크 — 2x 이상, 또는 거래량 데이터 없으면 통과
    vol_avg = volume.rolling(20).mean().iloc[-1]
    if vol_avg <= 0 or volume.iloc[-1] <= 0:
        # 거래량 데이터 없음 (testnet 등) → 조건 스킵, 점수 부여
        score += 1
        details["vol_spike"] = True
        details["vol_ratio"] = 0
        details["vol_note"] = "no_data_pass"
    else:
        vol_ratio = volume.iloc[-1] / vol_avg
        if vol_ratio >= 1.2:
            score += 1
            details["vol_spike"] = True
        else:
            details["vol_spike"] = False
        details["vol_ratio"] = round(float(vol_ratio), 2)

    # 3. 대형 캔들 확인 — 몸통 비율 50% 이상으로 완화
    candle_dir, body_ratio = detect_big_candle(df)
    if candle_dir == direction:
        score += 1
        details["big_candle"] = True
    elif float(body_ratio) >= 0.5 and candle_dir != "NONE":
        # 방향은 다르지만 강한 캔들이면 0.5점... 정수라 패스
        details["big_candle"] = False
    else:
        details["big_candle"] = False
    details["body_ratio"] = round(body_ratio, 2)

    # 4. 15분봉 방향 일치 (있으면 보너스)
    if htf is not None and len(htf) > 20:
        htf_ema3 = htf["close"].ewm(span=3, adjust=False).mean().iloc[-1]
        htf_ema8 = htf["close"].ewm(span=8, adjust=False).mean().iloc[-1]
        if (direction == "LONG" and htf_ema3 > htf_ema8) or \
           (direction == "SHORT" and htf_ema3 < htf_ema8):
            score += 1
            details["htf_agree"] = True
        else:
            details["htf_agree"] = False
    else:
        details["htf_agree"] = False

    return score, details


# ─── 전략 클래스 ───────────────────────────────────────────

@register
class AggressiveMomentumRider(Strategy):
    """v4 — 적극적 모멘텀 라이더.

    원칙:
    - 모멘텀 폭발 감지 → 진입
    - 횡보(스퀴즈) → 돌파 방향 진입
    - TP/SL: 수수료 커버 + 2:1 RR 보장
    - 트레일링으로 수익 극대화
    - 연패 2회 → 방향 반전 (flip mode)
    """

    SL_ATR_MULT = 2.0         # 여유있는 손절 (노이즈에 안 걸림)
    TP_ATR_MULT = 4.0         # 수수료 커버 + 2:1 RR
    PARTIAL_TP_ATR_MULT = 2.0  # 절반 수익에서 부분익절
    TRAILING_ATR_MULT = 3.0   # 3 ATR 수익 시 트레일링 시작
    TRAILING_DIST_ATR = 1.0   # 1 ATR 거리로 추적
    SCORE_THRESHOLD = 2        # 4점 중 2점 (최소한의 확인)

    def __init__(self) -> None:
        self.state = AggressiveState()

    @property
    def name(self) -> str:
        return "aggressive_momentum_rider"

    @property
    def label(self) -> str:
        return "v4. Aggressive Momentum Rider"

    @property
    def description(self) -> str:
        return (
            "적극적 초단타. 모멘텀 폭발 즉시 진입, 타이트 TP/SL(0.5/0.75 ATR). "
            "횡보에서도 스퀴즈 돌파 매매. 쿨다운 없음, 연패 시 방향 반전."
        )

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.SIGNAL_ONLY

    def record_result(self, pnl: float) -> None:
        self.state.total_trades += 1
        # 청산 후 쿨다운: 수익이면 5틱(2.5분), 손실이면 10틱(5분) 대기
        if pnl >= 0:
            self.state.wins += 1
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
            self.state.flip_mode = False
            self.state.cooldown_remaining = 5
        else:
            self.state.losses += 1
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0
            self.state.cooldown_remaining = 10
            # 2연패 → 방향 반전 모드
            if self.state.consecutive_losses >= 2:
                self.state.flip_mode = not self.state.flip_mode
                self.state.consecutive_losses = 0
        self.state.close()

    def evaluate(self, symbol: str, candles: pd.DataFrame,
                 htf_candles: pd.DataFrame | None = None) -> Signal:
        if len(candles) < 30:
            return self._hold(symbol, reason="insufficient_data")

        # ── 쿨다운 (청산 후 대기) ──
        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            return self._hold(symbol, reason="cooldown",
                              remaining=self.state.cooldown_remaining)

        df = candles.copy()
        close = df["close"]
        price = close.iloc[-1]

        # ATR
        atr = ta.volatility.AverageTrueRange(
            df["high"], df["low"], close, window=14
        ).average_true_range().iloc[-1]

        if atr <= 0 or np.isnan(atr):
            return self._hold(symbol, reason="zero_atr")

        # ── 포지션 보유 중: 청산은 엔진이 ATR 기반으로 처리 ──
        if self.state.position_side != "NONE":
            self.state.update_price(price)
            return self._evaluate_exit(symbol, df, price, atr, htf_candles)

        # ── 시장 상태 ──
        regime = compute_micro_regime(df, atr)

        # ── 방향 결정 ──
        direction = self._determine_direction(df, atr, regime)
        if direction == "NONE":
            return self._hold(symbol, reason="no_direction", regime=regime.value)

        # ── 연패 반전 모드 ──
        if self.state.flip_mode:
            direction = "SHORT" if direction == "LONG" else "LONG"

        # ── 점수 체크 ──
        score, details = compute_aggressive_score(df, htf_candles, direction, atr)

        if score < self.SCORE_THRESHOLD:
            return self._hold(symbol, reason="low_score",
                              score=score, details=details, regime=regime.value)

        # ── 진입 ──
        self.state.open(direction, price, atr)
        signal_type = SignalType.BUY if direction == "LONG" else SignalType.SELL
        confidence = score / 4.0

        return Signal(
            symbol=symbol, type=signal_type, confidence=confidence,
            source=self.name,
            metadata={
                "direction": direction,
                "regime": regime.value,
                "score": score,
                "details": details,
                "atr": round(atr, 2),
                "flip_mode": self.state.flip_mode,
                "consecutive_losses": self.state.consecutive_losses,
                "consecutive_wins": self.state.consecutive_wins,
                "sl_atr_mult": self.SL_ATR_MULT,
                "tp_atr_mult": self.TP_ATR_MULT,
                "partial_tp_atr_mult": self.PARTIAL_TP_ATR_MULT,
            },
        )

    def _determine_direction(self, df: pd.DataFrame, atr: float,
                             regime: MicroRegime) -> str:
        """시장 상태별 방향 결정."""
        close = df["close"]

        if regime in (MicroRegime.MOMENTUM_UP, MicroRegime.MOMENTUM_DOWN):
            # 모멘텀 방향 그대로
            burst_dir, _ = detect_momentum_burst(df, atr)
            return burst_dir

        if regime == MicroRegime.SQUEEZE:
            # 스퀴즈: 직전 캔들 방향으로 (돌파 기대)
            candle_dir, _ = detect_big_candle(df)
            if candle_dir != "NONE":
                return candle_dir
            # 대형 캔들 없으면 EMA 방향
            ema3 = close.ewm(span=3, adjust=False).mean().iloc[-1]
            ema5 = close.ewm(span=5, adjust=False).mean().iloc[-1]
            return "LONG" if ema3 > ema5 else "SHORT"

        if regime == MicroRegime.CHOPPY:
            # 무질서: RSI 극단에서 역추세 (평균회귀)
            rsi = ta.momentum.RSIIndicator(close, window=7).rsi().iloc[-1]
            if rsi < 25:
                return "LONG"
            elif rsi > 75:
                return "SHORT"
            # RSI 중립이면 모멘텀 방향
            ema3 = close.ewm(span=3, adjust=False).mean().iloc[-1]
            ema5 = close.ewm(span=5, adjust=False).mean().iloc[-1]
            return "LONG" if ema3 > ema5 else "SHORT"

        return "NONE"

    def _evaluate_exit(self, symbol: str, df: pd.DataFrame, price: float,
                       atr: float, htf: pd.DataFrame | None) -> Signal:
        """전략 기반 추가 청산 판단. 기본 TP/SL은 엔진이 처리."""
        side = self.state.position_side

        # 진입 후 최소 보유: 10틱(30초×10=5분) 동안은 전략 청산 안함
        # (엔진의 ATR 기반 SL/TP는 여전히 작동)
        self.state.ticks_in_position += 1
        if self.state.ticks_in_position < 10:
            return self._hold(symbol, reason="min_hold", side=side,
                              ticks=self.state.ticks_in_position)

        # 모멘텀 반전 감지 — 강한 반전(2 ATR 이상)만 청산
        burst_dir, burst_str = detect_momentum_burst(df, atr)
        if burst_str >= 2.0:
            if (side == "LONG" and burst_dir == "SHORT") or \
               (side == "SHORT" and burst_dir == "LONG"):
                return Signal(
                    symbol=symbol, type=SignalType.CLOSE, confidence=0.9,
                    source=self.name,
                    metadata={"reason": "momentum_reversal", "burst_strength": round(burst_str, 2)},
                )

        return self._hold(symbol, reason="hold", side=side)

    def _hold(self, symbol: str, reason: str = "", **kwargs) -> Signal:
        return Signal(symbol=symbol, type=SignalType.HOLD, confidence=0.0,
                      source=self.name, metadata={"reason": reason, **kwargs})
