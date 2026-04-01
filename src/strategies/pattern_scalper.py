"""v12. Multi-Pattern Scalper — 멀티 패턴 + 거래량 분석 전략.

7가지 차트 패턴 + 거래량 분석으로 방향 예측:

반전 패턴:
  - 쌍바닥 (Double Bottom) → LONG
  - 쌍봉 (Double Top) → SHORT
  - 역머리어깨 (Inverse H&S) → LONG
  - 머리어깨 (Head & Shoulders) → SHORT

지속 패턴:
  - 상승 깃발 (Bull Flag) → LONG
  - 하락 깃발 (Bear Flag) → SHORT
  - 삼각수렴 돌파 (Triangle) → 방향

거래량 분석:
  - 다이버전스 (가격↑ + 거래량↓ = 약세)
  - 셀링/바잉 클라이맥스
  - OBV 추세

판단 로직:
1. 모든 패턴 스캔 → 가장 강한 패턴 선택
2. 거래량 분석으로 방향 확인/필터
3. 패턴 방향 + 거래량 방향 일치 시 confidence 강화
4. 상충 시 약화 (but 패턴이 우선)
5. HTF(1시간봉) 추세 보너스
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import ta

from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.patterns import scan_all_patterns, analyze_volume, PatternResult
from src.strategies.registry import register


KST = timezone(timedelta(hours=9))
EXECUTION_COST = 0.0006
MAX_TRADES_PER_HOUR = 6
COOLDOWN_BASE = 1


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
    last_pattern_name: str = ""

    def check_trade_limit(self, hour: int) -> bool:
        if hour != self.last_hour:
            self.last_hour = hour
            self.trades_this_hour = 0
        return self.trades_this_hour < MAX_TRADES_PER_HOUR


@register
class PatternScalper(Strategy):
    """v12 — 멀티 패턴 + 거래량 분석 전략.

    7가지 차트 패턴 동시 스캔 + 거래량 다이버전스/OBV 분석.
    횡보장에서도 작동, ADX 필터 없음.
    """

    SL_ATR_MULT = 1.5
    TP_ATR_MULT = 2.5

    def __init__(self) -> None:
        self.state = V12State()

    @property
    def name(self) -> str:
        return "pattern_scalper"

    @property
    def label(self) -> str:
        return "v12. Multi-Pattern Scalper"

    @property
    def description(self) -> str:
        return (
            "7가지 차트 패턴(쌍바닥/쌍봉/머리어깨/깃발/삼각수렴) + "
            "거래량 다이버전스/OBV 분석. 횡보장 가능."
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
        self.state.last_pattern_name = ""

    def evaluate(self, symbol: str, candles: pd.DataFrame,
                 htf_candles: pd.DataFrame | None = None) -> Signal:
        if len(candles) < 130:
            return self._hold(symbol, reason="insufficient_data")

        close = candles["close"].values.astype(float)
        high = candles["high"].values.astype(float)
        low = candles["low"].values.astype(float)
        volume = candles["volume"].values.astype(float)
        price = close[-1]

        # ── 쿨다운 ──
        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            return self._hold(symbol, reason="cooldown")

        # ── 시간당 매매 제한 ──
        kst_hour = datetime.now(KST).hour
        if not self.state.check_trade_limit(kst_hour):
            return self._hold(symbol, reason="trade_limit")

        # ── 포지션 보유 중 → 청산 판단 ──
        if self.state.position_side != "NONE":
            return self._evaluate_exit(symbol, candles, htf_candles, price)

        # ── ATR ──
        atr_series = ta.volatility.AverageTrueRange(
            candles["high"], candles["low"], candles["close"], window=14
        ).average_true_range()
        atr = float(atr_series.iloc[-1])
        if atr <= 0 or pd.isna(atr):
            return self._hold(symbol, reason="zero_atr")

        # ── 1. 패턴 스캔 (5분봉 우선, 없으면 기본 캔들) ──
        candles_5m = candles.attrs.get("candles_5m") if hasattr(candles, "attrs") else None
        if candles_5m is not None and not candles_5m.empty and len(candles_5m) > 60:
            p_close = candles_5m["close"].values.astype(float)
            p_high = candles_5m["high"].values.astype(float)
            p_low = candles_5m["low"].values.astype(float)
            p_vol = candles_5m["volume"].values.astype(float)
            pattern_src = "5m"
        else:
            p_close, p_high, p_low, p_vol = close, high, low, volume
            pattern_src = "base"

        patterns = scan_all_patterns(p_low, p_high, p_close, p_vol, atr)

        if not patterns:
            return self._hold(symbol, reason="no_pattern")

        # 상충 패턴 필터: LONG과 SHORT 패턴이 동시에 있으면 방향별 합산 비교
        long_pats = [p for p in patterns if p.direction == "LONG"]
        short_pats = [p for p in patterns if p.direction == "SHORT"]
        long_score = sum(p.strength for p in long_pats)
        short_score = sum(p.strength for p in short_pats)

        if long_pats and short_pats:
            # 양방향 패턴 동시 → 더 강한 쪽만 남김, 차이 작으면 스킵
            if abs(long_score - short_score) < 0.3:
                return self._hold(symbol, reason="conflicting_patterns",
                                  long_score=round(long_score, 2),
                                  short_score=round(short_score, 2))
            if long_score > short_score:
                patterns = long_pats
            else:
                patterns = short_pats

        # ── 2. 거래량 분석 ──
        vol_signal = analyze_volume(close, volume)

        # ── 3. HTF 추세 ──
        htf_bias = None
        if htf_candles is not None and len(htf_candles) > 30:
            htf_close = htf_candles["close"]
            ema8 = htf_close.ewm(span=8, adjust=False).mean().iloc[-1]
            ema21 = htf_close.ewm(span=21, adjust=False).mean().iloc[-1]
            htf_bias = "LONG" if ema8 > ema21 else "SHORT"

        # ── 4. 최적 패턴 선택 + 방향 결정 ──
        best = patterns[0]

        # 거래량 방향과 패턴 방향 일치도 계산
        vol_agrees = (
            (best.direction == "LONG" and vol_signal.bias == "BULLISH") or
            (best.direction == "SHORT" and vol_signal.bias == "BEARISH")
        )
        vol_conflicts = (
            (best.direction == "LONG" and vol_signal.bias == "BEARISH") or
            (best.direction == "SHORT" and vol_signal.bias == "BULLISH")
        )

        # confidence 계산
        confidence = best.strength
        if vol_agrees:
            confidence *= 1.3  # 거래량 동의 → 보너스
        elif vol_conflicts:
            confidence *= 0.6  # 거래량 반대 → 페널티

        if htf_bias == best.direction:
            confidence *= 1.2  # HTF 동의
        elif htf_bias and htf_bias != best.direction:
            confidence *= 0.8  # HTF 반대

        confidence = min(1.0, confidence)

        if confidence < 0.15:
            return self._hold(symbol, reason="low_confidence",
                              pattern=best.name, confidence=round(confidence, 3))

        # ── 5. RR 체크 ──
        if best.direction == "LONG":
            rr = (best.tp_price - price) / (price - best.sl_price) if price > best.sl_price else 0
        else:
            rr = (price - best.tp_price) / (best.sl_price - price) if best.sl_price > price else 0

        if rr <= 0:
            return self._hold(symbol, reason="bad_rr", pattern=best.name)

        # ── 6. 진입 ──
        self.state.position_side = best.direction
        self.state.entry_price = price
        self.state.entry_atr = atr
        self.state.trades_this_hour += 1
        self.state.last_pattern_name = best.name

        signal_type = SignalType.BUY if best.direction == "LONG" else SignalType.SELL

        return Signal(
            symbol=symbol, type=signal_type,
            confidence=confidence,
            source=self.name,
            metadata={
                "pattern": best.name,
                "pattern_src": pattern_src,
                "direction": best.direction,
                "strength": round(best.strength, 3),
                "sl_price": round(best.sl_price, 2),
                "tp_price": round(best.tp_price, 2),
                "real_rr": round(rr, 2),
                "neckline": round(best.neckline, 2) if best.neckline else 0,
                "pattern_height": round(best.pattern_height, 2),
                "atr": round(atr, 2),
                # 거래량 분석
                "vol_bias": vol_signal.bias,
                "vol_strength": round(vol_signal.strength, 2),
                "vol_signals": vol_signal.signals,
                "vol_agrees": vol_agrees,
                # HTF
                "htf_bias": htf_bias or "NONE",
                # 패턴 상세
                "detail": best.detail or {},
                # 다른 감지된 패턴
                "other_patterns": [p.name for p in patterns[1:]],
            },
        )

    def _evaluate_exit(self, symbol: str, candles: pd.DataFrame,
                       htf_candles: pd.DataFrame | None,
                       price: float) -> Signal:
        """반대 방향 패턴 감지 시 청산."""
        side = self.state.position_side
        close = candles["close"].values.astype(float)
        high = candles["high"].values.astype(float)
        low = candles["low"].values.astype(float)
        volume = candles["volume"].values.astype(float)

        atr_series = ta.volatility.AverageTrueRange(
            candles["high"], candles["low"], candles["close"], window=14
        ).average_true_range()
        atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 1.0

        patterns = scan_all_patterns(low, high, close, volume, atr)

        # 반대 방향 패턴이 있으면 청산
        for p in patterns:
            if side == "LONG" and p.direction == "SHORT" and p.strength > 0.3:
                return Signal(symbol=symbol, type=SignalType.CLOSE,
                              confidence=p.strength, source=self.name,
                              metadata={"reason": f"reverse_{p.name}"})
            if side == "SHORT" and p.direction == "LONG" and p.strength > 0.3:
                return Signal(symbol=symbol, type=SignalType.CLOSE,
                              confidence=p.strength, source=self.name,
                              metadata={"reason": f"reverse_{p.name}"})

        # 거래량 역전 체크
        vol = analyze_volume(close, volume)
        if side == "LONG" and vol.bias == "BEARISH" and vol.strength > 0.6:
            return Signal(symbol=symbol, type=SignalType.CLOSE,
                          confidence=0.6, source=self.name,
                          metadata={"reason": "vol_bearish_reversal",
                                    "vol_signals": vol.signals})
        if side == "SHORT" and vol.bias == "BULLISH" and vol.strength > 0.6:
            return Signal(symbol=symbol, type=SignalType.CLOSE,
                          confidence=0.6, source=self.name,
                          metadata={"reason": "vol_bullish_reversal",
                                    "vol_signals": vol.signals})

        # HTF 추세 역전
        if htf_candles is not None and len(htf_candles) > 30:
            htf_close = htf_candles["close"]
            ema8 = htf_close.ewm(span=8, adjust=False).mean().iloc[-1]
            ema21 = htf_close.ewm(span=21, adjust=False).mean().iloc[-1]
            ema50 = htf_close.ewm(span=50, adjust=False).mean().iloc[-1]
            if side == "LONG" and ema8 < ema21 < ema50:
                return Signal(symbol=symbol, type=SignalType.CLOSE,
                              confidence=0.7, source=self.name,
                              metadata={"reason": "htf_reversal"})
            if side == "SHORT" and ema8 > ema21 > ema50:
                return Signal(symbol=symbol, type=SignalType.CLOSE,
                              confidence=0.7, source=self.name,
                              metadata={"reason": "htf_reversal"})

        return self._hold(symbol, reason="hold_position", side=side)

    def _hold(self, symbol: str, reason: str = "", **kwargs) -> Signal:
        return Signal(symbol=symbol, type=SignalType.HOLD, confidence=0.0,
                      source=self.name, metadata={"reason": reason, **kwargs})
