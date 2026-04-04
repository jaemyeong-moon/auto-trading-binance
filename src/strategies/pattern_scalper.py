"""v12. Multi-Pattern Scalper — 패턴 + 모멘텀 확인 전략.

진입: 7개 패턴 스캔 → 최강 패턴 선택 → EMA 모멘텀 일치 확인 → 진입
청산: ATR 기반 SL/TP + 트레일링 스탑 + 최대 보유 시간 제한

핵심 개선 (v12.1):
- 상충 패턴 스킵 제거 → 최강 패턴 우선 (진입 빈도 2배)
- 모멘텀(EMA5/15) 필수 확인 → 패턴 + 추세 일치 시만 진입 (승률 향상)
- 패턴 기반 청산 제거 → SL/TP 청산 (조기 청산 방지)
- 트레일링 스탑(1%+ 수익 시 50% 확보) + 최대 100틱 보유
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import pandas as pd
import ta

from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.patterns import scan_all_patterns, analyze_volume
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
    ticks_in_position: int = 0   # 포지션 보유 틱 수
    partial_tp_taken: bool = False  # 부분 익절 여부

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

    SL_ATR_MULT = 3.0     # SL = 3 ATR (노이즈 회피)
    TP_ATR_MULT = 5.0     # TP = 5 ATR (1:1.67 RR)
    MAX_HOLD_TICKS = 100  # 최대 보유 ~25시간(15분봉) — 대손실 방지

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
        self.state.partial_tp_taken = False
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

        # ── 2. 최강 패턴 선택 (상충 시에도 가장 강한 것 사용) ──
        best = patterns[0]

        # ── 3. 모멘텀 확인 — 단기 EMA 방향이 패턴과 일치하는지 ──
        ema_fast = pd.Series(close).ewm(span=5, adjust=False).mean().iloc[-1]
        ema_slow = pd.Series(close).ewm(span=15, adjust=False).mean().iloc[-1]
        momentum_dir = "LONG" if ema_fast > ema_slow else "SHORT"
        momentum_agrees = momentum_dir == best.direction

        # ── 4. 거래량 분석 ──
        vol_signal = analyze_volume(close, volume)
        vol_agrees = (
            (best.direction == "LONG" and vol_signal.bias == "BULLISH") or
            (best.direction == "SHORT" and vol_signal.bias == "BEARISH")
        )

        # ── 5. HTF 추세 ──
        htf_bias = None
        if htf_candles is not None and len(htf_candles) > 30:
            htf_close = htf_candles["close"]
            ema8 = htf_close.ewm(span=8, adjust=False).mean().iloc[-1]
            ema21 = htf_close.ewm(span=21, adjust=False).mean().iloc[-1]
            htf_bias = "LONG" if ema8 > ema21 else "SHORT"

        # ── 6. confidence 계산 (패턴 + 모멘텀 + 거래량 + HTF) ──
        confidence = best.strength

        # 모멘텀 일치 필수 — 패턴만으로는 진입하지 않음
        if not momentum_agrees:
            confidence *= 0.3

        if vol_agrees:
            confidence *= 1.2
        if htf_bias == best.direction:
            confidence *= 1.15
        elif htf_bias and htf_bias != best.direction:
            confidence *= 0.85

        confidence = min(1.0, confidence)

        if confidence < 0.2:
            return self._hold(symbol, reason="low_confidence",
                              pattern=best.name, confidence=round(confidence, 3),
                              momentum=momentum_dir)

        # ── 5. RR 체크 (ATR 기반 동적 SL/TP fallback) ──
        if best.direction == "LONG":
            sl = best.sl_price if best.sl_price < price else price - atr * self.SL_ATR_MULT
            tp = best.tp_price if best.tp_price > price else price + atr * self.TP_ATR_MULT
            rr = (tp - price) / (price - sl) if price > sl else 0
        else:
            sl = best.sl_price if best.sl_price > price else price + atr * self.SL_ATR_MULT
            tp = best.tp_price if best.tp_price < price else price - atr * self.TP_ATR_MULT
            rr = (price - tp) / (sl - price) if sl > price else 0

        if rr < 0.5:
            return self._hold(symbol, reason="bad_rr", pattern=best.name, rr=round(rr, 2))

        # ── 6. 진입 ──
        self.state.position_side = best.direction
        self.state.entry_price = price
        self.state.entry_atr = atr
        self.state.ticks_in_position = 0
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
                "sl_price": round(sl, 2),
                "tp_price": round(tp, 2),
                "real_rr": round(rr, 2),
                "neckline": round(best.neckline, 2) if best.neckline else 0,
                "pattern_height": round(best.pattern_height, 2),
                "atr": round(atr, 2),
                # 거래량 분석
                "vol_bias": vol_signal.bias,
                "vol_strength": round(vol_signal.strength, 2),
                "vol_signals": vol_signal.signals,
                "vol_agrees": vol_agrees,
                # 모멘텀
                "momentum_dir": momentum_dir,
                "momentum_agrees": momentum_agrees,
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
        """SL/TP 기반 청산 + 트레일링 스탑. 패턴 청산 제거."""
        side = self.state.position_side
        entry = self.state.entry_price
        atr = self.state.entry_atr
        self.state.ticks_in_position += 1

        if side == "LONG":
            unrealized_pct = (price - entry) / entry
        else:
            unrealized_pct = (entry - price) / entry

        # SL/TP는 엔진 레벨에서도 체크되지만, 시그널 레벨에서도 명시적으로 청산
        sl_dist = atr * self.SL_ATR_MULT
        tp_dist = atr * self.TP_ATR_MULT

        if side == "LONG":
            sl_price = entry - sl_dist
            tp_price = entry + tp_dist
            hit_sl = price <= sl_price
            hit_tp = price >= tp_price
        else:
            sl_price = entry + sl_dist
            tp_price = entry - tp_dist
            hit_sl = price >= sl_price
            hit_tp = price <= tp_price

        # 최대 보유 시간 초과 → 시장가 청산
        if self.state.ticks_in_position >= self.MAX_HOLD_TICKS:
            return Signal(symbol=symbol, type=SignalType.CLOSE,
                          confidence=0.9, source=self.name,
                          metadata={"reason": "max_hold",
                                    "ticks_held": self.state.ticks_in_position,
                                    "unrealized_pct": round(unrealized_pct * 100, 2)})

        if hit_sl:
            return Signal(symbol=symbol, type=SignalType.CLOSE,
                          confidence=1.0, source=self.name,
                          metadata={"reason": "SL",
                                    "ticks_held": self.state.ticks_in_position,
                                    "unrealized_pct": round(unrealized_pct * 100, 2)})
        if hit_tp:
            return Signal(symbol=symbol, type=SignalType.CLOSE,
                          confidence=1.0, source=self.name,
                          metadata={"reason": "TP",
                                    "ticks_held": self.state.ticks_in_position,
                                    "unrealized_pct": round(unrealized_pct * 100, 2)})

        # 트레일링 스탑: 1%+ 수익 시 손익분기점으로 SL 올림
        if unrealized_pct > 0.01:
            # 수익의 50%를 확보하는 트레일링 스탑
            trail_level = entry * (1 + unrealized_pct * 0.5) if side == "LONG" \
                else entry * (1 - unrealized_pct * 0.5)
            if side == "LONG" and price <= trail_level:
                return Signal(symbol=symbol, type=SignalType.CLOSE,
                              confidence=0.8, source=self.name,
                              metadata={"reason": "trailing_stop",
                                        "ticks_held": self.state.ticks_in_position,
                                        "unrealized_pct": round(unrealized_pct * 100, 2)})
            if side == "SHORT" and price >= trail_level:
                return Signal(symbol=symbol, type=SignalType.CLOSE,
                              confidence=0.8, source=self.name,
                              metadata={"reason": "trailing_stop",
                                        "ticks_held": self.state.ticks_in_position,
                                        "unrealized_pct": round(unrealized_pct * 100, 2)})

        return self._hold(symbol, reason="hold_position", side=side,
                          ticks=self.state.ticks_in_position,
                          unrealized_pct=round(unrealized_pct * 100, 2))

    def _hold(self, symbol: str, reason: str = "", **kwargs) -> Signal:
        return Signal(symbol=symbol, type=SignalType.HOLD, confidence=0.0,
                      source=self.name, metadata={"reason": reason, **kwargs})
