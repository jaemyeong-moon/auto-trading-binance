"""v9. 가이드 기반 전략 — "유리할 때만 베팅하고, 나머지는 쉬어라"

구조:
1. EMA50/200 추세 판단 → 한 방향만 거래
2. EMA20 눌림목 진입
3. ATR 변동성 필터 → 횡보 스킵
4. 1:2 손익비 고정
5. 1% 리스크 포지션 사이징
6. 수수료+슬리피지 사전 차감
7. 추세 없으면 쉬어라
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import ta

from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.registry import register


LEVERAGE = 5
RISK_PCT = 0.01           # 잔고의 1% 리스크
RR_RATIO = 2.0            # 1:2 손익비
TOTAL_COST_PCT = 0.001    # 수수료 0.08% + 슬리피지 0.02% = 0.1%
SL_ATR_MULT = 2.0         # SL = 2 ATR
MAX_DAILY_TRADES = 10
COOLDOWN_WIN = 12          # 1분 (5초×12)
COOLDOWN_LOSS = 60         # 5분


@dataclass
class V9State:
    position_side: str = "NONE"
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    quantity: float = 0.0
    risk_amt: float = 0.0

    # 엔진 호환
    entry_atr: float = 0.0
    partial_tp_taken: bool = False
    highest_since_entry: float = 0.0
    lowest_since_entry: float = float("inf")

    cooldown_remaining: int = 0
    daily_trades: int = 0
    last_trade_day: int = 0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0

    def open(self, side: str, price: float, sl: float, tp: float,
             qty: float, risk: float, atr: float) -> None:
        self.position_side = side
        self.entry_price = price
        self.sl_price = sl
        self.tp_price = tp
        self.quantity = qty
        self.risk_amt = risk
        self.entry_atr = atr
        self.highest_since_entry = price
        self.lowest_since_entry = price
        self.partial_tp_taken = False

    def close(self) -> None:
        self.position_side = "NONE"
        self.entry_price = 0.0
        self.entry_atr = 0.0

    def update_price(self, price: float) -> None:
        self.highest_since_entry = max(self.highest_since_entry, price)
        self.lowest_since_entry = min(self.lowest_since_entry, price)


@register
class AggressiveMomentumRider(Strategy):
    """v9 — 유리할 때만 베팅.

    EMA50/200 추세 → EMA20 눌림목 → RSI 확인 → 1:2 RR → 1% 리스크.
    추세 없으면 쉰다. 수수료 사전 차감.
    """

    # 엔진 ATR SL/TP 비활성화 (자체 관리)
    SL_ATR_MULT = 99.0
    TP_ATR_MULT = 99.0
    PARTIAL_TP_ATR_MULT = 99.0
    TRAILING_ATR_MULT = 99.0
    TRAILING_DIST_ATR = 99.0

    def __init__(self) -> None:
        self.state = V9State()

    @property
    def name(self) -> str:
        return "aggressive_momentum_rider"

    @property
    def label(self) -> str:
        return "v9. Trend Pullback"

    @property
    def description(self) -> str:
        return "EMA50/200 추세 + EMA20 눌림목 진입. 1:2 RR, 1% 리스크. 추세 없으면 쉰다."

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.SIGNAL_ONLY

    def record_result(self, pnl: float) -> None:
        self.state.total_trades += 1
        self.state.daily_trades += 1
        if pnl >= 0:
            self.state.wins += 1
            self.state.cooldown_remaining = COOLDOWN_WIN
        else:
            self.state.losses += 1
            self.state.cooldown_remaining = COOLDOWN_LOSS
        self.state.close()

    def evaluate(self, symbol: str, candles: pd.DataFrame,
                 htf_candles: pd.DataFrame | None = None) -> Signal:
        # 15분봉이 핵심 — 없으면 판단 불가
        if htf_candles is None or len(htf_candles) < 210:
            return self._hold(symbol, reason="insufficient_htf")

        from datetime import datetime
        today = datetime.now().day
        if self.state.last_trade_day != today:
            self.state.daily_trades = 0
            self.state.last_trade_day = today

        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            return self._hold(symbol, reason="cooldown")

        if self.state.daily_trades >= MAX_DAILY_TRADES:
            return self._hold(symbol, reason="daily_limit")

        # 1분봉 현재가
        price = float(candles.iloc[-1]["close"])

        # ── 포지션 보유 중: SL/TP 체크 ──
        if self.state.position_side != "NONE":
            self.state.update_price(price)
            return self._manage_position(symbol, price, candles)

        # ══ 15분봉 분석 ══
        close_15 = htf_candles["close"].astype(float)
        high_15 = htf_candles["high"].astype(float)
        low_15 = htf_candles["low"].astype(float)

        ema50 = close_15.ewm(span=50, adjust=False).mean()
        ema200 = close_15.ewm(span=200, adjust=False).mean()
        ema20 = close_15.ewm(span=20, adjust=False).mean()
        atr_15 = ta.volatility.AverageTrueRange(
            high_15, low_15, close_15, window=14
        ).average_true_range()
        rsi_15 = ta.momentum.RSIIndicator(close_15, window=14).rsi()

        e50 = float(ema50.iloc[-1])
        e200 = float(ema200.iloc[-1])
        e20 = float(ema20.iloc[-1])
        atr = float(atr_15.iloc[-1])
        rsi = float(rsi_15.iloc[-1])
        htf_price = float(close_15.iloc[-1])
        htf_low = float(low_15.iloc[-1])
        htf_high = float(high_15.iloc[-1])

        if atr <= 0 or np.isnan(atr):
            return self._hold(symbol, reason="zero_atr")

        # ── 1. 추세 판단 ──
        if e50 > e200:
            bias = "LONG"
        elif e50 < e200:
            bias = "SHORT"
        else:
            return self._hold(symbol, reason="no_trend",
                              ema50=round(e50, 2), ema200=round(e200, 2))

        # ── 2. 변동성 필터 ──
        atr_pct = atr / htf_price
        if atr_pct < TOTAL_COST_PCT * 3:
            return self._hold(symbol, reason="low_volatility",
                              atr_pct=round(atr_pct * 100, 4))

        # ── 3. 눌림목 감지 ──
        if bias == "LONG":
            pullback = htf_low <= e20 * 1.002 and htf_price > e20
            rsi_ok = 35 < rsi < 60
        else:
            pullback = htf_high >= e20 * 0.998 and htf_price < e20
            rsi_ok = 40 < rsi < 65

        if not pullback:
            return self._hold(symbol, reason="no_pullback",
                              bias=bias, price_vs_ema20=round((htf_price/e20-1)*100, 3))

        if not rsi_ok:
            return self._hold(symbol, reason="rsi_filter",
                              bias=bias, rsi=round(rsi, 1))

        # ── 4. 1분봉 모멘텀 확인 (정밀 타이밍) ──
        if len(candles) > 10:
            ema5_1m = candles["close"].ewm(span=5, adjust=False).mean().iloc[-1]
            ema13_1m = candles["close"].ewm(span=13, adjust=False).mean().iloc[-1]
            if bias == "LONG" and ema5_1m < ema13_1m:
                return self._hold(symbol, reason="1m_not_ready", bias=bias)
            if bias == "SHORT" and ema5_1m > ema13_1m:
                return self._hold(symbol, reason="1m_not_ready", bias=bias)

        # ── 5. 수수료 커버 체크 ──
        sl_dist = atr * SL_ATR_MULT
        tp_dist = sl_dist * RR_RATIO
        expected_profit_pct = tp_dist / price
        if expected_profit_pct < TOTAL_COST_PCT:
            return self._hold(symbol, reason="fee_not_covered")

        # ── 6. 포지션 사이징 (1% 리스크) ──
        # 잔고는 엔진이 넘겨줌 — 여기서는 signal만 생성
        if bias == "LONG":
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist

        self.state.open(bias, price, sl, tp, 0, 0, atr)

        return Signal(
            symbol=symbol,
            type=SignalType.BUY if bias == "LONG" else SignalType.SELL,
            confidence=0.8,
            source=self.name,
            metadata={
                "direction": bias,
                "entry_type": "pullback",
                "sl": round(sl, 2),
                "tp": round(tp, 2),
                "atr": round(atr, 2),
                "rsi": round(rsi, 1),
                "ema50": round(e50, 2),
                "ema200": round(e200, 2),
                "rr": RR_RATIO,
            },
        )

    def _manage_position(self, symbol: str, price: float,
                         candles: pd.DataFrame) -> Signal:
        """SL/TP + 모멘텀 기반 판단."""
        side = self.state.position_side
        entry = self.state.entry_price
        sl = self.state.sl_price
        tp = self.state.tp_price

        # SL 도달
        if (side == "LONG" and price <= sl) or \
           (side == "SHORT" and price >= sl):
            return Signal(
                symbol=symbol, type=SignalType.CLOSE, confidence=0.95,
                source=self.name,
                metadata={"reason": "stop_loss", "entry": round(entry, 2)},
            )

        # TP 도달 → 모멘텀 분석
        if (side == "LONG" and price >= tp) or \
           (side == "SHORT" and price <= tp):

            # 모멘텀 지속 여부 확인
            momentum = self._check_momentum(side, candles)
            if momentum > 0:
                # 모멘텀 지속 → TP 연장
                atr = self.state.entry_atr
                if side == "LONG":
                    self.state.tp_price = tp + atr
                else:
                    self.state.tp_price = tp - atr
                return self._hold(symbol, reason="tp_extended",
                                  new_tp=round(self.state.tp_price, 2))
            else:
                # 모멘텀 약화 → 익절
                return Signal(
                    symbol=symbol, type=SignalType.CLOSE, confidence=0.9,
                    source=self.name,
                    metadata={
                        "reason": "take_profit",
                        "entry": round(entry, 2),
                        "momentum": round(momentum, 3),
                    },
                )

        # 수익 중 + 모멘텀 반전 → 조기 익절 (수수료 넘는 수익만)
        if side == "LONG":
            change = (price - entry) / entry
        else:
            change = (entry - price) / entry

        if change > TOTAL_COST_PCT:
            momentum = self._check_momentum(side, candles)
            if momentum < -0.3:
                return Signal(
                    symbol=symbol, type=SignalType.CLOSE, confidence=0.85,
                    source=self.name,
                    metadata={
                        "reason": "early_tp",
                        "net_pct": round((change - TOTAL_COST_PCT) * 100, 3),
                    },
                )

        return self._hold(symbol, reason="hold",
                          change_pct=round(change * 100, 3))

    def _check_momentum(self, side: str, candles: pd.DataFrame) -> float:
        """최근 캔들의 모멘텀. 양수=방향 지속, 음수=반전."""
        if len(candles) < 10:
            return 0.0

        close = candles["close"].values[-10:].astype(float)
        recent = float(np.mean(close[-3:]))
        prev = float(np.mean(close[-6:-3]))

        if prev <= 0:
            return 0.0

        change = (recent - prev) / prev * 100

        if side == "LONG":
            return change
        else:
            return -change

    def _hold(self, symbol: str, reason: str = "", **kwargs) -> Signal:
        return Signal(symbol=symbol, type=SignalType.HOLD, confidence=0.0,
                      source=self.name, metadata={"reason": reason, **kwargs})
