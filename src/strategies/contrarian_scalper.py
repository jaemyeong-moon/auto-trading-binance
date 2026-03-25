"""v10. 역추세 전략 — "v9가 틀렸으니, 반대로 간다"

v9 분석:
- EMA50>200 → LONG 진입 → 손실
- EMA50<200 → SHORT 진입 → 손실
- 눌림목 진입 → 반등 안 하고 추세 이탈

v10 핵심 반전:
1. 추세 반대 방향 진입 (평균회귀 전략)
   - EMA50>200 (상승추세) → 과열 구간에서 SHORT
   - EMA50<200 (하락추세) → 과매도 구간에서 LONG
2. 눌림목 대신 과열/과매도 감지
   - RSI 극단값 (>70 SHORT, <30 LONG)
3. 타이트한 SL + 넉넉한 TP (평균회귀는 빠르게 먹고 빠진다)
4. 수수료 사전 차감 유지
5. 15분봉 기준 (1분봉 노이즈 최소화)
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import ta

from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.registry import register


LEVERAGE = 5
RISK_PCT = 0.01           # 잔고의 1% 리스크
RR_RATIO = 1.5            # 1:1.5 손익비 (평균회귀는 빠르게)
TOTAL_COST_PCT = 0.001    # 수수료+슬리피지 0.1%
SL_ATR_MULT = 1.5         # SL = 1.5 ATR (v9의 2.0보다 타이트)
MAX_DAILY_TRADES = 8
COOLDOWN_WIN = 6           # 30초 (5초×6)
COOLDOWN_LOSS = 36         # 3분


@dataclass
class V10State:
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
    trailing_stop_price: float | None = None

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
        self.trailing_stop_price = None

    def close(self) -> None:
        self.position_side = "NONE"
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.trailing_stop_price = None

    def update_price(self, price: float) -> None:
        self.highest_since_entry = max(self.highest_since_entry, price)
        self.lowest_since_entry = min(self.lowest_since_entry, price)


@register
class ContrarianScalper(Strategy):
    """v10 — 역추세 스캘퍼.

    v9가 추세 따라가서 지면, v10은 반대로 간다.
    과열/과매도 구간에서 평균회귀를 노린다.
    """

    # 엔진 ATR SL/TP 비활성화 (자체 관리)
    SL_ATR_MULT = 99.0
    TP_ATR_MULT = 99.0
    PARTIAL_TP_ATR_MULT = 99.0
    TRAILING_ATR_MULT = 99.0
    TRAILING_DIST_ATR = 99.0

    def __init__(self) -> None:
        self.state = V10State()

    @property
    def name(self) -> str:
        return "contrarian_scalper"

    @property
    def label(self) -> str:
        return "v10. Contrarian Scalper"

    @property
    def description(self) -> str:
        return "역추세 전략 — 과열 시 SHORT, 과매도 시 LONG. v9 반대 방향. 1:1.5 RR, 1% 리스크."

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
        if htf_candles is None or len(htf_candles) < 210:
            return self._hold(symbol, reason="insufficient_htf")

        today = datetime.now().day
        if self.state.last_trade_day != today:
            self.state.daily_trades = 0
            self.state.last_trade_day = today

        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            return self._hold(symbol, reason="cooldown")

        if self.state.daily_trades >= MAX_DAILY_TRADES:
            return self._hold(symbol, reason="daily_limit")

        price = float(candles.iloc[-1]["close"])

        # ── 포지션 보유 중 ──
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

        # 볼린저 밴드 (과열/과매도 판단)
        bb = ta.volatility.BollingerBands(close_15, window=20, window_dev=2)
        bb_upper = float(bb.bollinger_hband().iloc[-1])
        bb_lower = float(bb.bollinger_lband().iloc[-1])
        bb_mid = float(bb.bollinger_mavg().iloc[-1])

        e50 = float(ema50.iloc[-1])
        e200 = float(ema200.iloc[-1])
        e20 = float(ema20.iloc[-1])
        atr = float(atr_15.iloc[-1])
        rsi = float(rsi_15.iloc[-1])
        htf_price = float(close_15.iloc[-1])

        if atr <= 0 or np.isnan(atr):
            return self._hold(symbol, reason="zero_atr")

        # ── 1. 추세 판단 (v9의 반대!) ──
        # v9: EMA50>200 → LONG | v10: EMA50>200 → SHORT (과열)
        # v9: EMA50<200 → SHORT | v10: EMA50<200 → LONG (과매도)
        if e50 > e200:
            trend = "UP"
            bias = "SHORT"  # 반대!
        elif e50 < e200:
            trend = "DOWN"
            bias = "LONG"   # 반대!
        else:
            return self._hold(symbol, reason="no_trend")

        # ── 2. 변동성 필터 (최소한의 움직임 필요) ──
        atr_pct = atr / htf_price
        if atr_pct < TOTAL_COST_PCT * 2.5:
            return self._hold(symbol, reason="low_volatility",
                              atr_pct=round(atr_pct * 100, 4))

        # ── 3. 과열/과매도 감지 (v9 눌림목 대신) ──
        if bias == "SHORT":
            # 상승추세에서 과열 감지: RSI 높고 + 볼린저 상단 터치
            overextended = rsi > 65 and htf_price > bb_upper * 0.998
            # 추가: 가격이 EMA20보다 많이 이탈 (과열)
            distance_from_ema = (htf_price - e20) / e20
            far_from_mean = distance_from_ema > 0.003
        else:
            # 하락추세에서 과매도 감지: RSI 낮고 + 볼린저 하단 터치
            overextended = rsi < 35 and htf_price < bb_lower * 1.002
            distance_from_ema = (e20 - htf_price) / e20
            far_from_mean = distance_from_ema > 0.003

        if not overextended:
            return self._hold(symbol, reason="not_overextended",
                              bias=bias, rsi=round(rsi, 1),
                              bb_pos="above" if bias == "SHORT" else "below")

        if not far_from_mean:
            return self._hold(symbol, reason="too_close_to_mean",
                              dist=round(distance_from_ema * 100, 3))

        # ── 4. 1분봉 반전 신호 확인 ──
        if len(candles) > 10:
            ema5_1m = candles["close"].ewm(span=5, adjust=False).mean()
            ema13_1m = candles["close"].ewm(span=13, adjust=False).mean()
            e5 = float(ema5_1m.iloc[-1])
            e13 = float(ema13_1m.iloc[-1])

            # v9 반대: 모멘텀이 꺾이기 시작할 때 진입
            if bias == "SHORT" and e5 > e13 * 1.001:
                # 아직 상승 모멘텀이 강함 → 좀 더 기다림
                return self._hold(symbol, reason="1m_momentum_too_strong", bias=bias)
            if bias == "LONG" and e5 < e13 * 0.999:
                # 아직 하락 모멘텀이 강함 → 좀 더 기다림
                return self._hold(symbol, reason="1m_momentum_too_strong", bias=bias)

        # ── 5. 수수료 커버 체크 ──
        sl_dist = atr * SL_ATR_MULT
        tp_dist = sl_dist * RR_RATIO
        expected_profit_pct = tp_dist / price
        if expected_profit_pct < TOTAL_COST_PCT:
            return self._hold(symbol, reason="fee_not_covered")

        # ── 6. SL/TP 계산 ──
        if bias == "SHORT":
            sl = price + sl_dist
            tp = price - tp_dist  # 평균회귀 목표: EMA20 방향
            # TP를 EMA20 근처로 제한 (욕심 금지)
            tp = max(tp, bb_mid * 0.999)
        else:
            sl = price - sl_dist
            tp = price + tp_dist
            tp = min(tp, bb_mid * 1.001)

        self.state.open(bias, price, sl, tp, 0, 0, atr)

        return Signal(
            symbol=symbol,
            type=SignalType.BUY if bias == "LONG" else SignalType.SELL,
            confidence=0.75,
            source=self.name,
            metadata={
                "direction": bias,
                "entry_type": "contrarian",
                "trend": trend,
                "sl": round(sl, 2),
                "tp": round(tp, 2),
                "atr": round(atr, 2),
                "rsi": round(rsi, 1),
                "bb_upper": round(bb_upper, 2),
                "bb_lower": round(bb_lower, 2),
                "distance_from_ema": round(distance_from_ema * 100, 3),
                "rr": RR_RATIO,
            },
        )

    def _manage_position(self, symbol: str, price: float,
                         candles: pd.DataFrame) -> Signal:
        """SL/TP + 평균회귀 도달 판단."""
        side = self.state.position_side
        entry = self.state.entry_price
        sl = self.state.sl_price
        tp = self.state.tp_price
        atr = self.state.entry_atr

        # SL 도달
        if (side == "LONG" and price <= sl) or \
           (side == "SHORT" and price >= sl):
            return Signal(
                symbol=symbol, type=SignalType.CLOSE, confidence=0.95,
                source=self.name,
                metadata={"reason": "stop_loss", "entry": round(entry, 2)},
            )

        # TP 도달
        if (side == "LONG" and price >= tp) or \
           (side == "SHORT" and price <= tp):
            return Signal(
                symbol=symbol, type=SignalType.CLOSE, confidence=0.9,
                source=self.name,
                metadata={"reason": "take_profit", "entry": round(entry, 2)},
            )

        # 수익 중 + 반전 모멘텀 약화 → 빠른 익절 (평균회귀 특성)
        if side == "LONG":
            change = (price - entry) / entry
        else:
            change = (entry - price) / entry

        # 수수료 이상 수익 + 반전 모멘텀 소진 → 조기 익절
        if change > TOTAL_COST_PCT * 2:
            momentum = self._check_momentum(side, candles)
            # 평균회귀는 모멘텀이 원래 추세로 돌아가면 즉시 탈출
            if momentum < -0.15:
                return Signal(
                    symbol=symbol, type=SignalType.CLOSE, confidence=0.85,
                    source=self.name,
                    metadata={
                        "reason": "mean_revert_done",
                        "net_pct": round((change - TOTAL_COST_PCT) * 100, 3),
                    },
                )

        # 트레일링 스탑 (수익 0.3% 이상 시 활성화)
        if change > 0.003:
            trail_dist = atr * 0.8 / entry if atr > 0 else 0.002
            if side == "LONG":
                new_stop = self.state.highest_since_entry * (1 - trail_dist)
                if self.state.trailing_stop_price is None or new_stop > self.state.trailing_stop_price:
                    self.state.trailing_stop_price = new_stop
                if price <= self.state.trailing_stop_price:
                    return Signal(
                        symbol=symbol, type=SignalType.CLOSE, confidence=0.88,
                        source=self.name,
                        metadata={"reason": "trailing_stop",
                                  "pct": round(change * 100, 3)},
                    )
            else:
                new_stop = self.state.lowest_since_entry * (1 + trail_dist)
                if self.state.trailing_stop_price is None or new_stop < self.state.trailing_stop_price:
                    self.state.trailing_stop_price = new_stop
                if price >= self.state.trailing_stop_price:
                    return Signal(
                        symbol=symbol, type=SignalType.CLOSE, confidence=0.88,
                        source=self.name,
                        metadata={"reason": "trailing_stop",
                                  "pct": round(change * 100, 3)},
                    )

        return self._hold(symbol, reason="hold",
                          change_pct=round(change * 100, 3))

    def _check_momentum(self, side: str, candles: pd.DataFrame) -> float:
        if len(candles) < 10:
            return 0.0
        close = candles["close"].values[-10:].astype(float)
        recent = float(np.mean(close[-3:]))
        prev = float(np.mean(close[-6:-3]))
        if prev <= 0:
            return 0.0
        change = (recent - prev) / prev * 100
        return change if side == "LONG" else -change

    def _hold(self, symbol: str, reason: str = "", **kwargs) -> Signal:
        return Signal(symbol=symbol, type=SignalType.HOLD, confidence=0.0,
                      source=self.name, metadata={"reason": reason, **kwargs})
