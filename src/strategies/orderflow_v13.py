"""v13. OrderFlow Scalper — 멀티 TF × 오더북 × 펀딩비 전략.

진입 결정 흐름:
  1. 멀티 TF 추세 일치 (5m/15m/1h EMA 방향 합산)
  2. 오더북 불균형 (bid_ask_ratio > 1.2 → LONG 유리, < 0.83 → SHORT 유리)
  3. 펀딩비 극단치 역방향 (funding_rate_signal 역행 포지션)
  4. OI 다이버전스 보조 확인
  5. 스코어 합산 → 임계치(3점 이상) 시 진입

오더북/펀딩비/OI 데이터 주입 방식: candles.attrs["orderbook"], candles.attrs["funding_rates"],
candles.attrs["oi_current"], candles.attrs["oi_prev"] 로 주입.
엔진 또는 테스트에서 candles.attrs 에 세팅하면 됨.
"""

from __future__ import annotations

import pandas as pd
import ta

from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.features.orderbook import bid_ask_ratio
from src.strategies.features.derivatives import funding_rate_signal, oi_divergence_signal, oi_change_rate
from src.strategies.registry import register

# ── 진입 임계치 ──
ENTRY_SCORE_THRESHOLD = 3      # 최소 3점 이상 시 진입
BID_ASK_LONG_THRESHOLD = 1.2   # bid > ask * 1.2 → LONG 유리
BID_ASK_SHORT_THRESHOLD = 0.83 # bid < ask * 0.83 → SHORT 유리 (1/1.2)

# EMA 기간
EMA_FAST = 9
EMA_SLOW = 21


@register
class OrderFlowScalper(Strategy):
    """v13 — 멀티 TF × 오더북 × 펀딩비 전략."""

    LEVERAGE = 5
    POSITION_SIZE_PCT = 0.15
    MAX_HOLD_HOURS = 6.0
    SL_ATR_MULT = 2.0
    TP_ATR_MULT = 4.0
    TIMEFRAMES = ["5m", "15m", "1h"]

    @property
    def name(self) -> str:
        return "orderflow_v13"

    @property
    def label(self) -> str:
        return "v13. OrderFlow Scalper"

    @property
    def description(self) -> str:
        return (
            "멀티 TF 추세(EMA) + 오더북 불균형 + 펀딩비 극단치 역방향 + OI 다이버전스 "
            "4가지 시그널 합산으로 포트폴리오 다변화."
        )

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.SIGNAL_ONLY

    # ─────────────────────────────────────────────
    # evaluate
    # ─────────────────────────────────────────────

    def evaluate(
        self,
        symbol: str,
        candles: pd.DataFrame,
        htf_candles: pd.DataFrame | None = None,
    ) -> Signal:
        """멀티 TF × 오더북 × 펀딩비 복합 신호 평가.

        오더북/펀딩비/OI 데이터는 candles.attrs 로 주입:
          attrs["orderbook"]     = {"bids": [[price, qty], ...], "asks": [...]}
          attrs["funding_rates"] = [float, ...]   # 최근 N회 (마지막이 최신)
          attrs["oi_current"]    = float           # 현재 OI
          attrs["oi_prev"]       = float           # 직전 OI
        """
        if len(candles) < 50:
            return self._hold(symbol, reason="insufficient_data")

        close = candles["close"].values.astype(float)
        high = candles["high"].values.astype(float)
        low = candles["low"].values.astype(float)
        price = close[-1]

        # ── ATR ──
        atr_series = ta.volatility.AverageTrueRange(
            candles["high"], candles["low"], candles["close"], window=14
        ).average_true_range()
        atr = float(atr_series.iloc[-1])
        if atr <= 0 or pd.isna(atr):
            return self._hold(symbol, reason="zero_atr")

        # ─────────────────────────────────────────
        # 1. 멀티 TF 추세 (주 TF + HTF 기반 EMA)
        # ─────────────────────────────────────────
        tf_score, tf_direction = self._multi_tf_score(candles, htf_candles)

        if tf_direction is None:
            return self._hold(symbol, reason="no_tf_consensus")

        # ─────────────────────────────────────────
        # 2. 오더북 불균형
        # ─────────────────────────────────────────
        ob_score, ob_ratio = self._orderbook_score(candles, tf_direction)

        # ─────────────────────────────────────────
        # 3. 펀딩비 극단치 역방향
        # ─────────────────────────────────────────
        fr_score, fr_signal = self._funding_score(candles, tf_direction)

        # ─────────────────────────────────────────
        # 4. OI 다이버전스 보조
        # ─────────────────────────────────────────
        oi_score, oi_div = self._oi_score(candles, close, tf_direction)

        # ─────────────────────────────────────────
        # 5. 스코어 합산
        # ─────────────────────────────────────────
        total_score = tf_score + ob_score + fr_score + oi_score
        # 최대 7점: tf(3) + ob(2) + fr(1) + oi(1)

        if total_score < ENTRY_SCORE_THRESHOLD:
            return self._hold(
                symbol, reason="low_score",
                score=total_score, threshold=ENTRY_SCORE_THRESHOLD,
                tf_direction=tf_direction,
            )

        # ─────────────────────────────────────────
        # 6. 진입 신호 생성
        # ─────────────────────────────────────────
        confidence = min(1.0, total_score / 7.0)

        signal_type = SignalType.BUY if tf_direction == "LONG" else SignalType.SELL

        if tf_direction == "LONG":
            sl_price = price - atr * self.SL_ATR_MULT
            tp_price = price + atr * self.TP_ATR_MULT
        else:
            sl_price = price + atr * self.SL_ATR_MULT
            tp_price = price - atr * self.TP_ATR_MULT

        return Signal(
            symbol=symbol,
            type=signal_type,
            confidence=round(confidence, 3),
            source=self.name,
            metadata={
                "direction": tf_direction,
                "total_score": total_score,
                "tf_score": tf_score,
                "ob_score": ob_score,
                "fr_score": fr_score,
                "oi_score": oi_score,
                "ob_ratio": round(ob_ratio, 3),
                "fr_signal": fr_signal,
                "oi_divergence": oi_div,
                "sl_price": round(sl_price, 4),
                "tp_price": round(tp_price, 4),
                "atr": round(atr, 4),
            },
        )

    # ─────────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────────

    def _multi_tf_score(
        self,
        candles: pd.DataFrame,
        htf_candles: pd.DataFrame | None,
    ) -> tuple[int, str | None]:
        """주 TF(candles) + HTF(htf_candles) EMA 방향으로 추세 합산.

        주 TF EMA 방향 → 1점
        HTF EMA 방향 (있을 경우) → 1점
        두 TF 모두 같은 방향이면 보너스 1점 (합산 3점 가능)

        Returns:
            (score, direction | None)
            direction 은 최소 1점 이상일 때 결정; 0점이면 None 반환.
        """
        close_series = pd.Series(candles["close"].values.astype(float))
        ema_fast = close_series.ewm(span=EMA_FAST, adjust=False).mean().iloc[-1]
        ema_slow = close_series.ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1]
        primary_dir = "LONG" if ema_fast > ema_slow else "SHORT"
        score = 1

        htf_dir: str | None = None
        if htf_candles is not None and len(htf_candles) >= 30:
            htf_close = pd.Series(htf_candles["close"].values.astype(float))
            htf_ema_fast = htf_close.ewm(span=EMA_FAST, adjust=False).mean().iloc[-1]
            htf_ema_slow = htf_close.ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1]
            htf_dir = "LONG" if htf_ema_fast > htf_ema_slow else "SHORT"
            score += 1
            if htf_dir == primary_dir:
                score += 1  # 두 TF 방향 일치 보너스

        # HTF 가 반대 방향이면 score 를 1 감소 (최소 0점)
        if htf_dir is not None and htf_dir != primary_dir:
            score -= 1

        if score <= 0:
            return 0, None

        return score, primary_dir

    def _orderbook_score(
        self,
        candles: pd.DataFrame,
        direction: str,
    ) -> tuple[int, float]:
        """오더북 불균형 점수.

        bid_ask_ratio >= BID_ASK_LONG_THRESHOLD → LONG 쪽 가중치 2점
        bid_ask_ratio <= BID_ASK_SHORT_THRESHOLD → SHORT 쪽 가중치 2점
        그 외 → 0점

        Returns:
            (score, ratio)
        """
        attrs = getattr(candles, "attrs", {})
        ob = attrs.get("orderbook")
        if ob is None:
            return 0, 1.0

        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            return 0, 1.0

        ratio = bid_ask_ratio(bids, asks)

        if direction == "LONG" and ratio >= BID_ASK_LONG_THRESHOLD:
            return 2, ratio
        if direction == "SHORT" and ratio <= BID_ASK_SHORT_THRESHOLD:
            return 2, ratio

        return 0, ratio

    def _funding_score(
        self,
        candles: pd.DataFrame,
        direction: str,
    ) -> tuple[int, str]:
        """펀딩비 극단치 역방향 점수.

        funding_rate_signal 이 "SHORT" 이고 direction 이 "SHORT" → 1점
        funding_rate_signal 이 "LONG"  이고 direction 이 "LONG"  → 1점
        NEUTRAL 또는 방향 불일치 → 0점

        Returns:
            (score, fr_signal_str)
        """
        attrs = getattr(candles, "attrs", {})
        rates = attrs.get("funding_rates", [])

        fr = funding_rate_signal(rates)

        if fr == "NEUTRAL":
            return 0, fr

        if fr == direction:
            return 1, fr

        return 0, fr

    def _oi_score(
        self,
        candles: pd.DataFrame,
        close: "np.ndarray",  # type: ignore[name-defined]
        direction: str,
    ) -> tuple[int, str]:
        """OI 다이버전스 보조 점수.

        BULLISH_CONFIRM + direction LONG   → 1점
        BULLISH_DIV     + direction LONG   → 1점 (OI 감소=청산 후 반등 기대)
        BEARISH_CONFIRM + direction SHORT  → 1점
        BEARISH_DIV     + direction SHORT  → 1점
        그 외 → 0점

        Returns:
            (score, div_signal)
        """
        attrs = getattr(candles, "attrs", {})
        oi_current = attrs.get("oi_current")
        oi_prev = attrs.get("oi_prev")

        if oi_current is None or oi_prev is None:
            return 0, "N/A"

        price_now = float(close[-1])
        price_prev = float(close[-2]) if len(close) >= 2 else price_now
        price_change_pct = (price_now - price_prev) / price_prev * 100.0 if price_prev != 0 else 0.0

        oi_chg = oi_change_rate(float(oi_current), float(oi_prev))
        div = oi_divergence_signal(price_change_pct, oi_chg)

        if direction == "LONG" and div in ("BULLISH_CONFIRM", "BULLISH_DIV"):
            return 1, div
        if direction == "SHORT" and div in ("BEARISH_CONFIRM", "BEARISH_DIV"):
            return 1, div

        return 0, div

    # ─────────────────────────────────────────────
    # 공통 헬퍼
    # ─────────────────────────────────────────────

    def _hold(self, symbol: str, reason: str = "", **kwargs) -> Signal:
        return Signal(
            symbol=symbol,
            type=SignalType.HOLD,
            confidence=0.0,
            source=self.name,
            metadata={"reason": reason, **kwargs},
        )
