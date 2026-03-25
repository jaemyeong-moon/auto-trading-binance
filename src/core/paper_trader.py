"""Paper Trader — 전략별 가상매매 시스템.

모든 등록된 전략을 실시간 데이터로 동시 실행하고,
가상 잔고/포지션/거래를 DB에 기록한다.
실제 주문 없이 전략 성과를 비교 검증하는 용도.
"""

import structlog

from src.core import database as db
from src.core.database import (
    PaperBalance, PaperPosition, PaperTrade, get_session,
)
from src.core.models import SignalType
from src.strategies.registry import list_strategies, get_strategy
from src.strategies.base import Strategy
from src.utils.timezone import now_kst

import ta
import numpy as np
import pandas as pd

logger = structlog.get_logger()

INITIAL_BALANCE = 200.0
LEVERAGE = 5
POSITION_PCT = 1.0            # 잔고 100% 투자
FEE_RATE = 0.0004             # 편도 0.04% (메이커/테이커 평균)
SLIPPAGE_RATE = 0.0002        # 슬리피지 0.02%
TOTAL_COST_RATE = (FEE_RATE + SLIPPAGE_RATE) * 2  # 왕복 총 비용 0.12%
SYMBOLS = ["BTCUSDT", "ETHUSDT"]


class PaperTrader:
    """모든 전략을 가상매매로 동시 실행."""

    def __init__(self) -> None:
        self._strategies: dict[str, Strategy] = {}
        self._initialized = False

    def _ensure_init(self) -> None:
        """전략 인스턴스 + DB 잔고 초기화."""
        if self._initialized:
            return

        for info in list_strategies():
            name = info["name"]
            self._strategies[name] = get_strategy(name)

            # DB에 잔고 없으면 생성
            with get_session() as session:
                bal = session.query(PaperBalance).filter_by(strategy=name).first()
                if not bal:
                    bal = PaperBalance(
                        strategy=name,
                        balance=INITIAL_BALANCE,
                        initial_balance=INITIAL_BALANCE,
                    )
                    session.add(bal)
                    session.commit()

        self._initialized = True

    async def tick(self, candles_map: dict[str, pd.DataFrame],
                   htf_map: dict[str, pd.DataFrame]) -> None:
        """매 틱마다 호출. 모든 전략 × 모든 심볼 평가."""
        self._ensure_init()

        for strategy_name, strategy in self._strategies.items():
            for symbol in SYMBOLS:
                candles = candles_map.get(symbol)
                htf = htf_map.get(symbol)
                if candles is None or candles.empty:
                    continue

                price = float(candles.iloc[-1]["close"])
                try:
                    self._process_tick(strategy_name, strategy, symbol,
                                       candles, htf, price)
                except Exception:
                    logger.exception("paper.tick_error",
                                     strategy=strategy_name, symbol=symbol)

    def _process_tick(self, name: str, strategy: Strategy, symbol: str,
                      candles: pd.DataFrame, htf: pd.DataFrame | None,
                      price: float) -> None:
        """단일 전략×심볼 틱 처리."""
        with get_session() as session:
            pos = session.query(PaperPosition).filter_by(
                strategy=name, symbol=symbol).first()

            # ATR 계산
            atr = 0.0
            if len(candles) > 14:
                atr = ta.volatility.AverageTrueRange(
                    candles["high"], candles["low"], candles["close"], window=14
                ).average_true_range().iloc[-1]
                if pd.isna(atr):
                    atr = 0.0

            # ── 포지션 있으면 SL/TP 체크 ──
            if pos:
                closed = self._check_exit(session, name, pos, price)
                if closed:
                    return

            # ── 포지션 없으면 진입 평가 ──
            if not pos:
                signal = strategy.evaluate(symbol, candles, htf)
                if signal.type in (SignalType.BUY, SignalType.SELL):
                    self._open(session, name, strategy, symbol, signal, price, atr)

    def _check_exit(self, session, name: str, pos: PaperPosition,
                    price: float) -> bool:
        """SL/TP 도달 체크. 청산 시 True."""
        if not pos.sl_price or not pos.tp_price:
            return False

        hit_sl = (pos.side == "LONG" and price <= pos.sl_price) or \
                 (pos.side == "SHORT" and price >= pos.sl_price)
        hit_tp = (pos.side == "LONG" and price >= pos.tp_price) or \
                 (pos.side == "SHORT" and price <= pos.tp_price)

        if not hit_sl and not hit_tp:
            return False

        target_price = pos.sl_price if hit_sl else pos.tp_price
        reason = "SL" if hit_sl else "TP"

        # 슬리피지 반영 청산가
        if pos.side == "LONG":
            exit_price = target_price * (1 - SLIPPAGE_RATE)  # 롱 청산은 살짝 싸게
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            exit_price = target_price * (1 + SLIPPAGE_RATE)  # 숏 청산은 살짝 비싸게
            pnl = (pos.entry_price - exit_price) * pos.quantity

        # 수수료: 진입(편도) + 청산(편도) + 슬리피지 양방향
        entry_fee = pos.entry_price * pos.quantity * (FEE_RATE + SLIPPAGE_RATE)
        exit_fee = exit_price * pos.quantity * (FEE_RATE + SLIPPAGE_RATE)
        fee = entry_fee + exit_fee
        net_pnl = pnl - fee

        # 거래 기록
        trade = PaperTrade(
            strategy=name, symbol=pos.symbol, side=pos.side,
            entry_price=pos.entry_price, exit_price=exit_price,
            quantity=pos.quantity, pnl=round(pnl, 4),
            fee=round(fee, 4), net_pnl=round(net_pnl, 4),
            sl_price=pos.sl_price, tp_price=pos.tp_price,
            reason=reason, opened_at=pos.opened_at, closed_at=now_kst(),
        )
        session.add(trade)

        # 잔고 업데이트
        bal = session.query(PaperBalance).filter_by(strategy=name).first()
        if bal:
            bal.balance += net_pnl
            bal.total_trades += 1
            if net_pnl >= 0:
                bal.wins += 1
            else:
                bal.losses += 1

        # 포지션 삭제
        session.delete(pos)
        session.commit()

        logger.info("paper.close", strategy=name, symbol=pos.symbol,
                     side=pos.side, reason=reason,
                     pnl=round(net_pnl, 4))
        return True

    def _open(self, session, name: str, strategy: Strategy, symbol: str,
              signal, price: float, atr: float) -> None:
        """가상 포지션 진입."""
        bal = session.query(PaperBalance).filter_by(strategy=name).first()
        if not bal or bal.balance < 10:
            return

        # 이미 이 전략×심볼에 포지션 있으면 스킵
        existing = session.query(PaperPosition).filter_by(
            strategy=name, symbol=symbol).first()
        if existing:
            return

        invest = bal.balance * POSITION_PCT
        qty = (invest * LEVERAGE) / price
        if qty <= 0 or invest < 5:
            return

        side = "LONG" if signal.type == SignalType.BUY else "SHORT"

        # 슬리피지 반영 체결가
        if side == "LONG":
            price = price * (1 + SLIPPAGE_RATE)  # 롱 진입은 살짝 비싸게
        else:
            price = price * (1 - SLIPPAGE_RATE)  # 숏 진입은 살짝 싸게

        # SL/TP 계산
        sl_mult = getattr(strategy, "SL_ATR_MULT", 6.0)
        tp_mult = getattr(strategy, "TP_ATR_MULT", 10.0)

        if atr > 0:
            sl_dist = atr * sl_mult
            tp_dist = atr * tp_mult
        else:
            sl_dist = price * 0.005
            tp_dist = price * 0.01

        if side == "LONG":
            sl_price = price - sl_dist
            tp_price = price + tp_dist
        else:
            sl_price = price + sl_dist
            tp_price = price - tp_dist

        pos = PaperPosition(
            strategy=name, symbol=symbol, side=side,
            entry_price=price, quantity=round(qty, 6),
            entry_atr=round(atr, 4) if atr > 0 else None,
            sl_price=round(sl_price, 2), tp_price=round(tp_price, 2),
        )
        session.add(pos)
        session.commit()

        logger.info("paper.open", strategy=name, symbol=symbol,
                     side=side, price=price, qty=round(qty, 6))
