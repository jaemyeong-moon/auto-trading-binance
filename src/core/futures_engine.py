"""Futures trading engine — v1(always flip) + v2(signal only) 듀얼 모드."""

import asyncio
import os

import pandas as pd
import structlog

from src.core import database as db
from src.core.auto_optimizer import run_and_save as run_optimizer
from src.core.models import SignalType
from src.exchange.futures_client import FuturesClient
from src.notifications import webhook
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.registry import get_strategy

logger = structlog.get_logger()

# AI Agent 실행 주기 (틱 수 기준, 기본 tick=30초 × 120 = 약 1시간)
AI_AGENT_INTERVAL_TICKS = 120


class FuturesEngine:
    """선물 거래 엔진. 전략의 mode에 따라 실행 방식이 달라짐."""

    def __init__(self, client: FuturesClient) -> None:
        self.client = client
        self.strategies: dict[str, Strategy] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._ai_agent = None  # lazy init
        self._ai_agent_enabled = bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )

    async def start_symbol(self, symbol: str) -> None:
        if symbol in self._tasks and not self._tasks[symbol].done():
            return

        leverage = db.get_setting_int("leverage")
        await self.client.set_leverage(symbol, leverage)

        strategy_name = db.get_setting("strategy")
        strategy = get_strategy(strategy_name)
        self.strategies[symbol] = strategy

        db.set_bot_running(symbol, True)
        task = asyncio.create_task(self._symbol_loop(symbol))
        self._tasks[symbol] = task
        logger.info("engine.start", symbol=symbol, strategy=strategy.name,
                     mode=strategy.mode.value, leverage=leverage)

    async def stop_symbol(self, symbol: str) -> None:
        db.set_bot_running(symbol, False)
        task = self._tasks.pop(symbol, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._close_current(symbol, reason="bot_stopped")
        logger.info("engine.stop", symbol=symbol)

    async def stop_all(self) -> None:
        for symbol in list(self._tasks.keys()):
            await self.stop_symbol(symbol)

    # ─── Main loop ─────────────────────────────────────────

    async def _symbol_loop(self, symbol: str) -> None:
        strategy = self.strategies[symbol]
        tick_count = 0
        while db.is_bot_running(symbol):
            try:
                tick_count += 1

                # 10분마다 시간 재동기화
                if tick_count % 20 == 0:
                    await self.client.sync_time()

                # 10분마다 TP/SL 자동 최적화 (1분봉 500개로)
                if tick_count % 20 == 1:
                    await self._run_auto_optimize(symbol)

                # AI Agent: 주기적으로 전략 성과 분석 및 신규 전략 생성
                if self._ai_agent_enabled and tick_count % AI_AGENT_INTERVAL_TICKS == 1:
                    await self._run_ai_agent(symbol)

                if strategy.mode == ExecutionMode.ALWAYS_FLIP:
                    await self._tick_always_flip(symbol, strategy)
                else:
                    await self._tick_signal_only(symbol, strategy)
            except asyncio.CancelledError:
                break
            except Exception as e:
                if "-1021" in str(e):
                    # Timestamp 에러 → 즉시 재동기화 후 계속
                    logger.warning("engine.timestamp_resync", symbol=symbol)
                    await self.client.sync_time()
                else:
                    logger.exception("engine.tick_error", symbol=symbol)
            tick_interval = db.get_setting_int("tick_interval")
            await asyncio.sleep(tick_interval)

    async def _run_ai_agent(self, symbol: str) -> None:
        """AI Agent 실행: 성과 분석 → 필요 시 신규 전략 생성 → 핫 스왑."""
        try:
            from src.core.strategy_agent import AIStrategyAgent

            if self._ai_agent is None:
                self._ai_agent = AIStrategyAgent()

            # 백테스트 검증용 캔들
            candles = await self.client.get_candles(symbol, interval="1m", limit=500)

            # 시장 요약 (현재 가격, 변동성 등)
            market_summary = ""
            if len(candles) > 50:
                close = candles["close"]
                price = close.iloc[-1]
                change_1h = ((price - close.iloc[-60]) / close.iloc[-60] * 100
                             if len(close) >= 60 else 0)
                import ta as ta_lib
                atr = ta_lib.volatility.AverageTrueRange(
                    candles["high"], candles["low"], close, window=14
                ).average_true_range().iloc[-1]
                market_summary = (
                    f"심볼: {symbol}\n"
                    f"현재가: {price:.2f}\n"
                    f"1시간 변동: {change_1h:+.2f}%\n"
                    f"ATR(14): {atr:.2f} ({atr/price*100:.3f}%)\n"
                )

            report = await self._ai_agent.run(
                candles_for_backtest=candles,
                market_data_summary=market_summary,
            )

            # 보고서 로깅
            report_text = self._ai_agent.format_report(report)
            logger.info("agent.report", report=report_text)

            # 전략이 교체되었으면 핫 스왑
            if report.action_taken == "strategy_switched" and report.new_strategy_name:
                new_strategy = get_strategy(report.new_strategy_name)
                self.strategies[symbol] = new_strategy
                logger.info("agent.hot_swap",
                             symbol=symbol,
                             old=report.current_strategy,
                             new=report.new_strategy_name)

                # 웹훅 알림
                if db.get_setting("webhook_url"):
                    await webhook.send_raw(
                        f"🤖 AI Agent: 전략 교체\n"
                        f"{report.current_strategy} → {report.new_strategy_name}\n"
                        f"사유: {report.performance.reason}"
                    )

        except ImportError as e:
            logger.warning("agent.llm_sdk_not_installed", error=str(e),
                           msg="pip install anthropic/openai/google-genai")
            self._ai_agent_enabled = False
        except Exception:
            logger.exception("agent.run_failed", symbol=symbol)

    async def _run_auto_optimize(self, symbol: str) -> None:
        """최근 데이터 기반 TP/SL 배수 자동 최적화."""
        try:
            candles = await self.client.get_candles(symbol, interval="1m", limit=500)
            htf = await self.client.get_candles(symbol, interval="15m", limit=100)
            if len(candles) < 200:
                return
            result = run_optimizer(candles, htf)
            if result:
                strategy = self.strategies.get(symbol)
                if strategy and hasattr(strategy, "SL_ATR_MULT"):
                    # 최소 하한 적용 — 수수료+스프레드 커버 필수
                    sl = max(result["sl_mult"], 1.0)
                    tp = max(result["tp_mult"], 2.0)
                    trail_act = max(result["trail_act_mult"], 2.0)
                    trail_dist = max(result["trail_dist_mult"], 0.5)

                    strategy.SL_ATR_MULT = sl
                    strategy.TP_ATR_MULT = tp
                    strategy.TRAILING_ATR_MULT = trail_act
                    strategy.TRAILING_DIST_ATR = trail_dist
                    if hasattr(strategy, "PARTIAL_TP_ATR_MULT"):
                        strategy.PARTIAL_TP_ATR_MULT = round(tp * 0.5, 2)
                    # 심볼별 optimizer 점수 저장 (진입 차단 판단용)
                    db.set_setting(f"opt_score_{symbol}", str(result["score"]))
                    logger.info("engine.auto_optimized", symbol=symbol,
                                 sl=sl, tp=tp,
                                 partial_tp=round(result["tp_mult"] * 0.4, 2),
                                 score=result["score"])
        except Exception:
            logger.exception("engine.auto_optimize_failed", symbol=symbol)

    # ═══════════════════════════════════════════════════════
    #  Mode 1: Always Flip (v1)
    # ═══════════════════════════════════════════════════════

    async def _fetch_candles(self, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """1분봉 300개 + 15분봉 100개 동시 조회."""
        candles_1m = await self.client.get_candles(symbol, interval="1m", limit=300)
        htf_candles = await self.client.get_candles(symbol, interval="15m", limit=100)
        return candles_1m, htf_candles

    async def _tick_always_flip(self, symbol: str, strategy: Strategy) -> None:
        candles, htf = await self._fetch_candles(symbol)
        if candles.empty:
            return

        price = float(candles.iloc[-1]["close"])
        pos = await self.client.get_position(symbol)
        tp_pct = db.get_setting_float("tp_pct")
        sl_pct = db.get_setting_float("sl_pct")

        # TP/SL
        if pos:
            change = self._calc_change(pos, price)
            if change >= tp_pct:
                logger.info("engine.tp", symbol=symbol, change=f"{change:.3%}")
                pnl = await self._close_current(symbol, reason="take_profit")
                strategy.record_result(pnl)
                signal = strategy.evaluate(symbol, candles, htf)
                if signal.type in (SignalType.BUY, SignalType.SELL):
                    d = "LONG" if signal.type == SignalType.BUY else "SHORT"
                    await self._open_position(symbol, d, price)
                return

            if change <= -sl_pct:
                logger.info("engine.sl", symbol=symbol, change=f"{change:.3%}")
                pnl = await self._close_current(symbol, reason="stop_loss")
                strategy.record_result(pnl)
                signal = strategy.evaluate(symbol, candles, htf)
                if signal.type in (SignalType.BUY, SignalType.SELL):
                    d = "LONG" if signal.type == SignalType.BUY else "SHORT"
                    await self._open_position(symbol, d, price)
                return

        signal = strategy.evaluate(symbol, candles, htf)
        if signal.type == SignalType.HOLD:
            return

        if pos:
            pnl = await self._close_current(symbol, reason="flip")
            strategy.record_result(pnl)

        direction = "LONG" if signal.type == SignalType.BUY else "SHORT"
        await self._open_position(symbol, direction, price)
        logger.info("engine.flip", symbol=symbol, direction=direction,
                     confidence=f"{signal.confidence:.2f}")

    # ═══════════════════════════════════════════════════════
    #  Mode 2: Signal Only (v2)
    # ═══════════════════════════════════════════════════════

    async def _tick_signal_only(self, symbol: str, strategy: Strategy) -> None:
        candles, htf = await self._fetch_candles(symbol)
        if candles.empty:
            return

        price = float(candles.iloc[-1]["close"])
        pos = await self.client.get_position(symbol)

        # ── 포지션 있을 때: 익절/손절/트레일링 관리 ──
        if pos:
            closed = await self._manage_exit_v2(symbol, strategy, pos, price, candles)
            if closed:
                return

        # ── 전략 평가 (1분봉 + 15분봉) ──
        signal = strategy.evaluate(symbol, candles, htf)

        if signal.type == SignalType.CLOSE and pos:
            close_reason = signal.metadata.get("reason", "signal")
            pnl = await self._close_current(symbol, reason=close_reason)
            strategy.record_result(pnl)
            logger.info("engine.close_signal", symbol=symbol, reason=close_reason)
            return

        if signal.type in (SignalType.BUY, SignalType.SELL) and not pos:
            # optimizer 점수가 0이면 신규 진입 차단 (돈 태우는 심볼 보호)
            opt_score = db.get_setting_float(f"opt_score_{symbol}")
            if opt_score <= 0:
                # DB에 심볼별 점수 없으면 글로벌 점수 확인
                opt_score = db.get_setting_float("auto_opt_score")
            if opt_score <= 0:
                logger.info("engine.skip_entry", symbol=symbol,
                             reason="optimizer_score_zero")
                return

            direction = "LONG" if signal.type == SignalType.BUY else "SHORT"
            await self._open_position(symbol, direction, price)
            logger.info("engine.entry", symbol=symbol, direction=direction,
                         score=signal.metadata.get("score"),
                         market=signal.metadata.get("market"),
                         confidence=f"{signal.confidence:.2f}")

    async def _manage_exit_v2(self, symbol: str, strategy: Strategy,
                               pos: dict, price: float, candles) -> bool:
        """포지션 관리: 부분익절 + 트레일링 + 손절. True면 청산됨.
        v3의 ATR 기반 동적 TP/SL도 지원."""
        entry = pos["entry_price"]
        side = pos["side"]
        qty = pos["quantity"]
        change = self._calc_change(pos, price)

        state = getattr(strategy, "state", None)
        entry_atr = getattr(state, "entry_atr", 0) if state else 0

        # ATR 기반 (v3) vs 고정 % (v2) 자동 선택
        if entry_atr > 0 and entry > 0:
            # v3: ATR 절대값 기반
            sl_mult = getattr(strategy, "SL_ATR_MULT", 1.0)
            tp_mult = getattr(strategy, "TP_ATR_MULT", 2.0)
            partial_mult = getattr(strategy, "PARTIAL_TP_ATR_MULT", 1.0)
            trail_act_mult = getattr(strategy, "TRAILING_ATR_MULT", 1.5)
            trail_dist_mult = getattr(strategy, "TRAILING_DIST_ATR", 0.5)

            sl = (entry_atr * sl_mult) / entry          # ATR → % 변환
            full_tp = (entry_atr * tp_mult) / entry
            partial_tp = (entry_atr * partial_mult) / entry
            trail_activate = (entry_atr * trail_act_mult) / entry
            trail_dist = (entry_atr * trail_dist_mult) / entry
        else:
            # v2: 고정 %
            partial_tp = getattr(strategy, "PARTIAL_TP_PCT", 0.005)
            full_tp = getattr(strategy, "FULL_TP_PCT", 0.012)
            trail_activate = getattr(strategy, "TRAILING_ACTIVATE_PCT", 0.008)
            trail_dist = getattr(strategy, "TRAILING_DISTANCE_PCT", 0.003)
            sl = getattr(strategy, "SL_PCT", db.get_setting_float("sl_pct"))

        # ── 손절 ──
        if change <= -sl:
            logger.info("engine.sl_v2", symbol=symbol, change=f"{change:.3%}")
            pnl = await self._close_current(symbol, reason="stop_loss")
            strategy.record_result(pnl)
            return True

        # ── 전량 익절 ──
        if change >= full_tp:
            logger.info("engine.full_tp", symbol=symbol, change=f"{change:.3%}")
            pnl = await self._close_current(symbol, reason="take_profit")
            strategy.record_result(pnl)
            return True

        # ── 부분 익절 (50%) ──
        if state and not state.partial_tp_taken and change >= partial_tp:
            half_qty = self._round_qty(symbol, qty * 0.5)
            # 최소 주문 수량 체크 (너무 작으면 거래소 거부)
            min_qty = self._min_qty(symbol)
            if half_qty >= min_qty and (qty - half_qty) >= min_qty:
                try:
                    if side == "LONG":
                        await self.client.close_long(symbol, half_qty)
                    else:
                        await self.client.close_short(symbol, half_qty)
                    state.partial_tp_taken = True
                    remaining_qty = self._round_qty(symbol, qty - half_qty)
                    db.update_position_quantity(symbol, remaining_qty)
                    logger.info("engine.partial_tp", symbol=symbol,
                                 closed_qty=half_qty, remaining=remaining_qty,
                                 change=f"{change:.3%}")
                    if db.get_setting("webhook_on_tp_sl") == "true":
                        await webhook.notify_partial_tp(symbol, price, half_qty, change)
                except Exception:
                    logger.exception("engine.partial_tp_failed", symbol=symbol)

        # ── 트레일링 스탑 활성화/업데이트 ──
        if state and change >= trail_activate:
            state.update_price(price)
            if side == "LONG":
                new_stop = state.highest_since_entry * (1 - trail_dist)
                if state.trailing_stop_price is None or new_stop > state.trailing_stop_price:
                    state.trailing_stop_price = new_stop
                if price <= state.trailing_stop_price:
                    logger.info("engine.trailing_stop", symbol=symbol)
                    pnl = await self._close_current(symbol, reason="trailing_stop")
                    strategy.record_result(pnl)
                    return True
            else:
                new_stop = state.lowest_since_entry * (1 + trail_dist)
                if state.trailing_stop_price is None or new_stop < state.trailing_stop_price:
                    state.trailing_stop_price = new_stop
                if price >= state.trailing_stop_price:
                    logger.info("engine.trailing_stop", symbol=symbol)
                    pnl = await self._close_current(symbol, reason="trailing_stop")
                    strategy.record_result(pnl)
                    return True

        return False

    # ─── Shared helpers ────────────────────────────────────

    def _calc_change(self, pos: dict, price: float) -> float:
        entry = pos["entry_price"]
        if pos["side"] == "LONG":
            return (price - entry) / entry
        return (entry - price) / entry

    def _dynamic_size_pct(self, balance: float) -> float:
        """잔고 기반 동적 투자 비율 — 1심볼 집중 운영 최적화.

        $150 미만: 25% (자본 보호하면서도 의미있는 포지션)
        $150~$300: 30% (기본)
        $300~$500: 35% (수익 확인 후 확대)
        $500~$1000: 40% (본격 스케일업)
        $1000+: 45% (최대)
        """
        if balance < 150:
            return 0.25
        elif balance < 300:
            return 0.30
        elif balance < 500:
            return 0.35
        elif balance < 1000:
            return 0.40
        else:
            return 0.45

    async def _open_position(self, symbol: str, direction: str, price: float) -> None:
        balance = await self.client.get_balance()
        size_pct = self._dynamic_size_pct(balance)
        leverage = db.get_setting_int("leverage")

        # 안전장치: 가용 잔고의 90%를 넘지 않도록 (증거금 여유)
        max_invest = balance * 0.9
        invest = min(balance * size_pct, max_invest)
        quantity = (invest * leverage) / price
        quantity = self._round_qty(symbol, quantity)

        if quantity <= 0 or invest < 5:
            logger.warning("engine.insufficient_balance",
                           symbol=symbol, balance=balance, invest=invest)
            return

        try:
            if direction == "LONG":
                await self.client.open_long(symbol, quantity)
            else:
                await self.client.open_short(symbol, quantity)
        except Exception as e:
            err_msg = str(e)
            if "-2019" in err_msg:
                logger.warning("engine.margin_insufficient",
                               symbol=symbol, balance=balance, invest=invest)
            else:
                logger.exception("engine.order_failed",
                                 symbol=symbol, direction=direction)
            return

        db.open_position(
            symbol=symbol, side=direction, entry_price=price,
            quantity=quantity, strategy=db.get_setting("strategy"),
        )
        logger.info("engine.position_opened", symbol=symbol, direction=direction,
                     qty=quantity, invest=f"${invest:.2f}",
                     balance=f"${balance:.2f}", size_pct=f"{size_pct:.0%}")

        if db.get_setting("webhook_on_open") == "true":
            await webhook.notify_open(symbol, direction, price, quantity, invest)

    async def _close_current(self, symbol: str, reason: str = "") -> float:
        pos = await self.client.get_position(symbol)
        if not pos:
            # 거래소에 포지션 없으면 DB 고아 레코드 정리
            db.delete_position(symbol)
            return 0.0
        entry_price = pos["entry_price"]
        side = pos["side"]
        try:
            if side == "LONG":
                await self.client.close_long(symbol, pos["quantity"])
            else:
                await self.client.close_short(symbol, pos["quantity"])
        except Exception:
            logger.exception("engine.close_failed", symbol=symbol)
            return 0.0
        current_price = await self.client.get_price(symbol)
        fee = await self.client.get_recent_fees(symbol)
        trade = db.close_position(symbol, exit_price=current_price, fee=fee)
        pnl = trade.pnl if trade else 0.0
        net_pnl = trade.net_pnl if trade else 0.0

        if db.get_setting("webhook_on_close") == "true":
            # 계좌 잔고 및 수익률 정보 포함
            summary = await self.client.get_account_summary()
            balance = summary["balance"]
            await webhook.notify_close(
                symbol, side, entry_price, current_price, pnl, reason,
                balance=balance, fee=fee, net_pnl=net_pnl)

        return pnl

    def _round_qty(self, symbol: str, quantity: float) -> float:
        precision = {
            "BTCUSDT": 3, "ETHUSDT": 3, "BNBUSDT": 2,
            "SOLUSDT": 1, "XRPUSDT": 0,
        }
        return round(quantity, precision.get(symbol, 3))

    def _min_qty(self, symbol: str) -> float:
        """심볼별 최소 주문 수량 (바이낸스 선물)."""
        minimums = {
            "BTCUSDT": 0.001, "ETHUSDT": 0.001, "BNBUSDT": 0.01,
            "SOLUSDT": 0.1, "XRPUSDT": 1,
        }
        return minimums.get(symbol, 0.001)
