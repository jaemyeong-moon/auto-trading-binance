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
        self._paper_task: asyncio.Task | None = None  # 독립 가상매매 태스크
        self._paper_trader = None  # lazy init
        self._ai_agent = None  # lazy init
        self._ai_agent_enabled = bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        self._settings_hash = db.get_settings_hash()
        self._current_strategy_name: dict[str, str] = {}
        self._restart_status: dict[str, str] = {}  # symbol -> "restarting" | ""

    async def start_symbol(self, symbol: str) -> None:
        if symbol in self._tasks and not self._tasks[symbol].done():
            return

        strategy_name = db.get_setting("strategy")
        strategy = get_strategy(strategy_name)
        self.strategies[symbol] = strategy
        self._current_strategy_name[symbol] = strategy_name

        # 전략이 자체 레버리지를 가지면 사용 (v6+)
        from src.strategies import aggressive_scalper as _as
        leverage = getattr(_as, "LEVERAGE", db.get_setting_int("leverage"))
        await self.client.set_leverage(symbol, leverage)

        db.set_bot_running(symbol, True)
        task = asyncio.create_task(self._symbol_loop(symbol))
        self._tasks[symbol] = task
        logger.info("engine.start", symbol=symbol, strategy=strategy.name,
                     mode=strategy.mode.value, leverage=leverage)

        # 가상매매 독립 루프 — 최초 1회만 시작
        if self._paper_task is None or self._paper_task.done():
            self._paper_task = asyncio.create_task(self._paper_loop())
            logger.info("engine.paper_loop_started")

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
        # 가상매매 루프도 종료
        if self._paper_task and not self._paper_task.done():
            self._paper_task.cancel()
            try:
                await self._paper_task
            except asyncio.CancelledError:
                pass

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

                # ── 설정 핫리로드 + 전략 자동 재시작 ──
                if tick_count % 3 == 0:
                    strategy = await self._check_hot_reload(symbol, strategy)

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
            logger.debug("engine.tick_done", symbol=symbol, tick=tick_count,
                         sleep=tick_interval)
            await asyncio.sleep(tick_interval)

    # ─── 독립 가상매매 루프 ──────────────────────────────────

    async def _paper_loop(self) -> None:
        """가상매매 독립 루프 — 실거래 엔진 상태와 무관하게 계속 실행."""
        from src.core.paper_trader import PaperTrader
        paper_symbols = ["BTCUSDT", "ETHUSDT"]
        tick_interval = 75  # 75초 (기존 tick 15초 × 5틱 = 75초 주기 유지)

        if self._paper_trader is None:
            self._paper_trader = PaperTrader()

        while True:
            try:
                candles_map = {}
                htf_map = {}
                for symbol in paper_symbols:
                    candles_15m = await self.client.get_candles(
                        symbol, interval="15m", limit=200)
                    htf = await self.client.get_candles(
                        symbol, interval="1h", limit=100)
                    if not candles_15m.empty:
                        candles_map[symbol] = candles_15m
                    if not htf.empty:
                        htf_map[symbol] = htf

                if candles_map:
                    await self._paper_trader.tick(candles_map, htf_map)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("paper.loop_error")
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
            candles = await self.client.get_candles(symbol, interval="15m", limit=300)
            htf = await self.client.get_candles(symbol, interval="1h", limit=100)
            if len(candles) < 200:
                return
            result = run_optimizer(candles, htf)
            if result:
                strategy = self.strategies.get(symbol)
                if strategy and hasattr(strategy, "SL_ATR_MULT"):
                    mode = getattr(strategy, "mode", None)
                    if mode == ExecutionMode.ALWAYS_FLIP:
                        # v1: 1:1 대칭 비율 (SL 편중 해소)
                        sl = max(result["sl_mult"], 2.0)
                        tp = max(result["tp_mult"], 2.0)
                    else:
                        # v2+: 넓은 SL로 노이즈 생존, 넓은 TP로 수수료 커버
                        sl = max(result["sl_mult"], 5.0)
                        tp = max(result["tp_mult"], 8.0)
                    trail_act = max(result["trail_act_mult"], 8.0)
                    trail_dist = max(result["trail_dist_mult"], 2.0)

                    strategy.SL_ATR_MULT = sl
                    strategy.TP_ATR_MULT = tp
                    strategy.TRAILING_ATR_MULT = trail_act
                    strategy.TRAILING_DIST_ATR = trail_dist
                    if hasattr(strategy, "PARTIAL_TP_ATR_MULT"):
                        strategy.PARTIAL_TP_ATR_MULT = round(tp * 0.6, 2)
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

    async def _check_hot_reload(self, symbol: str, strategy: Strategy) -> Strategy:
        """설정 변경 감지 → 핫리로드. 전략 변경 시 graceful restart."""
        try:
            new_hash = db.get_settings_hash()
            if new_hash == self._settings_hash:
                return strategy

            self._settings_hash = new_hash
            logger.info("engine.settings_reloaded", symbol=symbol)

            # 전략 변경 감지
            new_strategy_name = db.get_setting("strategy")
            old_strategy_name = self._current_strategy_name.get(symbol, "")

            if new_strategy_name != old_strategy_name:
                self._restart_status[symbol] = "restarting"
                logger.info("engine.strategy_switch",
                            symbol=symbol, old=old_strategy_name,
                            new=new_strategy_name)

                new_strategy = get_strategy(new_strategy_name)
                self.strategies[symbol] = new_strategy
                self._current_strategy_name[symbol] = new_strategy_name

                # 레버리지 재설정
                from src.strategies import aggressive_scalper as _as
                leverage = getattr(_as, "LEVERAGE", db.get_setting_int("leverage"))
                await self.client.set_leverage(symbol, leverage)

                self._restart_status[symbol] = ""
                logger.info("engine.strategy_switched",
                            symbol=symbol, strategy=new_strategy_name)
                return new_strategy

            # 레버리지만 변경된 경우
            from src.strategies import aggressive_scalper as _as
            leverage = getattr(_as, "LEVERAGE", db.get_setting_int("leverage"))
            await self.client.set_leverage(symbol, leverage)

        except Exception:
            logger.exception("engine.hot_reload_failed", symbol=symbol)
            self._restart_status[symbol] = ""
        return strategy

    def _log_signal(self, symbol: str, signal, source: str = "real") -> None:
        """전략 평가 결과를 DB에 기록."""
        try:
            db.log_signal(
                symbol=symbol,
                strategy=signal.source or db.get_setting("strategy"),
                signal_type=signal.type.value,
                confidence=signal.confidence,
                metadata=signal.metadata,
                source=source,
            )
        except Exception:
            pass  # 로깅 실패가 거래에 영향 주면 안 됨

    def get_restart_status(self) -> dict[str, str]:
        """각 심볼의 재시작 상태 반환."""
        return dict(self._restart_status)

    async def _fetch_candles(self, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """15분봉(메인) + 1시간봉(HTF) 매 틱."""
        candles_15m = await self.client.get_candles(symbol, interval="15m", limit=200)
        htf_candles = await self.client.get_candles(symbol, interval="1h", limit=250)
        return candles_15m, htf_candles

    def _record_trail(self, symbol: str, pos: dict, price: float) -> None:
        """실거래 포지션 궤적 기록."""
        db_pos = db.get_position(symbol)
        if not db_pos or not db_pos.sl_price or not db_pos.tp_price:
            return
        try:
            db.record_trail(
                trade_type="real", symbol=symbol, side=pos["side"],
                entry_price=pos["entry_price"],
                sl_price=db_pos.sl_price, tp_price=db_pos.tp_price,
                price=price,
            )
        except Exception:
            logger.debug("engine.trail_record_failed", symbol=symbol)

    async def _tick_always_flip(self, symbol: str, strategy: Strategy) -> None:
        candles, htf = await self._fetch_candles(symbol)
        if candles.empty:
            return

        price = float(candles.iloc[-1]["close"])
        pos = await self.client.get_position(symbol)

        # ATR 기반 (v1.1) vs 고정 % (v1) 자동 선택
        state = getattr(strategy, "state", None)
        entry_atr = getattr(state, "entry_atr", 0) if state else 0
        if entry_atr > 0 and hasattr(strategy, "SL_ATR_MULT") and price > 0:
            sl_pct = (entry_atr * strategy.SL_ATR_MULT) / price
            tp_pct = (entry_atr * strategy.TP_ATR_MULT) / price
        else:
            tp_pct = db.get_setting_float("tp_pct")
            sl_pct = db.get_setting_float("sl_pct")

        # 궤적 기록
        if pos:
            self._record_trail(symbol, pos, price)

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
        self._log_signal(symbol, signal)

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

        # ── 궤적 기록 ──
        if pos:
            self._record_trail(symbol, pos, price)

        # ── 포지션 있을 때: 익절/손절/트레일링 관리 ──
        if pos:
            closed = await self._manage_exit_v2(symbol, strategy, pos, price, candles)
            if closed:
                return

        # ── 전략 평가 (1분봉 + 15분봉) ──
        signal = strategy.evaluate(symbol, candles, htf)
        self._log_signal(symbol, signal)

        if signal.type == SignalType.CLOSE and pos:
            close_reason = signal.metadata.get("reason", "signal")
            pnl = await self._close_current(symbol, reason=close_reason)
            strategy.record_result(pnl)
            logger.info("engine.close_signal", symbol=symbol, reason=close_reason)
            return

        # ── DCA 물타기: 포지션 있는데 같은 방향 신호 = 추가 매수 ──
        if signal.type in (SignalType.BUY, SignalType.SELL) and pos:
            entry_type = signal.metadata.get("entry_type", "")
            if entry_type in ("dca1", "dca2"):
                size_pct = signal.metadata.get("size_pct", 0.10)
                await self._add_dca_position(symbol, signal, price, size_pct)
                return

        if signal.type in (SignalType.BUY, SignalType.SELL) and not pos:
            # 신호 confidence가 낮으면 진입 차단
            if signal.confidence < 0.6:
                logger.info("engine.skip_entry", symbol=symbol,
                             reason="low_confidence",
                             confidence=f"{signal.confidence:.2f}")
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
        if state and not getattr(state, "partial_tp_taken", False) and change >= partial_tp:
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

    async def _add_dca_position(self, symbol: str, signal, price: float,
                                size_pct: float) -> None:
        """DCA 물타기 — 기존 포지션에 추가 매수/매도."""
        balance = await self.client.get_balance()
        from src.strategies import aggressive_scalper as _as
        leverage = getattr(_as, "LEVERAGE", 5)
        invest = balance * size_pct
        quantity = (invest * leverage) / price
        quantity = self._round_qty(symbol, quantity)

        # 최소 notional 충족
        min_notional = 100
        if quantity * price < min_notional:
            min_qty = self._round_qty(symbol, min_notional / price + 10 ** -self._qty_precision(symbol))
            if (min_qty * price) / leverage <= balance * 0.9:
                quantity = min_qty
                invest = (quantity * price) / leverage
            else:
                return

        if quantity <= 0 or invest < 5:
            return

        direction = signal.metadata.get("direction", "LONG")
        try:
            if direction == "LONG":
                await self.client.open_long(symbol, quantity)
            else:
                await self.client.open_short(symbol, quantity)
        except Exception:
            logger.exception("engine.dca_failed", symbol=symbol)
            return

        # 전략 state 업데이트
        strategy = self.strategies.get(symbol)
        state = getattr(strategy, "state", None)
        if state and hasattr(state, "add_dca"):
            state.add_dca(price, quantity)

        dca_level = signal.metadata.get("dca_level", 0)
        avg_entry = state.avg_entry if state else price
        logger.info("engine.dca", symbol=symbol, direction=direction,
                     dca_level=dca_level, qty=quantity,
                     avg_entry=f"${avg_entry:,.2f}",
                     invest=f"${invest:.2f}")

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

        # 전략이 자체 포지션 설정을 가지면 사용
        strategy = self.strategies.get(symbol)
        from src.strategies import aggressive_scalper as _as
        if hasattr(_as, "LEVERAGE"):
            leverage = _as.LEVERAGE
            # 적응형: 전략 state에서 동적 size_pct 읽기
            state = getattr(strategy, "state", None)
            params = getattr(state, "params", None)
            size_pct = getattr(params, "grid_size_pct", 0.06) if params else 0.06
        else:
            size_pct = self._dynamic_size_pct(balance)
            leverage = db.get_setting_int("leverage")

        # 안전장치: 가용 잔고의 90%를 넘지 않도록 (증거금 여유)
        max_invest = balance * 0.9
        invest = min(balance * size_pct, max_invest)
        quantity = (invest * leverage) / price
        quantity = self._round_qty(symbol, quantity)

        # 바이낸스 선물 최소 주문금액 (notional) 충족 확인
        min_notional = 100  # USDT
        notional = quantity * price
        if notional < min_notional:
            # 최소 notional을 충족하도록 quantity 보정
            min_qty = self._round_qty(symbol, min_notional / price + 10 ** -self._qty_precision(symbol))
            required_margin = (min_qty * price) / leverage
            if required_margin <= balance * 0.9:
                quantity = min_qty
                invest = (quantity * price) / leverage
                logger.info("engine.notional_adjusted",
                            symbol=symbol, notional=f"${quantity * price:.0f}",
                            margin=f"${invest:.2f}")
            else:
                logger.warning("engine.insufficient_for_min_notional",
                               symbol=symbol, balance=balance,
                               required_margin=f"${required_margin:.2f}")
                return

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
            elif "-4164" in err_msg:
                logger.warning("engine.min_notional_error",
                               symbol=symbol, notional=f"${quantity * price:.0f}",
                               balance=balance)
            else:
                logger.exception("engine.order_failed",
                                 symbol=symbol, direction=direction)
            return

        # SL/TP 가격 계산 + DB 저장 + 거래소 주문
        sl_price, tp_price = self._calc_sl_tp(direction, price, strategy)
        db.open_position(
            symbol=symbol, side=direction, entry_price=price,
            quantity=quantity, strategy=db.get_setting("strategy"),
            sl_price=sl_price, tp_price=tp_price,
        )
        logger.info("engine.position_opened", symbol=symbol, direction=direction,
                     qty=quantity, invest=f"${invest:.2f}",
                     balance=f"${balance:.2f}", size_pct=f"{size_pct:.0%}")

        # 거래소에 SL/TP 주문 (봇 꺼져도 보호)
        await self._place_exchange_sl_tp(symbol, direction, price, strategy)

        if db.get_setting("webhook_on_open") == "true":
            await webhook.notify_open(symbol, direction, price, quantity, invest)

    async def _close_current(self, symbol: str, reason: str = "") -> float:
        # 거래소 SL/TP 주문 취소 (봇이 직접 청산하므로)
        await self.client.cancel_open_orders(symbol)

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
        trade = db.close_position(symbol, exit_price=current_price, fee=fee, reason=reason)
        pnl = trade.pnl if trade else 0.0
        net_pnl = trade.net_pnl if trade else 0.0

        # 궤적을 거래에 연결
        if trade:
            db.link_trails_to_trade("real", trade.id, symbol, entry_price)

        if db.get_setting("webhook_on_close") == "true":
            # 계좌 잔고 및 수익률 정보 포함
            summary = await self.client.get_account_summary()
            balance = summary["balance"]
            await webhook.notify_close(
                symbol, side, entry_price, current_price, pnl, reason,
                balance=balance, fee=fee, net_pnl=net_pnl)

        return pnl

    def _calc_sl_tp(
        self, direction: str, entry_price: float, strategy: Strategy,
    ) -> tuple[float, float]:
        """SL/TP 가격 계산. (sl_price, tp_price) 반환."""
        state = getattr(strategy, "state", None)

        if state and hasattr(state, "sl_price") and getattr(state, "sl_price", 0) > 0:
            return state.sl_price, state.tp_price

        entry_atr = getattr(state, "entry_atr", 0) if state else 0
        sl_mult = getattr(strategy, "SL_ATR_MULT", 6.0)
        tp_mult = getattr(strategy, "TP_ATR_MULT", 10.0)

        if entry_atr > 0 and sl_mult < 50:
            sl_dist = entry_atr * sl_mult
            tp_dist = entry_atr * tp_mult
        else:
            sl_pct = db.get_setting_float("sl_pct") or 0.005
            tp_pct = db.get_setting_float("tp_pct") or 0.01
            sl_dist = entry_price * sl_pct
            tp_dist = entry_price * tp_pct

        if direction == "LONG":
            return entry_price - sl_dist, entry_price + tp_dist
        return entry_price + sl_dist, entry_price - tp_dist

    async def _place_exchange_sl_tp(
        self, symbol: str, direction: str, entry_price: float, strategy: Strategy,
    ) -> None:
        """진입 후 거래소에 SL/TP 주문. _calc_sl_tp 재사용."""
        try:
            sl_price, tp_price = self._calc_sl_tp(direction, entry_price, strategy)
            pos = await self.client.get_position(symbol)
            qty = pos["quantity"] if pos else 0

            if qty > 0 and sl_price > 0 and tp_price > 0:
                await self.client.place_sl_tp_orders(
                    symbol=symbol, side=direction, quantity=qty,
                    sl_price=sl_price, tp_price=tp_price,
                )
                logger.info("engine.sl_tp_placed", symbol=symbol,
                            sl=f"${sl_price:,.2f}", tp=f"${tp_price:,.2f}")
        except Exception:
            logger.exception("engine.sl_tp_place_failed", symbol=symbol)

    def _qty_precision(self, symbol: str) -> int:
        precision = {
            "BTCUSDT": 3, "ETHUSDT": 3, "BNBUSDT": 2,
            "SOLUSDT": 1, "XRPUSDT": 0,
        }
        return precision.get(symbol, 3)

    def _round_qty(self, symbol: str, quantity: float) -> float:
        return round(quantity, self._qty_precision(symbol))

    def _min_qty(self, symbol: str) -> float:
        """심볼별 최소 주문 수량 (바이낸스 선물)."""
        minimums = {
            "BTCUSDT": 0.001, "ETHUSDT": 0.001, "BNBUSDT": 0.01,
            "SOLUSDT": 0.1, "XRPUSDT": 1,
        }
        return minimums.get(symbol, 0.001)
