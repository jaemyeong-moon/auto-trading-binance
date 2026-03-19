"""Core trading engine — orchestrates the trading loop per symbol."""

import asyncio
from datetime import datetime

import structlog

from src.core.config import settings
from src.core.models import Signal, SignalType
from src.core import database as db
from src.exchange.binance_client import BinanceClient
from src.strategies.base import Strategy

logger = structlog.get_logger()


class TradingEngine:
    """Runs trading loops for individual symbols. Writes all state to DB."""

    def __init__(self, strategy: Strategy, client: BinanceClient) -> None:
        self.strategy = strategy
        self.client = client
        self._tasks: dict[str, asyncio.Task] = {}

    async def start_symbol(self, symbol: str) -> None:
        """Start trading loop for a specific symbol."""
        if symbol in self._tasks and not self._tasks[symbol].done():
            return
        db.set_bot_running(symbol, True)
        task = asyncio.create_task(self._symbol_loop(symbol))
        self._tasks[symbol] = task
        logger.info("engine.start_symbol", symbol=symbol, strategy=self.strategy.name)

    async def stop_symbol(self, symbol: str) -> None:
        """Stop trading loop for a specific symbol."""
        db.set_bot_running(symbol, False)
        task = self._tasks.pop(symbol, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.info("engine.stop_symbol", symbol=symbol)

    async def stop_all(self) -> None:
        for symbol in list(self._tasks.keys()):
            await self.stop_symbol(symbol)

    def is_running(self, symbol: str) -> bool:
        task = self._tasks.get(symbol)
        return task is not None and not task.done()

    async def _symbol_loop(self, symbol: str) -> None:
        """Main loop for one symbol."""
        interval_sec = self._interval_seconds()
        while db.is_bot_running(symbol):
            try:
                await self._tick(symbol)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("engine.tick_error", symbol=symbol)
            await asyncio.sleep(interval_sec)

    async def _tick(self, symbol: str) -> None:
        """Single evaluation cycle for a symbol."""
        candles = await self.client.get_candles(
            symbol=symbol, interval=settings.trading.interval
        )
        if candles.empty:
            return

        current_price = float(candles.iloc[-1]["close"])
        signal = self.strategy.evaluate(symbol, candles)
        position = db.get_position(symbol)

        logger.info(
            "engine.tick",
            symbol=symbol,
            price=current_price,
            signal=signal.type.value,
            confidence=f"{signal.confidence:.2f}",
            has_position=position is not None,
        )

        if signal.type == SignalType.BUY and position is None:
            await self._open_position(symbol, current_price)
        elif signal.type == SignalType.SELL and position is not None:
            await self._close_position(symbol, current_price)
        elif position is not None:
            # Check stop-loss / take-profit
            change = (current_price - position.entry_price) / position.entry_price
            if change <= -settings.risk.stop_loss_pct:
                logger.info("engine.stop_loss", symbol=symbol, change=f"{change:.2%}")
                await self._close_position(symbol, current_price)
            elif change >= settings.risk.take_profit_pct:
                logger.info("engine.take_profit", symbol=symbol, change=f"{change:.2%}")
                await self._close_position(symbol, current_price)

    async def _open_position(self, symbol: str, price: float) -> None:
        """Open a new position."""
        balance = await self.client.get_balance("USDT")
        invest = balance * settings.trading.position_size_pct
        quantity = invest / price

        # Round quantity to reasonable precision
        quantity = round(quantity, 6)
        if quantity <= 0:
            logger.warning("engine.insufficient_balance", symbol=symbol, balance=balance)
            return

        try:
            await self.client.place_order(symbol=symbol, side="BUY", quantity=quantity)
        except Exception:
            logger.exception("engine.order_failed", symbol=symbol, side="BUY")
            return

        db.open_position(
            symbol=symbol, side="BUY", entry_price=price,
            quantity=quantity, strategy=self.strategy.name,
        )
        logger.info(
            "engine.position_opened",
            symbol=symbol, price=price, quantity=quantity, invest=invest,
        )

    async def _close_position(self, symbol: str, price: float) -> None:
        """Close an existing position."""
        position = db.get_position(symbol)
        if not position:
            return

        try:
            await self.client.place_order(symbol=symbol, side="SELL", quantity=position.quantity)
        except Exception:
            logger.exception("engine.order_failed", symbol=symbol, side="SELL")
            return

        trade = db.close_position(symbol, exit_price=price)
        if trade:
            logger.info(
                "engine.position_closed",
                symbol=symbol, entry=trade.entry_price, exit=price,
                pnl=trade.pnl, pnl_pct=f"{trade.pnl_pct:.2f}%",
            )

    def _interval_seconds(self) -> int:
        mapping = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
        return mapping.get(settings.trading.interval, 3600)
