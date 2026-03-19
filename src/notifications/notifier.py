"""Notification system — Telegram and console alerts."""

import httpx
import structlog

from src.core.models import Signal

logger = structlog.get_logger()


class TelegramNotifier:
    """Send trade alerts to Telegram."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self._base_url = f"https://api.telegram.org/bot{bot_token}"

    async def send(self, message: str) -> None:
        """Send a text message to Telegram."""
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{self._base_url}/sendMessage",
                    json={"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"},
                )
                resp.raise_for_status()
            except Exception:
                logger.exception("telegram.send_failed")

    async def notify_signal(self, signal: Signal) -> None:
        """Format and send a trading signal notification."""
        emoji = {"buy": "🟢", "sell": "🔴", "hold": "⏸️"}
        msg = (
            f"{emoji.get(signal.type.value, '❓')} <b>{signal.type.value.upper()}</b>\n"
            f"심볼: <code>{signal.symbol}</code>\n"
            f"신뢰도: {signal.confidence:.0%}\n"
            f"전략: {signal.source}\n"
        )
        await self.send(msg)

    async def notify_trade(
        self, symbol: str, side: str, price: float, quantity: float, pnl: float | None = None
    ) -> None:
        """Send trade execution notification."""
        emoji = "📈" if side == "BUY" else "📉"
        msg = (
            f"{emoji} <b>체결: {side}</b>\n"
            f"심볼: <code>{symbol}</code>\n"
            f"가격: {price:,.2f}\n"
            f"수량: {quantity:.6f}\n"
        )
        if pnl is not None:
            pnl_emoji = "✅" if pnl >= 0 else "❌"
            msg += f"손익: {pnl_emoji} {pnl:+,.2f} USDT\n"
        await self.send(msg)


class ConsoleNotifier:
    """Print alerts to console (fallback when Telegram is not configured)."""

    async def notify_signal(self, signal: Signal) -> None:
        logger.info(
            "signal.alert",
            symbol=signal.symbol,
            type=signal.type.value,
            confidence=f"{signal.confidence:.0%}",
            source=signal.source,
        )

    async def notify_trade(
        self, symbol: str, side: str, price: float, quantity: float, pnl: float | None = None
    ) -> None:
        logger.info(
            "trade.alert",
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            pnl=pnl,
        )
