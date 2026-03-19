"""Application entry point — Futures scalping bot."""

import asyncio
import logging
import signal
import sys

import structlog
from rich.console import Console

from src.core.config import settings
from src.core import database as db
from src.core.futures_engine import FuturesEngine
from src.exchange.futures_client import FuturesClient

console = Console()


async def _run() -> None:
    db.init_db()

    client = FuturesClient()
    await client.connect()

    engine = FuturesEngine(client=client)

    # 설정된 심볼 모두 시작
    symbols = settings.trading.symbols
    for symbol in symbols:
        await engine.start_symbol(symbol)

    console.print("[bold green]⚡ Futures Scalper started[/bold green]")
    console.print(f"  Symbols: {symbols}")
    console.print(f"  Leverage: 5x")
    console.print(f"  Strategy: momentum_flip_scalper")
    console.print(f"  Testnet: {settings.exchange.testnet}")

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    try:
        await stop_event.wait()
    finally:
        await engine.stop_all()
        await client.disconnect()
        console.print("[bold red]⚡ Futures Scalper stopped[/bold red]")


def main() -> None:
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
