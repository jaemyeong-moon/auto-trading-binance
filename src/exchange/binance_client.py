"""Binance exchange client wrapping python-binance."""

from binance import AsyncClient, BinanceSocketManager
import pandas as pd
import structlog

from src.core.config import settings

logger = structlog.get_logger()


class BinanceClient:
    """Async wrapper around Binance API."""

    def __init__(self) -> None:
        self._client: AsyncClient | None = None
        self._socket_manager: BinanceSocketManager | None = None

    async def connect(self) -> None:
        """Initialize the async Binance client."""
        self._client = await AsyncClient.create(
            api_key=settings.exchange.api_key,
            api_secret=settings.exchange.api_secret,
            testnet=settings.exchange.testnet,
        )
        logger.info("binance.connected", testnet=settings.exchange.testnet)

    async def disconnect(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.close_connection()
            logger.info("binance.disconnected")

    @property
    def client(self) -> AsyncClient:
        if self._client is None:
            raise RuntimeError("BinanceClient not connected. Call connect() first.")
        return self._client

    async def get_candles(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data as a DataFrame."""
        raw = await self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(
            raw,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ],
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        return df

    async def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
    ) -> dict:
        """Place an order on Binance."""
        logger.info("binance.order", symbol=symbol, side=side, quantity=quantity, type=order_type)
        result = await self.client.create_order(
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
        )
        logger.info("binance.order_filled", order_id=result.get("orderId"))
        return result

    async def get_balance(self, asset: str = "USDT") -> float:
        """Get available balance for an asset."""
        account = await self.client.get_account()
        for balance in account["balances"]:
            if balance["asset"] == asset:
                return float(balance["free"])
        return 0.0

    async def get_ticker_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        ticker = await self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker["price"])
