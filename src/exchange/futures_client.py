"""Binance Futures client."""

from binance import AsyncClient
import pandas as pd
import structlog

from src.core.config import settings

logger = structlog.get_logger()


class FuturesClient:
    """Async wrapper around Binance Futures API."""

    def __init__(self) -> None:
        self._client: AsyncClient | None = None

    async def connect(self) -> None:
        self._client = await AsyncClient.create(
            api_key=settings.exchange.api_key,
            api_secret=settings.exchange.api_secret,
            testnet=settings.exchange.testnet,
        )
        self._client.REQUEST_TIMEOUT = 10
        self._client.REQUEST_RECVWINDOW = 60000
        await self.sync_time()
        logger.info("futures.connected", testnet=settings.exchange.testnet,
                     time_offset=f"{self._client.timestamp_offset}ms")

    async def sync_time(self) -> None:
        """서버 시간과 동기화. 주기적으로 호출 권장."""
        import time
        server_time = await self._client.futures_time()
        self._client.timestamp_offset = server_time["serverTime"] - int(time.time() * 1000)
        logger.debug("futures.time_synced", offset=f"{self._client.timestamp_offset}ms")

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close_connection()

    @property
    def client(self) -> AsyncClient:
        if self._client is None:
            raise RuntimeError("FuturesClient not connected.")
        return self._client

    async def set_leverage(self, symbol: str, leverage: int = 5) -> None:
        try:
            await self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            logger.info("futures.leverage_set", symbol=symbol, leverage=leverage)
        except Exception as e:
            logger.warning("futures.leverage_error", symbol=symbol, error=str(e))

    async def get_candles(
        self, symbol: str, interval: str = "1m", limit: int = 200,
    ) -> pd.DataFrame:
        MAX_PER_REQUEST = 1500

        if limit <= MAX_PER_REQUEST:
            raw = await self.client.futures_klines(
                symbol=symbol, interval=interval, limit=limit,
            )
        else:
            # 1500개 초과 시 여러 번 나눠서 조회
            raw = []
            remaining = limit
            end_time = None
            while remaining > 0:
                batch = min(remaining, MAX_PER_REQUEST)
                params = dict(symbol=symbol, interval=interval, limit=batch)
                if end_time:
                    params["endTime"] = end_time
                batch_raw = await self.client.futures_klines(**params)
                if not batch_raw:
                    break
                raw = batch_raw + raw  # 시간순 정렬
                end_time = batch_raw[0][0] - 1  # 다음 배치는 이전 시점부터
                remaining -= len(batch_raw)
                if len(batch_raw) < batch:
                    break

        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)
        return df

    async def get_price(self, symbol: str) -> float:
        ticker = await self.client.futures_symbol_ticker(symbol=symbol)
        return float(ticker["price"])

    async def get_balance(self) -> float:
        try:
            balances = await self.client.futures_account_balance()
            for b in balances:
                if b["asset"] == "USDT":
                    return float(b["availableBalance"])
            return 0.0
        except Exception:
            logger.exception("futures.balance_error")
            return 0.0

    async def open_long(self, symbol: str, quantity: float) -> dict:
        """Open long (or close short and open long)."""
        result = await self.client.futures_create_order(
            symbol=symbol, side="BUY", type="MARKET",
            quantity=quantity,
        )
        logger.info("futures.long_opened", symbol=symbol, qty=quantity,
                     order_id=result.get("orderId"))
        return result

    async def open_short(self, symbol: str, quantity: float) -> dict:
        """Open short (or close long and open short)."""
        result = await self.client.futures_create_order(
            symbol=symbol, side="SELL", type="MARKET",
            quantity=quantity,
        )
        logger.info("futures.short_opened", symbol=symbol, qty=quantity,
                     order_id=result.get("orderId"))
        return result

    async def close_long(self, symbol: str, quantity: float) -> dict:
        result = await self.client.futures_create_order(
            symbol=symbol, side="SELL", type="MARKET",
            quantity=quantity, reduceOnly=True,
        )
        logger.info("futures.long_closed", symbol=symbol, qty=quantity)
        return result

    async def close_short(self, symbol: str, quantity: float) -> dict:
        result = await self.client.futures_create_order(
            symbol=symbol, side="BUY", type="MARKET",
            quantity=quantity, reduceOnly=True,
        )
        logger.info("futures.short_closed", symbol=symbol, qty=quantity)
        return result

    async def get_position(self, symbol: str) -> dict | None:
        """Get current futures position info."""
        positions = await self.client.futures_position_information(symbol=symbol)
        for p in positions:
            if p["symbol"] == symbol and float(p["positionAmt"]) != 0:
                qty = abs(float(p["positionAmt"]))
                entry = float(p["entryPrice"])
                mark = float(p.get("markPrice", entry))
                notional = abs(float(p.get("notional", qty * mark)))
                margin = float(p.get("positionInitialMargin", 0))
                return {
                    "symbol": symbol,
                    "side": "LONG" if float(p["positionAmt"]) > 0 else "SHORT",
                    "quantity": qty,
                    "entry_price": entry,
                    "mark_price": mark,
                    "unrealized_pnl": float(p["unRealizedProfit"]),
                    "notional": notional,        # 포지션 규모 (레버리지 적용)
                    "margin": margin,            # 실제 투입 증거금
                }
        return None

    async def get_recent_fees(self, symbol: str, limit: int = 10) -> float:
        """최근 거래의 수수료 합계 (마지막 포지션 왕복 수수료 추정)."""
        try:
            trades = await self.client.futures_account_trades(
                symbol=symbol, limit=limit)
            if not trades:
                return 0.0
            # 가장 최근 거래들의 수수료 합산 (같은 orderId 그룹)
            # 마지막 2개 주문(진입+청산)의 수수료
            order_ids = set()
            total_fee = 0.0
            for t in reversed(trades):
                oid = t["orderId"]
                if oid not in order_ids:
                    order_ids.add(oid)
                if len(order_ids) > 2:
                    break
                total_fee += float(t["commission"])
            return total_fee
        except Exception:
            logger.exception("futures.fee_query_error", symbol=symbol)
            return 0.0

    async def get_account_summary(self) -> dict:
        """Get account balance + total unrealized PnL."""
        balances = await self.client.futures_account_balance()
        usdt = next((b for b in balances if b["asset"] == "USDT"), None)
        if not usdt:
            return {"balance": 0, "available": 0, "unrealized_pnl": 0}
        return {
            "balance": float(usdt["balance"]),
            "available": float(usdt.get("availableBalance", usdt.get("withdrawAvailable", 0))),
            "unrealized_pnl": float(usdt.get("crossUnPnl", 0)),
        }
