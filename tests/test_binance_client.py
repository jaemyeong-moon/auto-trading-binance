"""Contract tests for FuturesClient — all Binance API calls are mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.exchange.futures_client import FuturesClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kline_row(
    open_time: int = 1_700_000_000_000,
    open: str = "50000.0",
    high: str = "50500.0",
    low: str = "49500.0",
    close: str = "50250.0",
    volume: str = "10.5",
) -> list:
    """One raw kline row as returned by Binance."""
    return [
        open_time, open, high, low, close, volume,
        open_time + 59_999, "525000.0", 100,
        "5.0", "250000.0", "0",
    ]


def _connected_client() -> FuturesClient:
    """Return a FuturesClient whose _client is an AsyncMock (no network)."""
    fc = FuturesClient()
    fc._client = AsyncMock()
    return fc


# ---------------------------------------------------------------------------
# get_candles
# ---------------------------------------------------------------------------

class TestGetCandles:
    async def test_returns_dataframe_with_correct_columns(self):
        fc = _connected_client()
        raw = [_make_kline_row(open_time=1_700_000_000_000 + i * 60_000) for i in range(5)]
        fc._client.futures_klines = AsyncMock(return_value=raw)

        df = await fc.get_candles("BTCUSDT", interval="1m", limit=5)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns
            assert df[col].dtype == float

    async def test_index_is_datetime(self):
        fc = _connected_client()
        raw = [_make_kline_row()]
        fc._client.futures_klines = AsyncMock(return_value=raw)

        df = await fc.get_candles("BTCUSDT")

        assert isinstance(df.index, pd.DatetimeIndex)

    async def test_empty_response_returns_empty_dataframe(self):
        fc = _connected_client()
        fc._client.futures_klines = AsyncMock(return_value=[])

        df = await fc.get_candles("BTCUSDT")

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    async def test_passes_start_time_when_given(self):
        fc = _connected_client()
        fc._client.futures_klines = AsyncMock(return_value=[_make_kline_row()])

        await fc.get_candles("BTCUSDT", interval="1h", limit=100, start_time=1_700_000_000_000)

        call_kwargs = fc._client.futures_klines.call_args.kwargs
        assert call_kwargs["startTime"] == 1_700_000_000_000

    async def test_large_limit_triggers_pagination(self):
        """limit > 1500 should call futures_klines multiple times."""
        fc = _connected_client()
        # First batch: 1500 rows; second batch: 200 rows
        batch_1 = [_make_kline_row(open_time=1_700_000_000_000 + i * 60_000) for i in range(1500)]
        batch_2 = [_make_kline_row(open_time=i * 60_000) for i in range(200)]
        fc._client.futures_klines = AsyncMock(side_effect=[batch_1, batch_2])

        df = await fc.get_candles("BTCUSDT", limit=1700)

        assert fc._client.futures_klines.call_count == 2
        assert len(df) == 1700

    async def test_duplicates_are_removed(self):
        fc = _connected_client()
        row = _make_kline_row()
        fc._client.futures_klines = AsyncMock(return_value=[row, row])

        df = await fc.get_candles("BTCUSDT")

        assert len(df) == 1

    async def test_raises_when_not_connected(self):
        fc = FuturesClient()  # _client is None

        with pytest.raises(RuntimeError, match="not connected"):
            await fc.get_candles("BTCUSDT")


# ---------------------------------------------------------------------------
# get_price
# ---------------------------------------------------------------------------

class TestGetPrice:
    async def test_returns_float(self):
        fc = _connected_client()
        fc._client.futures_symbol_ticker = AsyncMock(return_value={"price": "67432.10"})

        price = await fc.get_price("BTCUSDT")

        assert isinstance(price, float)
        assert price == pytest.approx(67432.10)

    async def test_passes_symbol(self):
        fc = _connected_client()
        fc._client.futures_symbol_ticker = AsyncMock(return_value={"price": "3000.0"})

        await fc.get_price("ETHUSDT")

        fc._client.futures_symbol_ticker.assert_awaited_once_with(symbol="ETHUSDT")


# ---------------------------------------------------------------------------
# get_balance
# ---------------------------------------------------------------------------

class TestGetBalance:
    async def test_returns_usdt_available_balance(self):
        fc = _connected_client()
        fc._client.futures_account_balance = AsyncMock(return_value=[
            {"asset": "BNB", "availableBalance": "5.0"},
            {"asset": "USDT", "availableBalance": "1234.56"},
        ])

        balance = await fc.get_balance()

        assert balance == pytest.approx(1234.56)

    async def test_returns_zero_when_no_usdt(self):
        fc = _connected_client()
        fc._client.futures_account_balance = AsyncMock(return_value=[
            {"asset": "BNB", "availableBalance": "1.0"},
        ])

        balance = await fc.get_balance()

        assert balance == 0.0

    async def test_returns_zero_on_api_error(self):
        fc = _connected_client()
        fc._client.futures_account_balance = AsyncMock(side_effect=Exception("connection refused"))

        balance = await fc.get_balance()

        assert balance == 0.0


# ---------------------------------------------------------------------------
# open_long / open_short
# ---------------------------------------------------------------------------

class TestOpenOrders:
    async def test_open_long_returns_order_id(self):
        fc = _connected_client()
        fc._client.futures_create_order = AsyncMock(return_value={"orderId": 111})

        result = await fc.open_long("BTCUSDT", 0.01)

        assert result["orderId"] == 111

    async def test_open_long_uses_buy_side(self):
        fc = _connected_client()
        fc._client.futures_create_order = AsyncMock(return_value={"orderId": 1})

        await fc.open_long("BTCUSDT", 0.01)

        kwargs = fc._client.futures_create_order.call_args.kwargs
        assert kwargs["side"] == "BUY"
        assert kwargs["type"] == "MARKET"
        assert "reduceOnly" not in kwargs

    async def test_open_short_returns_order_id(self):
        fc = _connected_client()
        fc._client.futures_create_order = AsyncMock(return_value={"orderId": 222})

        result = await fc.open_short("BTCUSDT", 0.01)

        assert result["orderId"] == 222

    async def test_open_short_uses_sell_side(self):
        fc = _connected_client()
        fc._client.futures_create_order = AsyncMock(return_value={"orderId": 2})

        await fc.open_short("BTCUSDT", 0.01)

        kwargs = fc._client.futures_create_order.call_args.kwargs
        assert kwargs["side"] == "SELL"
        assert kwargs["type"] == "MARKET"
        assert "reduceOnly" not in kwargs


# ---------------------------------------------------------------------------
# close_long / close_short
# ---------------------------------------------------------------------------

class TestCloseOrders:
    async def test_close_long_sends_reduce_only(self):
        fc = _connected_client()
        fc._client.futures_create_order = AsyncMock(return_value={"orderId": 333})

        result = await fc.close_long("BTCUSDT", 0.01)

        kwargs = fc._client.futures_create_order.call_args.kwargs
        assert kwargs["reduceOnly"] is True
        assert kwargs["side"] == "SELL"
        assert result["orderId"] == 333

    async def test_close_short_sends_reduce_only(self):
        fc = _connected_client()
        fc._client.futures_create_order = AsyncMock(return_value={"orderId": 444})

        result = await fc.close_short("BTCUSDT", 0.01)

        kwargs = fc._client.futures_create_order.call_args.kwargs
        assert kwargs["reduceOnly"] is True
        assert kwargs["side"] == "BUY"
        assert result["orderId"] == 444


# ---------------------------------------------------------------------------
# set_leverage
# ---------------------------------------------------------------------------

class TestSetLeverage:
    async def test_calls_change_leverage(self):
        fc = _connected_client()
        fc._client.futures_change_leverage = AsyncMock(return_value={})

        await fc.set_leverage("BTCUSDT", leverage=10)

        fc._client.futures_change_leverage.assert_awaited_once_with(
            symbol="BTCUSDT", leverage=10
        )

    async def test_does_not_raise_on_api_error(self):
        fc = _connected_client()
        fc._client.futures_change_leverage = AsyncMock(
            side_effect=Exception("leverage not supported")
        )

        # Should silently swallow the error
        await fc.set_leverage("BTCUSDT", leverage=20)


# ---------------------------------------------------------------------------
# get_position
# ---------------------------------------------------------------------------

class TestGetPosition:
    async def test_returns_long_position(self):
        fc = _connected_client()
        fc._client.futures_position_information = AsyncMock(return_value=[
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0.05",
                "entryPrice": "60000.0",
                "markPrice": "61000.0",
                "unRealizedProfit": "50.0",
                "notional": "3050.0",
                "positionInitialMargin": "610.0",
            }
        ])

        pos = await fc.get_position("BTCUSDT")

        assert pos is not None
        assert pos["side"] == "LONG"
        assert pos["quantity"] == pytest.approx(0.05)
        assert pos["entry_price"] == pytest.approx(60000.0)
        assert pos["unrealized_pnl"] == pytest.approx(50.0)

    async def test_returns_short_position(self):
        fc = _connected_client()
        fc._client.futures_position_information = AsyncMock(return_value=[
            {
                "symbol": "BTCUSDT",
                "positionAmt": "-0.03",
                "entryPrice": "60000.0",
                "markPrice": "59000.0",
                "unRealizedProfit": "30.0",
                "notional": "-1770.0",
                "positionInitialMargin": "354.0",
            }
        ])

        pos = await fc.get_position("BTCUSDT")

        assert pos is not None
        assert pos["side"] == "SHORT"
        assert pos["quantity"] == pytest.approx(0.03)

    async def test_returns_none_when_no_position(self):
        fc = _connected_client()
        fc._client.futures_position_information = AsyncMock(return_value=[
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0",
                "entryPrice": "0.0",
                "markPrice": "60000.0",
                "unRealizedProfit": "0.0",
                "notional": "0.0",
                "positionInitialMargin": "0.0",
            }
        ])

        pos = await fc.get_position("BTCUSDT")

        assert pos is None

    async def test_returns_none_when_symbol_not_found(self):
        fc = _connected_client()
        fc._client.futures_position_information = AsyncMock(return_value=[
            {
                "symbol": "ETHUSDT",
                "positionAmt": "1.0",
                "entryPrice": "3000.0",
                "markPrice": "3100.0",
                "unRealizedProfit": "100.0",
                "notional": "3100.0",
                "positionInitialMargin": "620.0",
            }
        ])

        pos = await fc.get_position("BTCUSDT")

        assert pos is None


# ---------------------------------------------------------------------------
# place_sl_tp_orders
# ---------------------------------------------------------------------------

class TestPlaceSlTpOrders:
    async def test_places_both_sl_and_tp(self):
        fc = _connected_client()
        fc._client.futures_create_order = AsyncMock(
            side_effect=[{"orderId": 9001}, {"orderId": 9002}]
        )

        result = await fc.place_sl_tp_orders(
            symbol="BTCUSDT", side="LONG",
            quantity=0.01, sl_price=58000.0, tp_price=65000.0,
        )

        assert result["sl_order"] == 9001
        assert result["tp_order"] == 9002
        assert fc._client.futures_create_order.call_count == 2

    async def test_long_position_uses_sell_side(self):
        fc = _connected_client()
        fc._client.futures_create_order = AsyncMock(
            side_effect=[{"orderId": 1}, {"orderId": 2}]
        )

        await fc.place_sl_tp_orders(
            symbol="BTCUSDT", side="LONG",
            quantity=0.01, sl_price=58000.0, tp_price=65000.0,
        )

        calls = fc._client.futures_create_order.call_args_list
        assert calls[0].kwargs["side"] == "SELL"  # SL
        assert calls[1].kwargs["side"] == "SELL"  # TP

    async def test_short_position_uses_buy_side(self):
        fc = _connected_client()
        fc._client.futures_create_order = AsyncMock(
            side_effect=[{"orderId": 1}, {"orderId": 2}]
        )

        await fc.place_sl_tp_orders(
            symbol="BTCUSDT", side="SHORT",
            quantity=0.01, sl_price=65000.0, tp_price=55000.0,
        )

        calls = fc._client.futures_create_order.call_args_list
        assert calls[0].kwargs["side"] == "BUY"
        assert calls[1].kwargs["side"] == "BUY"

    async def test_sl_uses_stop_market_type(self):
        fc = _connected_client()
        fc._client.futures_create_order = AsyncMock(
            side_effect=[{"orderId": 1}, {"orderId": 2}]
        )

        await fc.place_sl_tp_orders(
            symbol="BTCUSDT", side="LONG",
            quantity=0.01, sl_price=58000.0, tp_price=65000.0,
        )

        sl_kwargs = fc._client.futures_create_order.call_args_list[0].kwargs
        tp_kwargs = fc._client.futures_create_order.call_args_list[1].kwargs
        assert sl_kwargs["type"] == "STOP_MARKET"
        assert tp_kwargs["type"] == "TAKE_PROFIT_MARKET"

    async def test_sl_order_error_still_places_tp(self):
        """If the SL order fails, TP should still be attempted."""
        fc = _connected_client()
        fc._client.futures_create_order = AsyncMock(
            side_effect=[Exception("SL rejected"), {"orderId": 9002}]
        )

        result = await fc.place_sl_tp_orders(
            symbol="BTCUSDT", side="LONG",
            quantity=0.01, sl_price=58000.0, tp_price=65000.0,
        )

        assert result["sl_order"] is None
        assert result["tp_order"] == 9002

    async def test_tp_order_error_returns_partial_result(self):
        fc = _connected_client()
        fc._client.futures_create_order = AsyncMock(
            side_effect=[{"orderId": 9001}, Exception("TP rejected")]
        )

        result = await fc.place_sl_tp_orders(
            symbol="BTCUSDT", side="LONG",
            quantity=0.01, sl_price=58000.0, tp_price=65000.0,
        )

        assert result["sl_order"] == 9001
        assert result["tp_order"] is None


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------

class TestErrorHandling:
    async def test_get_price_propagates_exception(self):
        fc = _connected_client()
        fc._client.futures_symbol_ticker = AsyncMock(side_effect=ConnectionError("timeout"))

        with pytest.raises(ConnectionError):
            await fc.get_price("BTCUSDT")

    async def test_open_long_propagates_exception(self):
        fc = _connected_client()
        fc._client.futures_create_order = AsyncMock(
            side_effect=Exception("insufficient margin")
        )

        with pytest.raises(Exception, match="insufficient margin"):
            await fc.open_long("BTCUSDT", 0.01)

    async def test_client_property_raises_before_connect(self):
        fc = FuturesClient()

        with pytest.raises(RuntimeError, match="not connected"):
            _ = fc.client

    async def test_disconnect_when_not_connected_is_safe(self):
        fc = FuturesClient()
        # Should not raise even if _client is None
        await fc.disconnect()
