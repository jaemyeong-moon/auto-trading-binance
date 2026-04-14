"""Order book feature utilities.

All functions are pure (no I/O) and operate on raw order book data
as returned by BinanceClient.get_order_book().

Order entries are [price, quantity] pairs, same format as Binance API.
"""

from __future__ import annotations


def bid_ask_ratio(
    bids: list[list[float]],
    asks: list[list[float]],
) -> float:
    """Compute the bid-to-ask depth ratio.

    Aggregates total notional depth on each side and returns the ratio.
    A value > 1.0 indicates buyers dominate (more bid liquidity than ask).
    A value < 1.0 indicates sellers dominate.
    Returns 1.0 when both sides are empty (neutral fallback).

    Args:
        bids: List of [price, qty] bid levels, best bid first.
        asks: List of [price, qty] ask levels, best ask first.

    Returns:
        bid_total_qty / ask_total_qty, or 1.0 if ask total is zero.
    """
    bid_total = sum(qty for _, qty in bids)
    ask_total = sum(qty for _, qty in asks)

    if ask_total == 0.0:
        return 1.0

    return bid_total / ask_total


def spread_pct(best_bid: float, best_ask: float) -> float:
    """Compute the bid-ask spread as a percentage of the mid price.

    Formula: (ask - bid) / mid * 100

    Args:
        best_bid: Highest bid price.
        best_ask: Lowest ask price.

    Returns:
        Spread percentage. Returns 0.0 when best_bid is 0 (safety guard).
    """
    if best_bid <= 0.0:
        return 0.0

    mid = (best_bid + best_ask) / 2.0
    return (best_ask - best_bid) / mid * 100.0


def detect_wall(
    orders: list[list[float]],
    threshold_mult: float = 3.0,
) -> list[dict]:
    """Detect abnormally large orders that may act as price walls.

    An order level is flagged as a wall when its quantity exceeds
    ``threshold_mult`` times the mean quantity across all provided levels.

    Args:
        orders: List of [price, qty] levels (bids or asks).
        threshold_mult: Multiplier applied to mean qty. Default 3.0.

    Returns:
        List of dicts with keys ``price`` and ``qty`` for each wall level,
        sorted by qty descending. Empty list when no walls are detected or
        when there are fewer than 2 levels.
    """
    if len(orders) < 2:
        return []

    quantities = [qty for _, qty in orders]
    mean_qty = sum(quantities) / len(quantities)
    cutoff = mean_qty * threshold_mult

    walls = [
        {"price": price, "qty": qty}
        for price, qty in orders
        if qty >= cutoff
    ]

    walls.sort(key=lambda w: w["qty"], reverse=True)
    return walls
