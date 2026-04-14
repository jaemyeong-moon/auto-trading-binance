"""Derivatives market feature utilities.

Pure functions for funding rate and open interest (OI) signals,
as used in futures markets (e.g. Binance USDT-M perpetuals).
"""

from __future__ import annotations


def funding_rate_signal(
    rates: list[float],
    extreme_threshold: float = 0.001,
) -> str:
    """Derive a directional bias from recent funding rate history.

    High positive funding means longs are paying shorts heavily —
    the market is likely over-bought, so contrarian SHORT is preferred.
    High negative funding means shorts are paying longs —
    the market may be over-sold, so contrarian LONG is preferred.
    Rates near zero are neutral.

    The last element in ``rates`` is treated as the most recent value.

    Args:
        rates: Sequence of funding rate values (e.g. 8-h snapshots).
              Positive values indicate longs paying shorts.
        extreme_threshold: Absolute rate level considered extreme.
                           Default 0.001 (0.1% per funding period).

    Returns:
        ``"SHORT"`` when the latest rate exceeds +threshold,
        ``"LONG"``  when the latest rate is below -threshold,
        ``"NEUTRAL"`` otherwise.  Returns ``"NEUTRAL"`` for an empty list.
    """
    if not rates:
        return "NEUTRAL"

    latest = rates[-1]

    if latest > extreme_threshold:
        return "SHORT"
    if latest < -extreme_threshold:
        return "LONG"
    return "NEUTRAL"


def oi_change_rate(current_oi: float, prev_oi: float) -> float:
    """Compute the percentage change in open interest.

    Formula: (current - prev) / prev * 100

    Args:
        current_oi: Current open interest value.
        prev_oi: Previous open interest value used as the base.

    Returns:
        OI change in percent. Returns 0.0 when prev_oi is 0 (safety guard).
    """
    if prev_oi == 0.0:
        return 0.0

    return (current_oi - prev_oi) / prev_oi * 100.0


def oi_divergence_signal(
    price_change_pct: float,
    oi_change_pct: float,
) -> str:
    """Detect price-OI divergence patterns.

    Classic interpretation:
    - Price up + OI down  → existing longs closing, weakening trend (BEARISH_DIV)
    - Price down + OI up  → new shorts opening into weakness, potential exhaustion (BULLISH_DIV)
    - Price up + OI up    → trend confirmed by new money (BULLISH_CONFIRM)
    - Price down + OI down → shorts closing, potential bottom (BEARISH_CONFIRM)

    Neutral zone: changes within ±0.0 are treated according to sign.
    Exactly zero is treated as "no change" and maps to the neutral-ish branch.

    Args:
        price_change_pct: Price change percentage (positive = up).
        oi_change_pct: OI change percentage (positive = increasing).

    Returns:
        One of: ``"BEARISH_DIV"``, ``"BULLISH_DIV"``,
        ``"BULLISH_CONFIRM"``, ``"BEARISH_CONFIRM"``.
    """
    price_up = price_change_pct > 0
    oi_up = oi_change_pct > 0

    if price_up and not oi_up:
        return "BEARISH_DIV"
    if not price_up and oi_up:
        return "BULLISH_DIV"
    if price_up and oi_up:
        return "BULLISH_CONFIRM"
    # price down, oi down
    return "BEARISH_CONFIRM"
