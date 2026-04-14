"""Risk management utilities for position sizing and drawdown control."""


class RiskManager:
    """Pure calculation class for trade risk controls.

    No external I/O — all state is passed in by the caller.

    Args:
        max_open_positions: Maximum number of simultaneously open positions.
        max_daily_loss_pct: Daily loss threshold as a fraction of balance
            (e.g. 0.05 = 5 %).  Once breached, new entries are blocked.
    """

    def __init__(
        self,
        max_open_positions: int = 3,
        max_daily_loss_pct: float = 0.05,
    ) -> None:
        if max_open_positions < 1:
            raise ValueError("max_open_positions must be >= 1")
        if not (0.0 < max_daily_loss_pct <= 1.0):
            raise ValueError("max_daily_loss_pct must be in (0, 1]")

        self.max_open_positions = max_open_positions
        self.max_daily_loss_pct = max_daily_loss_pct

    # ── Public API ─────────────────────────────────────────────────────────

    def can_open(
        self,
        current_positions: int,
        daily_pnl_pct: float,
    ) -> tuple[bool, str]:
        """Determine whether a new position may be opened.

        Args:
            current_positions: Number of positions currently open.
            daily_pnl_pct: Today's realised PnL as a fraction of balance
                (negative = loss, e.g. -0.03 = -3 %).

        Returns:
            (allowed, reason) where reason is an empty string when allowed,
            or one of "max_positions" / "daily_dd_limit" when blocked.
        """
        if current_positions >= self.max_open_positions:
            return False, "max_positions"

        if not self.daily_dd_ok(daily_pnl_pct):
            return False, "daily_dd_limit"

        return True, ""

    def position_size(
        self,
        balance: float,
        strategy_pct: float,
        atr: float = 0.0,
        atr_baseline: float = 0.0,
    ) -> float:
        """Calculate notional position size.

        Base formula: balance * strategy_pct

        When both *atr* and *atr_baseline* are positive, applies a
        volatility scalar that shrinks the size when current ATR is
        higher than the baseline:

            scalar = atr_baseline / atr   (capped to [0.5, 1.0])

        This means a 2× ATR spike halves the allocated size, while
        calm markets use the full strategy percentage.

        Args:
            balance: Available account balance in quote currency.
            strategy_pct: Base fraction of balance to risk (0 – 1).
            atr: Current ATR value.
            atr_baseline: Reference (average) ATR used for scaling.

        Returns:
            Notional size >= 0.  Returns 0 when balance <= 0.
        """
        if balance <= 0:
            return 0.0

        # Clamp strategy_pct to a sensible range
        strategy_pct = max(0.0, min(strategy_pct, 1.0))

        base_size = balance * strategy_pct

        if atr > 0 and atr_baseline > 0:
            scalar = atr_baseline / atr
            scalar = max(0.5, min(scalar, 1.0))
            return base_size * scalar

        return base_size

    def daily_dd_ok(self, daily_pnl_pct: float) -> bool:
        """Return True if today's loss has not exceeded the daily limit.

        Args:
            daily_pnl_pct: Fraction of balance lost today (negative = loss).
        """
        # A negative pnl_pct whose absolute value exceeds the limit means
        # we have blown through the drawdown threshold.
        return daily_pnl_pct > -self.max_daily_loss_pct
