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

    def kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        safety_factor: float = 0.25,
        max_pct: float = 0.25,
    ) -> float:
        """Kelly Criterion 기반 최적 포지션 비율.

        Kelly 공식:
            kelly_pct = win_rate - (1 - win_rate) / (avg_win / avg_loss)

        실제 적용:
            result = kelly_pct * safety_factor
            result = min(result, max_pct)
            result = max(result, 0.0)   # 음수 Kelly → 0 (거래하지 말 것)

        Args:
            win_rate: 전략 승률 (0.0 – 1.0).
            avg_win: 이기는 거래의 평균 수익률 (양수, e.g. 0.02 = 2 %).
            avg_loss: 지는 거래의 평균 손실률 (양수, e.g. 0.01 = 1 %).
            safety_factor: Kelly 값에 곱하는 안전계수 (기본 0.25 = 1/4 Kelly).
            max_pct: 결과 상한 (기본 0.25 = 25 %).

        Returns:
            권장 포지션 비율 (0.0 – max_pct).
            avg_loss == 0 또는 safety_factor == 0 이면 0.0 반환.
        """
        if avg_loss <= 0.0 or safety_factor <= 0.0:
            return 0.0

        win_loss_ratio = avg_win / avg_loss
        kelly_pct = win_rate - (1.0 - win_rate) / win_loss_ratio

        # 음수 Kelly → 기댓값 마이너스, 포지션 진입 금지
        if kelly_pct <= 0.0:
            return 0.0

        result = kelly_pct * safety_factor
        return min(result, max_pct)

    def check_correlation(
        self,
        symbol: str,
        open_symbols: list[str],
        correlation_matrix: dict[tuple[str, str], float],
        threshold: float = 0.8,
    ) -> tuple[bool, str]:
        """Check whether *symbol* is highly correlated with any open position.

        Args:
            symbol: The symbol being considered for entry.
            open_symbols: List of symbols that already have open positions.
            correlation_matrix: Mapping of ``(sym_a, sym_b) -> correlation``.
                Both ``(A, B)`` and ``(B, A)`` orderings are normalised
                internally, so you only need to supply one direction.
            threshold: Correlation value above which entry is blocked
                (exclusive; default 0.8).

        Returns:
            ``(True, "")`` when the symbol may be traded, or
            ``(False, "correlated_with_{sym}")`` when a correlation above
            the threshold is found.  Same-symbol is always blocked.
        """
        for sym in open_symbols:
            if sym == symbol:
                return False, f"correlated_with_{sym}"

            # Normalise key ordering so both (A,B) and (B,A) work
            corr = correlation_matrix.get((symbol, sym))
            if corr is None:
                corr = correlation_matrix.get((sym, symbol))

            if corr is not None and corr > threshold:
                return False, f"correlated_with_{sym}"

        return True, ""

    def daily_dd_ok(self, daily_pnl_pct: float) -> bool:
        """Return True if today's loss has not exceeded the daily limit.

        Args:
            daily_pnl_pct: Fraction of balance lost today (negative = loss).
        """
        # A negative pnl_pct whose absolute value exceeds the limit means
        # we have blown through the drawdown threshold.
        return daily_pnl_pct > -self.max_daily_loss_pct
