"""Unit tests for RiskManager."""

import pytest

from src.core.risk_manager import RiskManager


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def rm() -> RiskManager:
    """Default RiskManager: 3 max positions, 5 % daily loss limit."""
    return RiskManager(max_open_positions=3, max_daily_loss_pct=0.05)


# ── Constructor validation ─────────────────────────────────────────────────────


class TestConstructor:
    def test_default_values(self):
        mgr = RiskManager()
        assert mgr.max_open_positions == 3
        assert mgr.max_daily_loss_pct == 0.05

    def test_custom_values(self):
        mgr = RiskManager(max_open_positions=5, max_daily_loss_pct=0.10)
        assert mgr.max_open_positions == 5
        assert mgr.max_daily_loss_pct == 0.10

    def test_invalid_max_positions_zero(self):
        with pytest.raises(ValueError):
            RiskManager(max_open_positions=0)

    def test_invalid_max_positions_negative(self):
        with pytest.raises(ValueError):
            RiskManager(max_open_positions=-1)

    def test_invalid_daily_loss_zero(self):
        with pytest.raises(ValueError):
            RiskManager(max_daily_loss_pct=0.0)

    def test_invalid_daily_loss_over_one(self):
        with pytest.raises(ValueError):
            RiskManager(max_daily_loss_pct=1.1)

    def test_daily_loss_exactly_one_is_valid(self):
        """100 % loss limit is extreme but technically valid."""
        mgr = RiskManager(max_daily_loss_pct=1.0)
        assert mgr.max_daily_loss_pct == 1.0


# ── can_open ──────────────────────────────────────────────────────────────────


class TestCanOpen:
    def test_normal_conditions_returns_true(self, rm):
        allowed, reason = rm.can_open(current_positions=0, daily_pnl_pct=0.0)
        assert allowed is True
        assert reason == ""

    def test_positions_below_max(self, rm):
        allowed, reason = rm.can_open(current_positions=2, daily_pnl_pct=0.0)
        assert allowed is True

    def test_positions_at_max_returns_false(self, rm):
        allowed, reason = rm.can_open(current_positions=3, daily_pnl_pct=0.0)
        assert allowed is False
        assert reason == "max_positions"

    def test_positions_above_max_returns_false(self, rm):
        allowed, reason = rm.can_open(current_positions=10, daily_pnl_pct=0.0)
        assert allowed is False
        assert reason == "max_positions"

    def test_daily_dd_exceeded_returns_false(self, rm):
        # -5 % exactly hits the limit → blocked
        allowed, reason = rm.can_open(current_positions=0, daily_pnl_pct=-0.05)
        assert allowed is False
        assert reason == "daily_dd_limit"

    def test_daily_dd_well_above_limit_returns_false(self, rm):
        allowed, reason = rm.can_open(current_positions=0, daily_pnl_pct=-0.20)
        assert allowed is False
        assert reason == "daily_dd_limit"

    def test_daily_dd_just_within_limit(self, rm):
        # -4.99 % is still inside the 5 % threshold
        allowed, reason = rm.can_open(current_positions=0, daily_pnl_pct=-0.0499)
        assert allowed is True

    def test_positive_pnl_always_allowed(self, rm):
        allowed, reason = rm.can_open(current_positions=1, daily_pnl_pct=0.10)
        assert allowed is True

    def test_positions_check_takes_priority_over_dd(self, rm):
        """Both conditions met — max_positions is checked first."""
        allowed, reason = rm.can_open(current_positions=3, daily_pnl_pct=-0.10)
        assert allowed is False
        assert reason == "max_positions"

    def test_single_position_limit(self):
        mgr = RiskManager(max_open_positions=1, max_daily_loss_pct=0.05)
        assert mgr.can_open(0, 0.0) == (True, "")
        assert mgr.can_open(1, 0.0)[0] is False


# ── position_size ──────────────────────────────────────────────────────────────


class TestPositionSize:
    def test_basic_calculation(self, rm):
        size = rm.position_size(balance=1000.0, strategy_pct=0.20)
        assert size == pytest.approx(200.0)

    def test_zero_balance_returns_zero(self, rm):
        assert rm.position_size(balance=0.0, strategy_pct=0.20) == 0.0

    def test_negative_balance_returns_zero(self, rm):
        assert rm.position_size(balance=-500.0, strategy_pct=0.20) == 0.0

    def test_strategy_pct_one(self, rm):
        size = rm.position_size(balance=500.0, strategy_pct=1.0)
        assert size == pytest.approx(500.0)

    def test_strategy_pct_zero(self, rm):
        assert rm.position_size(balance=1000.0, strategy_pct=0.0) == 0.0

    def test_strategy_pct_clamped_above_one(self, rm):
        """Oversized pct is clamped to 1.0 (100 % of balance)."""
        size = rm.position_size(balance=1000.0, strategy_pct=2.0)
        assert size == pytest.approx(1000.0)

    def test_strategy_pct_clamped_below_zero(self, rm):
        size = rm.position_size(balance=1000.0, strategy_pct=-0.5)
        assert size == 0.0

    # ATR scaling
    def test_atr_equal_to_baseline_no_change(self, rm):
        size = rm.position_size(
            balance=1000.0, strategy_pct=0.20, atr=10.0, atr_baseline=10.0
        )
        assert size == pytest.approx(200.0)

    def test_atr_double_baseline_halves_size(self, rm):
        """ATR = 2× baseline → scalar = 0.5 → half the base size."""
        size = rm.position_size(
            balance=1000.0, strategy_pct=0.20, atr=20.0, atr_baseline=10.0
        )
        assert size == pytest.approx(100.0)

    def test_atr_triple_baseline_capped_at_half(self, rm):
        """Scalar is capped at 0.5, so ≥2× ATR gives the same minimum."""
        size_triple = rm.position_size(
            balance=1000.0, strategy_pct=0.20, atr=30.0, atr_baseline=10.0
        )
        assert size_triple == pytest.approx(100.0)

    def test_atr_half_baseline_no_upsize(self, rm):
        """Calm market (ATR < baseline) — scalar capped at 1.0, no increase."""
        size = rm.position_size(
            balance=1000.0, strategy_pct=0.20, atr=5.0, atr_baseline=10.0
        )
        assert size == pytest.approx(200.0)

    def test_atr_without_baseline_ignored(self, rm):
        """ATR alone (no baseline) behaves like no ATR adjustment."""
        size = rm.position_size(
            balance=1000.0, strategy_pct=0.20, atr=10.0, atr_baseline=0.0
        )
        assert size == pytest.approx(200.0)

    def test_baseline_without_atr_ignored(self, rm):
        size = rm.position_size(
            balance=1000.0, strategy_pct=0.20, atr=0.0, atr_baseline=10.0
        )
        assert size == pytest.approx(200.0)

    def test_large_balance(self, rm):
        size = rm.position_size(balance=1_000_000.0, strategy_pct=0.05)
        assert size == pytest.approx(50_000.0)


# ── daily_dd_ok ────────────────────────────────────────────────────────────────


class TestDailyDdOk:
    def test_zero_pnl_ok(self, rm):
        assert rm.daily_dd_ok(0.0) is True

    def test_positive_pnl_ok(self, rm):
        assert rm.daily_dd_ok(0.10) is True

    def test_small_loss_ok(self, rm):
        assert rm.daily_dd_ok(-0.01) is True

    def test_loss_just_below_limit(self, rm):
        assert rm.daily_dd_ok(-0.0499) is True

    def test_loss_exactly_at_limit_blocked(self, rm):
        # -0.05 is NOT strictly greater than -0.05
        assert rm.daily_dd_ok(-0.05) is False

    def test_loss_above_limit_blocked(self, rm):
        assert rm.daily_dd_ok(-0.10) is False

    def test_extreme_loss_blocked(self, rm):
        assert rm.daily_dd_ok(-1.0) is False

    def test_custom_limit(self):
        mgr = RiskManager(max_daily_loss_pct=0.02)
        assert mgr.daily_dd_ok(-0.019) is True
        assert mgr.daily_dd_ok(-0.02) is False
        assert mgr.daily_dd_ok(-0.021) is False
