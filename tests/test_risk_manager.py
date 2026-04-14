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


# ── kelly_size ─────────────────────────────────────────────────────────────────


class TestKellySize:
    """Kelly Criterion 기반 포지션 비율 계산 테스트."""

    def test_standard_case_60pct_win_2to1_ratio(self, rm):
        """승률 60 %, 평균수익 2 %, 평균손실 1 % → Kelly 40 % × 0.25 = 10 %."""
        # kelly = 0.60 - (1 - 0.60) / (0.02 / 0.01)
        #       = 0.60 - 0.40 / 2.0
        #       = 0.60 - 0.20
        #       = 0.40
        # result = 0.40 * 0.25 = 0.10
        result = rm.kelly_size(win_rate=0.60, avg_win=0.02, avg_loss=0.01)
        assert result == pytest.approx(0.10)

    def test_negative_kelly_returns_zero(self, rm):
        """승률 40 %, 손익비 < 1 → Kelly 음수 → 0 % (거래하지 말 것)."""
        # kelly = 0.40 - 0.60 / (0.005 / 0.01) = 0.40 - 1.20 = -0.80  → 0
        result = rm.kelly_size(win_rate=0.40, avg_win=0.005, avg_loss=0.01)
        assert result == 0.0

    def test_cap_at_max_pct(self, rm):
        """Kelly > max_pct → max_pct 반환."""
        # kelly = 0.90 - 0.10 / (0.10 / 0.01) = 0.90 - 0.01 = 0.89
        # result = 0.89 * 0.25 = 0.2225 → capped at max_pct=0.20
        result = rm.kelly_size(
            win_rate=0.90, avg_win=0.10, avg_loss=0.01, max_pct=0.20
        )
        assert result == pytest.approx(0.20)

    def test_avg_loss_zero_returns_zero(self, rm):
        """avg_loss = 0 → 0 % (0으로 나누기 방지)."""
        result = rm.kelly_size(win_rate=0.60, avg_win=0.02, avg_loss=0.0)
        assert result == 0.0

    def test_safety_factor_zero_returns_zero(self, rm):
        """안전계수 0 → 0 %."""
        result = rm.kelly_size(
            win_rate=0.60, avg_win=0.02, avg_loss=0.01, safety_factor=0.0
        )
        assert result == 0.0

    def test_breakeven_kelly_is_zero(self, rm):
        """Kelly = 0 인 경우 (수학적 손익분기) → 0 % 반환."""
        # kelly = win_rate - (1 - win_rate) / ratio = 0
        # win_rate * ratio = 1 - win_rate  →  win_rate * (ratio + 1) = 1
        # ratio=1, win_rate=0.5: kelly = 0.5 - 0.5 / 1.0 = 0.0
        result = rm.kelly_size(win_rate=0.50, avg_win=0.01, avg_loss=0.01)
        assert result == 0.0

    def test_default_max_pct_cap(self, rm):
        """기본 max_pct(0.25)로 상한이 적용되는지 확인."""
        # kelly = 0.99 - 0.01 / (10.0 / 0.01) = 0.99 - 0.01/1000 ≈ 0.98999
        # result = 0.98999 * 0.25 ≈ 0.2475 → under cap
        # Use safety_factor=1.0 so result = kelly_pct, easily exceeds 0.25
        # kelly = 0.80 - 0.20 / (2.0 / 1.0) = 0.80 - 0.10 = 0.70
        # result = 0.70 * 1.0 = 0.70 → capped at 0.25
        result = rm.kelly_size(
            win_rate=0.80, avg_win=2.0, avg_loss=1.0,
            safety_factor=1.0, max_pct=0.25,
        )
        assert result == pytest.approx(0.25)

    def test_custom_safety_factor(self, rm):
        """safety_factor 를 변경하면 결과에 정비례 반영."""
        # kelly = 0.60 - 0.40 / 2.0 = 0.40
        result_half = rm.kelly_size(
            win_rate=0.60, avg_win=0.02, avg_loss=0.01,
            safety_factor=0.50, max_pct=1.0,
        )
        # 0.40 * 0.50 = 0.20
        assert result_half == pytest.approx(0.20)

    def test_result_never_negative(self, rm):
        """어떤 입력에도 결과는 0 이상이어야 한다."""
        result = rm.kelly_size(win_rate=0.10, avg_win=0.001, avg_loss=0.50)
        assert result >= 0.0


# ── check_correlation ──────────────────────────────────────────────────────────


class TestCheckCorrelation:
    """Task 13.5 — 심볼 간 상관도 필터 테스트."""

    def test_high_correlation_blocked(self, rm):
        """상관도 0.9 > threshold(0.8) → 진입 차단."""
        matrix = {("BTCUSDT", "ETHUSDT"): 0.9}
        allowed, reason = rm.check_correlation(
            "BTCUSDT", ["ETHUSDT"], matrix, threshold=0.8
        )
        assert allowed is False
        assert reason == "correlated_with_ETHUSDT"

    def test_low_correlation_allowed(self, rm):
        """상관도 0.5 <= threshold(0.8) → 진입 허용."""
        matrix = {("BTCUSDT", "SOLUSDT"): 0.5}
        allowed, reason = rm.check_correlation(
            "BTCUSDT", ["SOLUSDT"], matrix, threshold=0.8
        )
        assert allowed is True
        assert reason == ""

    def test_empty_open_positions_allowed(self, rm):
        """열린 포지션 없음 → 항상 허용."""
        allowed, reason = rm.check_correlation("BTCUSDT", [], {}, threshold=0.8)
        assert allowed is True
        assert reason == ""

    def test_same_symbol_blocked(self, rm):
        """같은 심볼이 이미 열려 있으면 상관도 무관하게 차단."""
        matrix: dict[tuple[str, str], float] = {}
        allowed, reason = rm.check_correlation(
            "BTCUSDT", ["BTCUSDT"], matrix, threshold=0.8
        )
        assert allowed is False
        assert reason == "correlated_with_BTCUSDT"

    def test_reverse_key_order_normalised(self, rm):
        """(B,A) 순서로 저장된 상관계수도 (A,B) 조회에서 정상 작동."""
        # 키를 역순으로 저장
        matrix = {("ETHUSDT", "BTCUSDT"): 0.95}
        allowed, reason = rm.check_correlation(
            "BTCUSDT", ["ETHUSDT"], matrix, threshold=0.8
        )
        assert allowed is False
        assert reason == "correlated_with_ETHUSDT"

    def test_threshold_at_exact_value_allowed(self, rm):
        """상관도 == threshold → 차단하지 않음 (strictly greater than)."""
        matrix = {("BTCUSDT", "ETHUSDT"): 0.8}
        allowed, reason = rm.check_correlation(
            "BTCUSDT", ["ETHUSDT"], matrix, threshold=0.8
        )
        assert allowed is True

    def test_multiple_open_symbols_one_correlated(self, rm):
        """여러 심볼 중 하나만 상관도 높아도 차단."""
        matrix = {
            ("BTCUSDT", "SOLUSDT"): 0.3,
            ("BTCUSDT", "ETHUSDT"): 0.92,
        }
        allowed, reason = rm.check_correlation(
            "BTCUSDT", ["SOLUSDT", "ETHUSDT"], matrix, threshold=0.8
        )
        assert allowed is False
        assert "ETHUSDT" in reason

    def test_missing_entry_in_matrix_treated_as_no_correlation(self, rm):
        """상관행렬에 항목 없음 → 상관도 없다고 보고 허용."""
        matrix: dict[tuple[str, str], float] = {}
        allowed, reason = rm.check_correlation(
            "BTCUSDT", ["BNBUSDT"], matrix, threshold=0.8
        )
        assert allowed is True
        assert reason == ""
