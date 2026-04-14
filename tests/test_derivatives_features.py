"""Unit tests for src/strategies/features/derivatives.py (Task 15.4)."""

from __future__ import annotations

import pytest

from src.strategies.features.derivatives import (
    funding_rate_signal,
    oi_change_rate,
    oi_divergence_signal,
)


# ── funding_rate_signal ────────────────────────────────────────────────────


class TestFundingRateSignal:
    def test_extreme_positive_returns_short(self):
        # Latest rate well above threshold → longs paying shorts → SHORT preferred
        rates = [0.0003, 0.0005, 0.0015]
        assert funding_rate_signal(rates, extreme_threshold=0.001) == "SHORT"

    def test_extreme_negative_returns_long(self):
        # Latest rate well below -threshold → shorts paying longs → LONG preferred
        rates = [-0.0003, -0.0005, -0.0015]
        assert funding_rate_signal(rates, extreme_threshold=0.001) == "LONG"

    def test_neutral_positive_returns_neutral(self):
        rates = [0.0001, 0.0002, 0.0005]
        assert funding_rate_signal(rates, extreme_threshold=0.001) == "NEUTRAL"

    def test_neutral_negative_returns_neutral(self):
        rates = [-0.0001, -0.0003, -0.0005]
        assert funding_rate_signal(rates, extreme_threshold=0.001) == "NEUTRAL"

    def test_zero_returns_neutral(self):
        assert funding_rate_signal([0.0, 0.0, 0.0]) == "NEUTRAL"

    def test_empty_list_returns_neutral(self):
        assert funding_rate_signal([]) == "NEUTRAL"

    def test_single_extreme_positive(self):
        assert funding_rate_signal([0.005]) == "SHORT"

    def test_single_extreme_negative(self):
        assert funding_rate_signal([-0.005]) == "LONG"

    def test_only_latest_value_matters(self):
        # History is extreme positive but latest is neutral
        rates = [0.002, 0.003, 0.0005]
        assert funding_rate_signal(rates, extreme_threshold=0.001) == "NEUTRAL"

    def test_custom_threshold(self):
        rates = [0.05]
        assert funding_rate_signal(rates, extreme_threshold=0.1) == "NEUTRAL"
        assert funding_rate_signal(rates, extreme_threshold=0.01) == "SHORT"


# ── oi_change_rate ─────────────────────────────────────────────────────────


class TestOiChangeRate:
    def test_increase(self):
        # 1000 → 1100 = +10%
        assert oi_change_rate(1100.0, 1000.0) == pytest.approx(10.0)

    def test_decrease(self):
        # 1000 → 900 = -10%
        assert oi_change_rate(900.0, 1000.0) == pytest.approx(-10.0)

    def test_no_change(self):
        assert oi_change_rate(500.0, 500.0) == pytest.approx(0.0)

    def test_zero_prev_oi_guard(self):
        """Should return 0.0 rather than raising ZeroDivisionError."""
        assert oi_change_rate(500.0, 0.0) == pytest.approx(0.0)

    def test_large_increase(self):
        assert oi_change_rate(200.0, 100.0) == pytest.approx(100.0)

    def test_fractional_change(self):
        result = oi_change_rate(100.5, 100.0)
        assert result == pytest.approx(0.5)


# ── oi_divergence_signal ───────────────────────────────────────────────────


class TestOiDivergenceSignal:
    def test_price_up_oi_down_bearish_divergence(self):
        """Rising price with falling OI: trend weakening."""
        assert oi_divergence_signal(2.0, -1.0) == "BEARISH_DIV"

    def test_price_down_oi_up_bullish_divergence(self):
        """Falling price with rising OI: new shorts may be over-extended."""
        assert oi_divergence_signal(-2.0, 1.0) == "BULLISH_DIV"

    def test_price_up_oi_up_bullish_confirm(self):
        """Price and OI both up: trend confirmed by new money."""
        assert oi_divergence_signal(2.0, 3.0) == "BULLISH_CONFIRM"

    def test_price_down_oi_down_bearish_confirm(self):
        """Price and OI both down: shorts covering, potential exhaustion."""
        assert oi_divergence_signal(-2.0, -1.0) == "BEARISH_CONFIRM"

    def test_zero_price_change_oi_up(self):
        """Zero price change treated as 'not up', so not bearish-div side."""
        result = oi_divergence_signal(0.0, 5.0)
        assert result == "BULLISH_DIV"

    def test_zero_price_change_oi_down(self):
        result = oi_divergence_signal(0.0, -5.0)
        assert result == "BEARISH_CONFIRM"

    def test_zero_oi_change_price_up(self):
        result = oi_divergence_signal(2.0, 0.0)
        assert result == "BEARISH_DIV"

    def test_zero_oi_change_price_down(self):
        result = oi_divergence_signal(-2.0, 0.0)
        assert result == "BEARISH_CONFIRM"

    def test_both_zero(self):
        result = oi_divergence_signal(0.0, 0.0)
        assert result == "BEARISH_CONFIRM"
