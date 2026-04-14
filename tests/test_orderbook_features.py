"""Unit tests for src/strategies/features/orderbook.py (Task 15.3)."""

from __future__ import annotations

import pytest

from src.strategies.features.orderbook import bid_ask_ratio, detect_wall, spread_pct


# ── bid_ask_ratio ──────────────────────────────────────────────────────────


class TestBidAskRatio:
    def test_equal_depth_returns_one(self):
        bids = [[100.0, 5.0], [99.5, 5.0]]
        asks = [[100.5, 5.0], [101.0, 5.0]]
        assert bid_ask_ratio(bids, asks) == pytest.approx(1.0)

    def test_bid_dominated_returns_above_one(self):
        # bids total qty=20, asks total qty=10 → ratio=2.0
        bids = [[100.0, 10.0], [99.0, 10.0]]
        asks = [[101.0, 5.0], [102.0, 5.0]]
        assert bid_ask_ratio(bids, asks) == pytest.approx(2.0)

    def test_ask_dominated_returns_below_one(self):
        # bids total=5, asks total=20 → ratio=0.25
        bids = [[100.0, 5.0]]
        asks = [[101.0, 10.0], [102.0, 10.0]]
        assert bid_ask_ratio(bids, asks) == pytest.approx(0.25)

    def test_empty_bids(self):
        asks = [[101.0, 5.0]]
        assert bid_ask_ratio([], asks) == pytest.approx(0.0)

    def test_empty_asks_returns_one(self):
        """Zero ask total is a guard — returns neutral 1.0."""
        bids = [[100.0, 5.0]]
        assert bid_ask_ratio(bids, []) == pytest.approx(1.0)

    def test_both_empty_returns_one(self):
        assert bid_ask_ratio([], []) == pytest.approx(1.0)

    def test_single_level_each(self):
        bids = [[100.0, 3.0]]
        asks = [[101.0, 6.0]]
        assert bid_ask_ratio(bids, asks) == pytest.approx(0.5)


# ── spread_pct ─────────────────────────────────────────────────────────────


class TestSpreadPct:
    def test_normal_spread(self):
        # mid = 100.25, spread = 0.5 → 0.5/100.25 * 100 ≈ 0.4988 %
        result = spread_pct(100.0, 100.5)
        assert result == pytest.approx((100.5 - 100.0) / 100.25 * 100.0)

    def test_zero_spread(self):
        assert spread_pct(100.0, 100.0) == pytest.approx(0.0)

    def test_zero_bid_guard(self):
        """Should return 0.0 rather than raising ZeroDivisionError."""
        assert spread_pct(0.0, 100.0) == pytest.approx(0.0)

    def test_tight_spread(self):
        result = spread_pct(50000.0, 50001.0)
        # spread = 1, mid = 50000.5 → ~0.002 %
        assert result == pytest.approx(1.0 / 50000.5 * 100.0)

    def test_wide_spread(self):
        result = spread_pct(90.0, 110.0)
        # mid = 100, spread = 20 → 20 %
        assert result == pytest.approx(20.0)


# ── detect_wall ────────────────────────────────────────────────────────────


class TestDetectWall:
    def test_single_large_order_detected(self):
        # mean = (1+1+1+30)/4 = 8.25, threshold = 24.75; 30 > 24.75
        orders = [[100.0, 1.0], [99.0, 1.0], [98.0, 1.0], [97.0, 30.0]]
        walls = detect_wall(orders, threshold_mult=3.0)
        assert len(walls) == 1
        assert walls[0]["price"] == pytest.approx(97.0)
        assert walls[0]["qty"] == pytest.approx(30.0)

    def test_multiple_walls_sorted_by_qty_desc(self):
        orders = [
            [100.0, 1.0],
            [99.0, 50.0],
            [98.0, 1.0],
            [97.0, 30.0],
        ]
        walls = detect_wall(orders, threshold_mult=3.0)
        # mean = (1+50+1+30)/4 = 20.5, cutoff = 61.5 → none qualify with mult=3
        # Let's use mult=1.5: cutoff = 30.75; 50 qualifies, 30 does not
        walls = detect_wall(orders, threshold_mult=1.5)
        assert len(walls) == 1
        assert walls[0]["qty"] == pytest.approx(50.0)

    def test_two_walls_returned_desc(self):
        orders = [
            [100.0, 1.0],
            [99.0, 20.0],
            [98.0, 10.0],
        ]
        # mean = (1+20+10)/3 ≈ 10.33, cutoff with mult=1.5 = 15.5 → 20 qualifies
        walls = detect_wall(orders, threshold_mult=1.5)
        assert len(walls) == 1
        assert walls[0]["qty"] == pytest.approx(20.0)

    def test_no_walls_returns_empty(self):
        # all quantities equal → no order exceeds mean * 3
        orders = [[100.0, 5.0], [99.0, 5.0], [98.0, 5.0], [97.0, 5.0]]
        assert detect_wall(orders) == []

    def test_single_order_returns_empty(self):
        """Need at least 2 levels to compute a meaningful mean."""
        assert detect_wall([[100.0, 100.0]]) == []

    def test_empty_orders_returns_empty(self):
        assert detect_wall([]) == []

    def test_custom_threshold_multiplier(self):
        orders = [[100.0, 1.0], [99.0, 1.0], [98.0, 6.0]]
        # mean = (1+1+6)/3 ≈ 2.67; mult=2.0 → cutoff = 5.33; 6 > 5.33
        walls = detect_wall(orders, threshold_mult=2.0)
        assert len(walls) == 1
        assert walls[0]["price"] == pytest.approx(98.0)

    def test_wall_dict_has_price_and_qty_keys(self):
        orders = [[100.0, 1.0], [99.0, 1.0], [98.0, 10.0]]
        walls = detect_wall(orders, threshold_mult=2.0)
        assert len(walls) >= 1
        assert "price" in walls[0]
        assert "qty" in walls[0]
