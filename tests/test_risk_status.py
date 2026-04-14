"""Unit tests for get_risk_status() — Task 13.7.

Uses an in-memory SQLite DB so no external services are needed.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest

from src.core.database import get_risk_status


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_trade(pnl: float, pnl_pct: float, strategy: str = "v12") -> MagicMock:
    t = MagicMock()
    t.pnl = pnl
    t.pnl_pct = pnl_pct
    t.strategy = strategy
    return t


# ── Return structure ──────────────────────────────────────────────────────────


class TestGetRiskStatusStructure:
    """get_risk_status() must return all required keys with correct types."""

    def test_returns_all_keys(self):
        with (
            patch("src.core.database.get_today_pnl", return_value=(0.0, 0)),
            patch("src.core.database.get_open_positions", return_value=[]),
            patch("src.core.database.get_trades", return_value=[]),
        ):
            result = get_risk_status()

        assert isinstance(result, dict)
        for key in (
            "daily_pnl",
            "daily_pnl_pct",
            "daily_dd_ok",
            "open_positions",
            "max_positions",
            "can_open",
            "kelly_sizes",
        ):
            assert key in result, f"Missing key: {key}"

    def test_daily_pnl_is_float(self):
        with (
            patch("src.core.database.get_today_pnl", return_value=(10.5, 1)),
            patch("src.core.database.get_open_positions", return_value=[]),
            patch("src.core.database.get_trades", return_value=[]),
        ):
            result = get_risk_status(balance=1000.0)
        assert isinstance(result["daily_pnl"], float)

    def test_open_positions_is_int(self):
        with (
            patch("src.core.database.get_today_pnl", return_value=(0.0, 0)),
            patch("src.core.database.get_open_positions", return_value=[MagicMock()]),
            patch("src.core.database.get_trades", return_value=[]),
        ):
            result = get_risk_status()
        assert isinstance(result["open_positions"], int)
        assert result["open_positions"] == 1

    def test_kelly_sizes_is_dict(self):
        with (
            patch("src.core.database.get_today_pnl", return_value=(0.0, 0)),
            patch("src.core.database.get_open_positions", return_value=[]),
            patch("src.core.database.get_trades", return_value=[]),
        ):
            result = get_risk_status()
        assert isinstance(result["kelly_sizes"], dict)

    def test_can_open_is_bool(self):
        with (
            patch("src.core.database.get_today_pnl", return_value=(0.0, 0)),
            patch("src.core.database.get_open_positions", return_value=[]),
            patch("src.core.database.get_trades", return_value=[]),
        ):
            result = get_risk_status()
        assert isinstance(result["can_open"], bool)

    def test_daily_dd_ok_is_bool(self):
        with (
            patch("src.core.database.get_today_pnl", return_value=(0.0, 0)),
            patch("src.core.database.get_open_positions", return_value=[]),
            patch("src.core.database.get_trades", return_value=[]),
        ):
            result = get_risk_status()
        assert isinstance(result["daily_dd_ok"], bool)


# ── DD / can_open logic ───────────────────────────────────────────────────────


class TestRiskStatusLogic:
    def test_no_loss_can_open_true(self):
        with (
            patch("src.core.database.get_today_pnl", return_value=(0.0, 0)),
            patch("src.core.database.get_open_positions", return_value=[]),
            patch("src.core.database.get_trades", return_value=[]),
        ):
            result = get_risk_status(balance=1000.0, max_open_positions=3)
        assert result["can_open"] is True
        assert result["daily_dd_ok"] is True

    def test_dd_exceeded_blocks_entry(self):
        # -60 USDT on 1000 balance = -6% > 5% limit
        with (
            patch("src.core.database.get_today_pnl", return_value=(-60.0, 3)),
            patch("src.core.database.get_open_positions", return_value=[]),
            patch("src.core.database.get_trades", return_value=[]),
        ):
            result = get_risk_status(
                balance=1000.0,
                max_open_positions=3,
                max_daily_loss_pct=0.05,
            )
        assert result["daily_dd_ok"] is False
        assert result["can_open"] is False

    def test_max_positions_reached_blocks_entry(self):
        mock_positions = [MagicMock(), MagicMock(), MagicMock()]
        with (
            patch("src.core.database.get_today_pnl", return_value=(0.0, 0)),
            patch("src.core.database.get_open_positions", return_value=mock_positions),
            patch("src.core.database.get_trades", return_value=[]),
        ):
            result = get_risk_status(balance=1000.0, max_open_positions=3)
        assert result["open_positions"] == 3
        assert result["can_open"] is False

    def test_max_positions_config_respected(self):
        with (
            patch("src.core.database.get_today_pnl", return_value=(0.0, 0)),
            patch("src.core.database.get_open_positions", return_value=[MagicMock()]),
            patch("src.core.database.get_trades", return_value=[]),
        ):
            result = get_risk_status(balance=1000.0, max_open_positions=5)
        assert result["max_positions"] == 5
        assert result["can_open"] is True  # 1 < 5

    def test_daily_pnl_pct_zero_when_no_balance(self):
        with (
            patch("src.core.database.get_today_pnl", return_value=(50.0, 1)),
            patch("src.core.database.get_open_positions", return_value=[]),
            patch("src.core.database.get_trades", return_value=[]),
        ):
            result = get_risk_status(balance=0.0)
        assert result["daily_pnl_pct"] == 0.0

    def test_daily_pnl_reflected(self):
        with (
            patch("src.core.database.get_today_pnl", return_value=(25.0, 2)),
            patch("src.core.database.get_open_positions", return_value=[]),
            patch("src.core.database.get_trades", return_value=[]),
        ):
            result = get_risk_status(balance=1000.0)
        assert result["daily_pnl"] == pytest.approx(25.0)
        assert result["daily_pnl_pct"] == pytest.approx(0.025)


# ── Kelly sizes ───────────────────────────────────────────────────────────────


class TestKellySizes:
    def test_no_trades_returns_empty_dict(self):
        with (
            patch("src.core.database.get_today_pnl", return_value=(0.0, 0)),
            patch("src.core.database.get_open_positions", return_value=[]),
            patch("src.core.database.get_trades", return_value=[]),
        ):
            result = get_risk_status()
        assert result["kelly_sizes"] == {}

    def test_single_strategy_kelly_computed(self):
        trades = [
            _make_trade(20.0, 2.0, "v12"),   # win
            _make_trade(15.0, 1.5, "v12"),   # win
            _make_trade(-10.0, -1.0, "v12"), # loss
            _make_trade(-8.0, -0.8, "v12"),  # loss
        ]
        with (
            patch("src.core.database.get_today_pnl", return_value=(0.0, 0)),
            patch("src.core.database.get_open_positions", return_value=[]),
            patch("src.core.database.get_trades", return_value=trades),
        ):
            result = get_risk_status()
        assert "v12" in result["kelly_sizes"]
        assert isinstance(result["kelly_sizes"]["v12"], float)
        assert 0.0 <= result["kelly_sizes"]["v12"] <= 0.25

    def test_multiple_strategies_kelly_computed(self):
        trades = [
            _make_trade(20.0, 2.0, "v12"),
            _make_trade(-10.0, -1.0, "v12"),
            _make_trade(30.0, 3.0, "v1"),
            _make_trade(-5.0, -0.5, "v1"),
        ]
        with (
            patch("src.core.database.get_today_pnl", return_value=(0.0, 0)),
            patch("src.core.database.get_open_positions", return_value=[]),
            patch("src.core.database.get_trades", return_value=trades),
        ):
            result = get_risk_status()
        assert "v12" in result["kelly_sizes"]
        assert "v1" in result["kelly_sizes"]

    def test_all_losses_kelly_is_zero(self):
        trades = [_make_trade(-10.0, -1.0, "v12") for _ in range(5)]
        with (
            patch("src.core.database.get_today_pnl", return_value=(0.0, 0)),
            patch("src.core.database.get_open_positions", return_value=[]),
            patch("src.core.database.get_trades", return_value=trades),
        ):
            result = get_risk_status()
        assert result["kelly_sizes"].get("v12", 0.0) == pytest.approx(0.0)

    def test_none_strategy_grouped_as_unknown(self):
        trades = [
            _make_trade(10.0, 1.0, None),
            _make_trade(-5.0, -0.5, None),
        ]
        with (
            patch("src.core.database.get_today_pnl", return_value=(0.0, 0)),
            patch("src.core.database.get_open_positions", return_value=[]),
            patch("src.core.database.get_trades", return_value=trades),
        ):
            result = get_risk_status()
        assert "unknown" in result["kelly_sizes"]
