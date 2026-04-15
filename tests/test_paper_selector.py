"""paper_selector: 페이퍼 승률 기반 자동 전략 스위칭 테스트."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.core import database as db
from src.core.paper_selector import evaluate_candidates, select_and_apply


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """임시 DB로 격리."""
    test_db = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", test_db)
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    monkeypatch.setattr(db, "engine", create_engine(f"sqlite:///{test_db}", echo=False))
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=db.engine))
    db.init_db()
    yield


def _seed(strategy: str, balance: float, total: int, wins: int, initial: float = 200.0):
    with db.get_session() as s:
        bal = db.PaperBalance(
            strategy=strategy, balance=balance, initial_balance=initial,
            total_trades=total, wins=wins, losses=total - wins,
        )
        s.add(bal)
        s.commit()


def test_no_candidates_pauses_trading():
    # 낮은 승률
    _seed("foo_strat", balance=150, total=20, wins=3)  # WR 15%

    with patch("src.core.paper_selector.list_strategies",
               return_value=[{"name": "foo_strat"}]):
        result = select_and_apply()

    assert result["action"] == "paused"
    assert db.get_setting("trading_paused") == "true"


def test_best_strategy_selected_when_eligible():
    _seed("loser", balance=100, total=30, wins=5, initial=200)      # WR 16.7%
    _seed("winner", balance=240, total=20, wins=13, initial=200)    # WR 65%, +20%
    _seed("ok", balance=220, total=15, wins=9, initial=200)         # WR 60%, +10%

    with patch("src.core.paper_selector.list_strategies",
               return_value=[{"name": "loser"}, {"name": "winner"}, {"name": "ok"}]):
        result = select_and_apply()

    assert result["action"] == "switched"
    assert result["strategy"] == "winner"
    assert db.get_setting("strategy") == "winner"
    assert db.get_setting("trading_paused") == "false"


def test_unregistered_strategies_excluded():
    _seed("legacy_deleted", balance=500, total=50, wins=40)  # 후보 될 듯하지만 레지스트리에 없음
    _seed("active_ok", balance=220, total=15, wins=9, initial=200)

    with patch("src.core.paper_selector.list_strategies",
               return_value=[{"name": "active_ok"}]):
        result = select_and_apply()

    assert result["strategy"] == "active_ok"


def test_min_trades_gate():
    _seed("too_few", balance=240, total=5, wins=5)  # 100% WR, but only 5 trades

    with patch("src.core.paper_selector.list_strategies",
               return_value=[{"name": "too_few"}]):
        eligible, th = evaluate_candidates()
    assert th["min_trades"] == 10
    assert eligible == []


def test_resume_from_pause_when_winner_emerges():
    # 먼저 정지 상태
    db.set_setting("trading_paused", "true")
    db.set_setting("trading_paused_reason", "no_eligible_strategy")

    _seed("winner", balance=250, total=20, wins=14, initial=200)  # WR 70%

    with patch("src.core.paper_selector.list_strategies",
               return_value=[{"name": "winner"}]):
        result = select_and_apply()

    assert result["action"] == "switched"
    assert db.get_setting("trading_paused") == "false"


def test_selector_disabled_short_circuits():
    db.set_setting("paper_selector_enabled", "false")
    result = select_and_apply()
    assert result["action"] == "disabled"
