"""역신호 래퍼: BUY↔SELL 뒤집기, 파라미터 상속, CLOSE/HOLD 보존 검증."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.registry import _REGISTRY, get_strategy, list_strategies
from src.strategies.reverse_wrapper import make_reverse


@dataclass
class _StubState:
    entry_atr: float = 0.0
    total_trades: int = 0


class _StubInner(Strategy):
    """테스트용 이너 전략: 고정 시그널 타입을 돌려준다."""

    LEVERAGE = 9
    POSITION_SIZE_PCT = 0.33
    SL_ATR_MULT = 1.7
    TP_ATR_MULT = 5.5
    TIMEFRAMES = ["1m", "5m"]

    _mode = ExecutionMode.SIGNAL_ONLY
    _return_type: SignalType = SignalType.BUY

    def __init__(self):
        self.state = _StubState(entry_atr=12.34, total_trades=5)
        self.recorded: list[float] = []

    @property
    def name(self) -> str:
        return "_stub_inner"

    @property
    def label(self) -> str:
        return "Stub Inner"

    @property
    def description(self) -> str:
        return "stub for tests"

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    def evaluate(self, symbol, candles, htf_candles=None):
        return Signal(
            symbol=symbol, type=self._return_type,
            confidence=0.77, source=self.name,
            metadata={"original_thing": "kept"},
        )

    def record_result(self, pnl: float) -> None:
        self.recorded.append(pnl)


@pytest.fixture
def reverse_cls():
    # 등록 후 테스트 종료 시 정리
    from src.strategies.registry import register
    register(_StubInner)
    cls = make_reverse("_stub_inner", "_stub_reverse")
    yield cls
    _REGISTRY.pop("_stub_inner", None)
    _REGISTRY.pop("_stub_reverse", None)


def test_buy_becomes_sell(reverse_cls):
    rev = reverse_cls()
    rev._inner._return_type = SignalType.BUY
    sig = rev.evaluate("BTCUSDT", pd.DataFrame())
    assert sig.type == SignalType.SELL
    assert sig.confidence == 0.77
    assert sig.source == "_stub_reverse"
    assert sig.metadata["reversed_from"] == "_stub_inner"
    assert sig.metadata["original_type"] == "buy"
    assert sig.metadata["original_thing"] == "kept"


def test_sell_becomes_buy(reverse_cls):
    rev = reverse_cls()
    rev._inner._return_type = SignalType.SELL
    assert rev.evaluate("X", pd.DataFrame()).type == SignalType.BUY


def test_hold_stays_hold(reverse_cls):
    rev = reverse_cls()
    rev._inner._return_type = SignalType.HOLD
    assert rev.evaluate("X", pd.DataFrame()).type == SignalType.HOLD


def test_close_stays_close(reverse_cls):
    """CLOSE는 방향 개념 없음 — 원본 청산 의도 그대로 전달되어야 함."""
    rev = reverse_cls()
    rev._inner._return_type = SignalType.CLOSE
    assert rev.evaluate("X", pd.DataFrame()).type == SignalType.CLOSE


def test_trading_params_inherited(reverse_cls):
    rev = reverse_cls()
    assert rev.LEVERAGE == 9
    assert rev.POSITION_SIZE_PCT == 0.33
    assert rev.SL_ATR_MULT == 1.7
    assert rev.TP_ATR_MULT == 5.5
    assert rev.TIMEFRAMES == ["1m", "5m"]


def test_mode_inherited(reverse_cls):
    rev = reverse_cls()
    assert rev.mode == ExecutionMode.SIGNAL_ONLY


def test_state_proxies_to_inner(reverse_cls):
    rev = reverse_cls()
    # state는 이너의 state 그대로 — futures_engine이 entry_atr 참조 필요
    assert rev.state is rev._inner.state
    assert rev.state.entry_atr == 12.34


def test_record_result_forwards_to_inner(reverse_cls):
    rev = reverse_cls()
    rev.record_result(1.23)
    rev.record_result(-0.5)
    assert rev._inner.recorded == [1.23, -0.5]


def test_reverse_variants_auto_registered():
    """reverse_wrapper 모듈 import 시점에 4개 변형이 등록되어야 한다."""
    names = {s["name"] for s in list_strategies()}
    for required in [
        "reverse_momentum_flip_scalper",
        "reverse_orderflow_v13",
        "reverse_pattern_scalper",
        "reverse_data_driven_scalper",
    ]:
        assert required in names, f"missing: {required}"


def test_unknown_inner_raises():
    with pytest.raises(ValueError, match="unknown inner strategy"):
        make_reverse("this_does_not_exist")
