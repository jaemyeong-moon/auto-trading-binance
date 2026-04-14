"""Unit tests for FuturesEngine tick loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.futures_engine import FuturesEngine
from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy


# ── Test Strategy Stubs ────────────────────────────────────


class _AlwaysFlipStrategy(Strategy):
    """Minimal ALWAYS_FLIP strategy for tests."""

    LEVERAGE = 5
    POSITION_SIZE_PCT = 0.20
    SL_ATR_MULT = 2.0
    TP_ATR_MULT = 4.0

    def __init__(self, signal_type: SignalType = SignalType.BUY):
        self._signal_type = signal_type

    @property
    def name(self) -> str:
        return "always_flip_stub"

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.ALWAYS_FLIP

    def evaluate(self, symbol, candles, htf_candles=None) -> Signal:
        return Signal(
            symbol=symbol,
            type=self._signal_type,
            confidence=0.8,
            source=self.name,
        )


class _SignalOnlyStrategy(Strategy):
    """Minimal SIGNAL_ONLY strategy for tests."""

    LEVERAGE = 5
    POSITION_SIZE_PCT = 0.20
    SL_PCT = 0.005
    FULL_TP_PCT = 0.012
    PARTIAL_TP_PCT = 0.006
    TRAILING_ACTIVATE_PCT = 0.008
    TRAILING_DISTANCE_PCT = 0.003

    def __init__(self, signal_type: SignalType = SignalType.BUY):
        self._signal_type = signal_type

    @property
    def name(self) -> str:
        return "signal_only_stub"

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.SIGNAL_ONLY

    def evaluate(self, symbol, candles, htf_candles=None) -> Signal:
        return Signal(
            symbol=symbol,
            type=self._signal_type,
            confidence=0.9,
            source=self.name,
        )


# ── Helpers ───────────────────────────────────────────────


def _make_engine_with_strategy(
    mock_futures_client: AsyncMock,
    strategy: Strategy,
    symbol: str = "BTCUSDT",
) -> FuturesEngine:
    """Create FuturesEngine with pre-loaded strategy (no DB calls)."""
    engine = FuturesEngine(mock_futures_client)
    engine.strategies[symbol] = strategy
    return engine


# ── _tick_signal_only Tests ───────────────────────────────


@pytest.mark.asyncio
async def test_tick_signal_only_buy_opens_long(mock_futures_client):
    """BUY 신호 + 포지션 없음 → open_long 호출."""
    strategy = _SignalOnlyStrategy(SignalType.BUY)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_position.return_value = None

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "open_position": MagicMock(),
        "get_open_positions": MagicMock(return_value=[]),
        "get_today_pnl": MagicMock(return_value=(0.0, 0)),
        "get_settings_hash": MagicMock(return_value="abc"),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_signal_only("BTCUSDT", strategy)

    mock_futures_client.open_long.assert_called_once()


@pytest.mark.asyncio
async def test_tick_signal_only_sell_opens_short(mock_futures_client):
    """SELL 신호 + 포지션 없음 → open_short 호출."""
    strategy = _SignalOnlyStrategy(SignalType.SELL)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_position.return_value = None

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "open_position": MagicMock(),
        "get_open_positions": MagicMock(return_value=[]),
        "get_today_pnl": MagicMock(return_value=(0.0, 0)),
        "get_settings_hash": MagicMock(return_value="abc"),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_signal_only("BTCUSDT", strategy)

    mock_futures_client.open_short.assert_called_once()


@pytest.mark.asyncio
async def test_tick_signal_only_hold_no_order(mock_futures_client):
    """HOLD 신호 → 주문 없음."""
    strategy = _SignalOnlyStrategy(SignalType.HOLD)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_position.return_value = None

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "get_settings_hash": MagicMock(return_value="abc"),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_signal_only("BTCUSDT", strategy)

    mock_futures_client.open_long.assert_not_called()
    mock_futures_client.open_short.assert_not_called()


@pytest.mark.asyncio
async def test_tick_signal_only_low_confidence_skips_entry(mock_futures_client):
    """confidence < 0.6 이면 진입 차단."""
    strategy = _SignalOnlyStrategy(SignalType.BUY)
    # Override evaluate to return low confidence
    strategy.evaluate = lambda sym, c, htf=None: Signal(
        symbol=sym, type=SignalType.BUY, confidence=0.5, source="test"
    )
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_position.return_value = None

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "get_settings_hash": MagicMock(return_value="abc"),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_signal_only("BTCUSDT", strategy)

    mock_futures_client.open_long.assert_not_called()


@pytest.mark.asyncio
async def test_tick_signal_only_close_signal_closes_position(mock_futures_client):
    """CLOSE 신호 + 포지션 있음 → 청산."""
    strategy = _SignalOnlyStrategy(SignalType.CLOSE)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_position.return_value = {
        "side": "LONG",
        "entry_price": 100.0,
        "quantity": 0.01,
    }

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "close_position": MagicMock(return_value=None),
        "delete_position": MagicMock(),
        "record_trail": MagicMock(),
        "get_position": MagicMock(return_value=None),
        "link_trails_to_trade": MagicMock(),
        "get_settings_hash": MagicMock(return_value="abc"),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_signal_only("BTCUSDT", strategy)

    mock_futures_client.close_long.assert_called_once()


# ── _tick_always_flip Tests ───────────────────────────────


@pytest.mark.asyncio
async def test_tick_always_flip_buy_no_position_opens_long(mock_futures_client):
    """BUY 신호 + 포지션 없음 → open_long 호출."""
    strategy = _AlwaysFlipStrategy(SignalType.BUY)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_position.return_value = None

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "open_position": MagicMock(),
        "get_open_positions": MagicMock(return_value=[]),
        "get_today_pnl": MagicMock(return_value=(0.0, 0)),
        "get_position": MagicMock(return_value=None),
        "record_trail": MagicMock(),
        "get_settings_hash": MagicMock(return_value="abc"),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_always_flip("BTCUSDT", strategy)

    mock_futures_client.open_long.assert_called_once()


@pytest.mark.asyncio
async def test_tick_always_flip_sell_no_position_opens_short(mock_futures_client):
    """SELL 신호 + 포지션 없음 → open_short 호출."""
    strategy = _AlwaysFlipStrategy(SignalType.SELL)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_position.return_value = None

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "open_position": MagicMock(),
        "get_open_positions": MagicMock(return_value=[]),
        "get_today_pnl": MagicMock(return_value=(0.0, 0)),
        "get_position": MagicMock(return_value=None),
        "record_trail": MagicMock(),
        "get_settings_hash": MagicMock(return_value="abc"),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_always_flip("BTCUSDT", strategy)

    mock_futures_client.open_short.assert_called_once()


@pytest.mark.asyncio
async def test_tick_always_flip_hold_no_order(mock_futures_client):
    """HOLD 신호 → 포지션 없을 때 주문 없음."""
    strategy = _AlwaysFlipStrategy(SignalType.HOLD)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_position.return_value = None

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "get_position": MagicMock(return_value=None),
        "get_settings_hash": MagicMock(return_value="abc"),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_always_flip("BTCUSDT", strategy)

    mock_futures_client.open_long.assert_not_called()
    mock_futures_client.open_short.assert_not_called()


@pytest.mark.asyncio
async def test_tick_always_flip_existing_position_flips_on_opposite_signal(
    mock_futures_client,
):
    """반대 방향 신호 + LONG 포지션 → 청산 후 SHORT 진입 (플립)."""
    strategy = _AlwaysFlipStrategy(SignalType.SELL)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    # 현재 LONG 포지션
    mock_futures_client.get_position.side_effect = [
        # _tick_always_flip 내부 첫 get_position (플립 분기)
        {"side": "LONG", "entry_price": 100.0, "quantity": 0.01},
        # _close_current 내부 get_position
        {"side": "LONG", "entry_price": 100.0, "quantity": 0.01},
        # _place_exchange_sl_tp 내부 get_position
        {"side": "SHORT", "entry_price": 100.0, "quantity": 0.01},
    ]

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "open_position": MagicMock(),
        "close_position": MagicMock(return_value=None),
        "delete_position": MagicMock(),
        "get_open_positions": MagicMock(return_value=[]),
        "get_today_pnl": MagicMock(return_value=(0.0, 0)),
        "get_position": MagicMock(return_value=None),
        "record_trail": MagicMock(),
        "link_trails_to_trade": MagicMock(),
        "get_settings_hash": MagicMock(return_value="abc"),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_always_flip("BTCUSDT", strategy)

    mock_futures_client.close_long.assert_called_once()
    mock_futures_client.open_short.assert_called_once()


# ── Error / Edge Case Tests ────────────────────────────────


@pytest.mark.asyncio
async def test_tick_signal_only_empty_candles_early_return(mock_futures_client):
    """캔들 데이터가 비어 있으면 조기 반환 (주문 없음)."""
    import pandas as pd

    strategy = _SignalOnlyStrategy(SignalType.BUY)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_candles.return_value = pd.DataFrame()

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "get_settings_hash": MagicMock(return_value="abc"),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_signal_only("BTCUSDT", strategy)

    mock_futures_client.open_long.assert_not_called()
    mock_futures_client.open_short.assert_not_called()


@pytest.mark.asyncio
async def test_tick_always_flip_empty_candles_early_return(mock_futures_client):
    """캔들 데이터가 비어 있으면 _tick_always_flip도 조기 반환."""
    import pandas as pd

    strategy = _AlwaysFlipStrategy(SignalType.BUY)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_candles.return_value = pd.DataFrame()

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_settings_hash": MagicMock(return_value="abc"),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_always_flip("BTCUSDT", strategy)

    mock_futures_client.open_long.assert_not_called()


@pytest.mark.asyncio
async def test_tick_signal_only_order_exception_does_not_propagate(
    mock_futures_client,
):
    """open_long が例外を投げても _tick_signal_only がクラッシュしない."""
    strategy = _SignalOnlyStrategy(SignalType.BUY)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_position.return_value = None
    mock_futures_client.open_long.side_effect = Exception("Exchange error -2019")

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "open_position": MagicMock(),
        "get_open_positions": MagicMock(return_value=[]),
        "get_today_pnl": MagicMock(return_value=(0.0, 0)),
        "get_settings_hash": MagicMock(return_value="abc"),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        # Should not raise
        await engine._tick_signal_only("BTCUSDT", strategy)

    # Exception was raised but caught inside _open_position
    mock_futures_client.open_long.assert_called_once()


# ── Leverage DB/Strategy Sync Regression Tests ────────────


class _HighLeverageStrategy(Strategy):
    """Strategy stub with LEVERAGE=7 (e.g. pattern_scalper)."""

    LEVERAGE = 7
    POSITION_SIZE_PCT = 0.20
    SL_ATR_MULT = 2.0
    TP_ATR_MULT = 4.0

    @property
    def name(self) -> str:
        return "high_leverage_stub"

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.ALWAYS_FLIP

    def evaluate(self, symbol, candles, htf_candles=None) -> Signal:
        return Signal(symbol=symbol, type=SignalType.HOLD, confidence=0.5, source=self.name)


class _NoLeverageAttrStrategy(Strategy):
    """Strategy stub without LEVERAGE attribute — relies on base default (5)."""

    POSITION_SIZE_PCT = 0.20
    SL_ATR_MULT = 2.0
    TP_ATR_MULT = 4.0

    @property
    def name(self) -> str:
        return "no_leverage_attr_stub"

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.ALWAYS_FLIP

    def evaluate(self, symbol, candles, htf_candles=None) -> Signal:
        return Signal(symbol=symbol, type=SignalType.HOLD, confidence=0.5, source=self.name)


@pytest.mark.asyncio
async def test_start_symbol_uses_strategy_leverage_7(mock_futures_client):
    """strategy.LEVERAGE=7 → set_leverage(symbol, 7) が呼ばれること."""
    strategy = _HighLeverageStrategy()
    engine = FuturesEngine(mock_futures_client)

    mock_futures_client.get_position.return_value = None

    db_patch = {
        "get_setting": MagicMock(return_value="high_leverage_stub"),
        "get_settings_hash": MagicMock(return_value="abc"),
        "set_bot_running": MagicMock(),
        "is_bot_running": MagicMock(return_value=False),
        "load_strategy_state": MagicMock(return_value=None),
        "get_position": MagicMock(return_value=None),
        "delete_position": MagicMock(),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch), \
         patch("src.core.futures_engine.get_strategy", return_value=strategy):
        await engine.start_symbol("BTCUSDT")
        # Cancel the background tasks so the test doesn't hang
        for task in list(engine._tasks.values()):
            task.cancel()
        if engine._paper_task:
            engine._paper_task.cancel()

    mock_futures_client.set_leverage.assert_called_once_with("BTCUSDT", 7)


@pytest.mark.asyncio
async def test_start_symbol_uses_default_leverage_5_when_no_attr(mock_futures_client):
    """LEVERAGE 속성 없는 전략 → getattr 기본값 5 → set_leverage(symbol, 5) 호출."""
    strategy = _NoLeverageAttrStrategy()
    # Confirm the attribute truly falls back to Strategy ABC's LEVERAGE=5
    assert getattr(strategy, "LEVERAGE", 5) == 5

    engine = FuturesEngine(mock_futures_client)

    mock_futures_client.get_position.return_value = None

    db_patch = {
        "get_setting": MagicMock(return_value="no_leverage_attr_stub"),
        "get_settings_hash": MagicMock(return_value="abc"),
        "set_bot_running": MagicMock(),
        "is_bot_running": MagicMock(return_value=False),
        "load_strategy_state": MagicMock(return_value=None),
        "get_position": MagicMock(return_value=None),
        "delete_position": MagicMock(),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch), \
         patch("src.core.futures_engine.get_strategy", return_value=strategy):
        await engine.start_symbol("BTCUSDT")
        for task in list(engine._tasks.values()):
            task.cancel()
        if engine._paper_task:
            engine._paper_task.cancel()

    mock_futures_client.set_leverage.assert_called_once_with("BTCUSDT", 5)


@pytest.mark.asyncio
async def test_start_symbol_leverage_value_passed_exactly(mock_futures_client):
    """레버리지 값이 바이낸스 호출에 정확히 전달되는지 — 값 변형 없음 검증."""
    strategy = _HighLeverageStrategy()
    engine = FuturesEngine(mock_futures_client)

    mock_futures_client.get_position.return_value = None

    db_patch = {
        "get_setting": MagicMock(return_value="high_leverage_stub"),
        "get_settings_hash": MagicMock(return_value="abc"),
        "set_bot_running": MagicMock(),
        "is_bot_running": MagicMock(return_value=False),
        "load_strategy_state": MagicMock(return_value=None),
        "get_position": MagicMock(return_value=None),
        "delete_position": MagicMock(),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch), \
         patch("src.core.futures_engine.get_strategy", return_value=strategy):
        await engine.start_symbol("ETHUSDT")
        for task in list(engine._tasks.values()):
            task.cancel()
        if engine._paper_task:
            engine._paper_task.cancel()

    call_args = mock_futures_client.set_leverage.call_args
    called_symbol, called_leverage = call_args.args
    assert called_symbol == "ETHUSDT", f"Expected ETHUSDT, got {called_symbol}"
    assert called_leverage == 7, f"Expected leverage 7, got {called_leverage}"
    assert isinstance(called_leverage, int), "Leverage must be passed as int, not float or str"


# ── RiskManager: max_open_positions Tests ─────────────────


@pytest.mark.asyncio
async def test_open_position_blocked_when_max_positions_exceeded(mock_futures_client):
    """열린 포지션 >= max_open_positions(3) → 진입 차단, open_long/short 미호출."""
    strategy = _SignalOnlyStrategy(SignalType.BUY)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    from src.core.risk_manager import RiskManager
    engine._risk_manager = RiskManager(max_open_positions=3)

    # 이미 3개 포지션이 열려 있음
    mock_positions = [MagicMock(), MagicMock(), MagicMock()]

    db_patch = {
        "get_open_positions": MagicMock(return_value=mock_positions),
        "get_today_pnl": MagicMock(return_value=(0.0, 0)),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._open_position("BTCUSDT", "LONG", 50000.0)

    mock_futures_client.open_long.assert_not_called()
    mock_futures_client.open_short.assert_not_called()


@pytest.mark.asyncio
async def test_open_position_allowed_when_below_max_positions(mock_futures_client):
    """열린 포지션 < max_open_positions(3) → 진입 허용, open_long 호출."""
    strategy = _SignalOnlyStrategy(SignalType.BUY)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    from src.core.risk_manager import RiskManager
    engine._risk_manager = RiskManager(max_open_positions=3)

    # 2개 포지션만 열려 있음 (3 미만)
    mock_positions = [MagicMock(), MagicMock()]

    db_patch = {
        "get_open_positions": MagicMock(return_value=mock_positions),
        "get_today_pnl": MagicMock(return_value=(0.0, 0)),
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "open_position": MagicMock(),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._open_position("BTCUSDT", "LONG", 50000.0)

    mock_futures_client.open_long.assert_called_once()


# ── Daily Drawdown Limit Tests ─────────────────────────────


@pytest.mark.asyncio
async def test_open_position_blocked_when_daily_loss_exceeds_limit(mock_futures_client):
    """일일 손실 -5% 초과(-60 USDT / 잔고 1000) → 진입 차단, open_long 미호출."""
    from src.core.risk_manager import RiskManager

    strategy = _SignalOnlyStrategy(SignalType.BUY)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)
    engine._risk_manager = RiskManager(max_open_positions=3, max_daily_loss_pct=0.05)

    # 잔고 1000, 손실 -60 → -6% (한도 5% 초과)
    mock_futures_client.get_balance.return_value = 1000.0

    db_patch = {
        "get_open_positions": MagicMock(return_value=[]),
        "get_today_pnl": MagicMock(return_value=(-60.0, 3)),
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._open_position("BTCUSDT", "LONG", 50000.0)

    mock_futures_client.open_long.assert_not_called()
    mock_futures_client.open_short.assert_not_called()


@pytest.mark.asyncio
async def test_open_position_allowed_when_daily_loss_within_limit(mock_futures_client):
    """일일 손실 -3% (-30 USDT / 잔고 1000) → 한도 5% 미달, 진입 허용."""
    from src.core.risk_manager import RiskManager

    strategy = _SignalOnlyStrategy(SignalType.BUY)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)
    engine._risk_manager = RiskManager(max_open_positions=3, max_daily_loss_pct=0.05)

    # 잔고 1000, 손실 -30 → -3% (한도 이내)
    mock_futures_client.get_balance.return_value = 1000.0

    db_patch = {
        "get_open_positions": MagicMock(return_value=[]),
        "get_today_pnl": MagicMock(return_value=(-30.0, 2)),
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "open_position": MagicMock(),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._open_position("BTCUSDT", "LONG", 50000.0)

    mock_futures_client.open_long.assert_called_once()


@pytest.mark.asyncio
async def test_open_position_allowed_when_no_trades_today(mock_futures_client):
    """당일 거래 없음 (PnL 0) → 진입 허용."""
    from src.core.risk_manager import RiskManager

    strategy = _SignalOnlyStrategy(SignalType.BUY)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)
    engine._risk_manager = RiskManager(max_open_positions=3, max_daily_loss_pct=0.05)

    mock_futures_client.get_balance.return_value = 1000.0

    db_patch = {
        "get_open_positions": MagicMock(return_value=[]),
        "get_today_pnl": MagicMock(return_value=(0.0, 0)),
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "open_position": MagicMock(),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._open_position("BTCUSDT", "LONG", 50000.0)

    mock_futures_client.open_long.assert_called_once()



# ── MAX_HOLD_HOURS Regression Tests ───────────────────────


class _MaxHoldStrategy(Strategy):
    """Strategy stub with configurable MAX_HOLD_HOURS.

    evaluate() increments an internal tick counter and emits a CLOSE
    signal (reason='max_hold') once ticks >= max_ticks, mirroring the
    logic in PatternScalper._evaluate_exit.
    """

    LEVERAGE = 5
    POSITION_SIZE_PCT = 0.20
    SL_PCT = 0.005
    FULL_TP_PCT = 0.50           # very wide - won't trigger in these tests
    PARTIAL_TP_PCT = 0.40
    TRAILING_ACTIVATE_PCT = 0.45
    TRAILING_DISTANCE_PCT = 0.10

    def __init__(self, max_hold_hours: float, signal_type: SignalType = SignalType.HOLD):
        self.MAX_HOLD_HOURS = max_hold_hours
        self._signal_type = signal_type
        self._ticks: int = 0   # test may pre-set to simulate elapsed ticks

    @property
    def name(self) -> str:
        return f"max_hold_stub_{self.MAX_HOLD_HOURS}h"

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.SIGNAL_ONLY

    def evaluate(self, symbol, candles, htf_candles=None) -> Signal:
        self._ticks += 1
        tick_interval_secs = 15  # same as PatternScalper._evaluate_exit assumption
        max_ticks = (
            int(self.MAX_HOLD_HOURS * 3600 / tick_interval_secs)
            if self.MAX_HOLD_HOURS > 0
            else 999_999
        )
        if self._ticks >= max_ticks:
            return Signal(
                symbol=symbol,
                type=SignalType.CLOSE,
                confidence=0.9,
                source=self.name,
                metadata={"reason": "max_hold", "ticks_held": self._ticks},
            )
        return Signal(
            symbol=symbol,
            type=self._signal_type,
            confidence=0.0,
            source=self.name,
        )


def _db_patch_for_close() -> dict:
    """Common db mock dict for tests that expect a position to be closed."""
    return {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "close_position": MagicMock(return_value=None),
        "delete_position": MagicMock(),
        "record_trail": MagicMock(),
        "get_position": MagicMock(return_value=None),
        "link_trails_to_trade": MagicMock(),
        "get_settings_hash": MagicMock(return_value="abc"),
    }


@pytest.mark.asyncio
async def test_max_hold_exceeded_long_position_closes(mock_futures_client):
    """MAX_HOLD_HOURS 경과 시 LONG 포지션 강제 청산 - close_long 호출 확인.

    1/240 h = 15 s => max_ticks = int(1/240 * 3600 / 15) = 1.
    첫 evaluate() 호출에서 _ticks=1 >= 1 => CLOSE 시그널 반환.
    """
    strategy = _MaxHoldStrategy(max_hold_hours=1 / 240)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_position.return_value = {
        "side": "LONG",
        "entry_price": 100.0,
        "quantity": 0.01,
    }

    with patch.multiple("src.core.futures_engine.db", **_db_patch_for_close()):
        await engine._tick_signal_only("BTCUSDT", strategy)

    mock_futures_client.close_long.assert_called_once()


@pytest.mark.asyncio
async def test_max_hold_exceeded_short_position_closes(mock_futures_client):
    """MAX_HOLD_HOURS 경과 시 SHORT 포지션 강제 청산 - close_short 호출 확인."""
    strategy = _MaxHoldStrategy(max_hold_hours=1 / 240)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_position.return_value = {
        "side": "SHORT",
        "entry_price": 100.0,
        "quantity": 0.01,
    }

    with patch.multiple("src.core.futures_engine.db", **_db_patch_for_close()):
        await engine._tick_signal_only("BTCUSDT", strategy)

    mock_futures_client.close_short.assert_called_once()


@pytest.mark.asyncio
async def test_max_hold_not_exceeded_does_not_close(mock_futures_client):
    """MAX_HOLD_HOURS 미경과 시 청산 안 됨 (첫 틱, 한도 960틱 기준)."""
    # 4-hour strategy: max_ticks=960; first tick -> _ticks=1 < 960 -> HOLD
    strategy = _MaxHoldStrategy(max_hold_hours=4.0)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_position.return_value = {
        "side": "LONG",
        "entry_price": 100.0,
        "quantity": 0.01,
    }

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "get_settings_hash": MagicMock(return_value="abc"),
        "get_position": MagicMock(return_value=None),
        "record_trail": MagicMock(),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_signal_only("BTCUSDT", strategy)

    mock_futures_client.close_long.assert_not_called()
    mock_futures_client.close_short.assert_not_called()


@pytest.mark.asyncio
async def test_max_hold_v12_4h_triggers_before_v9_6h(mock_futures_client):
    """v12(4h=960틱)은 청산되지만 v9(6h=1440틱)은 같은 틱 수에서 청산 안 됨.

    두 전략이 서로 다른 MAX_HOLD_HOURS 기준으로 독립적으로 동작함을 검증.
    """
    tick_interval_secs = 15
    v12_max_ticks = int(4.0 * 3600 / tick_interval_secs)   # 960
    v9_max_ticks = int(6.0 * 3600 / tick_interval_secs)    # 1440

    assert v12_max_ticks == 960, "v12 max_ticks sanity"
    assert v9_max_ticks == 1440, "v9 max_ticks sanity"

    # -- v12 at exactly 960 ticks -> CLOSE expected --------
    v12_strategy = _MaxHoldStrategy(max_hold_hours=4.0)
    v12_strategy._ticks = v12_max_ticks - 1   # evaluate() increments to 960
    engine_v12 = _make_engine_with_strategy(mock_futures_client, v12_strategy)
    mock_futures_client.get_position.return_value = {
        "side": "LONG",
        "entry_price": 100.0,
        "quantity": 0.01,
    }

    with patch.multiple("src.core.futures_engine.db", **_db_patch_for_close()):
        await engine_v12._tick_signal_only("BTCUSDT", v12_strategy)

    mock_futures_client.close_long.assert_called_once()
    mock_futures_client.reset_mock()

    # -- v9 at same 960 ticks -> below 1440 threshold -> no CLOSE -----
    v9_strategy = _MaxHoldStrategy(max_hold_hours=6.0)
    v9_strategy._ticks = v12_max_ticks - 1   # 959 -> 960 after evaluate()
    engine_v9 = _make_engine_with_strategy(mock_futures_client, v9_strategy)
    mock_futures_client.get_position.return_value = {
        "side": "LONG",
        "entry_price": 100.0,
        "quantity": 0.01,
    }

    db_patch_hold = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "get_settings_hash": MagicMock(return_value="abc"),
        "get_position": MagicMock(return_value=None),
        "record_trail": MagicMock(),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch_hold):
        await engine_v9._tick_signal_only("BTCUSDT", v9_strategy)

    mock_futures_client.close_long.assert_not_called()


@pytest.mark.asyncio
async def test_max_hold_zero_means_no_time_limit(mock_futures_client):
    """MAX_HOLD_HOURS=0 이면 시간 제한 없음 - 고(高)틱에서도 청산 안 됨."""
    strategy = _MaxHoldStrategy(max_hold_hours=0.0)
    strategy._ticks = 9_999  # well beyond any realistic threshold
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_position.return_value = {
        "side": "LONG",
        "entry_price": 100.0,
        "quantity": 0.01,
    }

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "get_settings_hash": MagicMock(return_value="abc"),
        "get_position": MagicMock(return_value=None),
        "record_trail": MagicMock(),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_signal_only("BTCUSDT", strategy)

    mock_futures_client.close_long.assert_not_called()
    mock_futures_client.close_short.assert_not_called()


# ── Candle fetch path test ─────────────────────────────────


@pytest.mark.asyncio
async def test_tick_signal_only_fetches_two_timeframes(mock_futures_client):
    """매 틱에서 15m + 1h 캔들을 각 1번씩 조회한다."""
    strategy = _SignalOnlyStrategy(SignalType.HOLD)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_position.return_value = None

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "get_settings_hash": MagicMock(return_value="abc"),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_signal_only("BTCUSDT", strategy)

    intervals_called = [
        call.kwargs.get("interval") or call.args[1]
        for call in mock_futures_client.get_candles.call_args_list
    ]
    assert "15m" in intervals_called
    assert "1h" in intervals_called


# ── TIMEFRAMES 클래스 속성 Tests ───────────────────────────


@pytest.mark.asyncio
async def test_fetch_candles_default_timeframes_calls_15m_and_1h(mock_futures_client):
    """기본 TIMEFRAMES=["15m","1h"] → get_candles 15m + 1h 각 1회, 총 2회 호출."""
    strategy = _SignalOnlyStrategy(SignalType.HOLD)
    # 기본 TIMEFRAMES는 Strategy ABC에서 상속 → ["15m", "1h"]
    assert strategy.TIMEFRAMES == ["15m", "1h"]

    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    await engine._fetch_candles("BTCUSDT", strategy)

    intervals_called = [
        call.kwargs.get("interval") or call.args[1]
        for call in mock_futures_client.get_candles.call_args_list
    ]
    assert intervals_called.count("15m") == 1
    assert intervals_called.count("1h") == 1
    assert len(intervals_called) == 2


@pytest.mark.asyncio
async def test_fetch_candles_custom_timeframes_5m_15m_1h_calls_three_times(mock_futures_client):
    """커스텀 TIMEFRAMES=["5m","15m","1h"] → get_candles 3회 호출 (5m, 15m, 1h 각 1번)."""
    class _CustomTFStrategy(_SignalOnlyStrategy):
        TIMEFRAMES = ["5m", "15m", "1h"]

    strategy = _CustomTFStrategy(SignalType.HOLD)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    await engine._fetch_candles("BTCUSDT", strategy)

    intervals_called = [
        call.kwargs.get("interval") or call.args[1]
        for call in mock_futures_client.get_candles.call_args_list
    ]
    # 3개 모두 각 1번씩 호출
    assert intervals_called.count("5m") == 1
    assert intervals_called.count("15m") == 1
    assert intervals_called.count("1h") == 1
    assert len(intervals_called) == 3


# ── Task 13.4: ATR 동적 포지션 사이징 엔진 통합 Tests ────────


class _AtrSignalStrategy(Strategy):
    """Strategy stub that exposes _last_signal for ATR sizing integration."""

    LEVERAGE = 5
    POSITION_SIZE_PCT = 0.20
    SL_ATR_MULT = 2.5

    def __init__(self, atr: float = 0.0):
        self._atr = atr
        self._last_signal: Signal | None = None

    @property
    def name(self) -> str:
        return "atr_signal_stub"

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.SIGNAL_ONLY

    def evaluate(self, symbol, candles, htf_candles=None) -> Signal:
        sig = Signal(
            symbol=symbol,
            type=SignalType.BUY,
            confidence=0.8,
            source=self.name,
            metadata={"atr": self._atr},
        )
        self._last_signal = sig
        return sig


@pytest.mark.asyncio
async def test_atr_position_sizing_no_atr_uses_full_size(mock_futures_client):
    """ATR=0(미설정) → scalar 적용 없음 → 기본 size_pct 그대로 사용 → 포지션 진입."""
    strategy = _AtrSignalStrategy(atr=0.0)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_balance.return_value = 1000.0

    db_patch = {
        "get_open_positions": MagicMock(return_value=[]),
        "get_today_pnl": MagicMock(return_value=(0.0, 0)),
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "open_position": MagicMock(),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._open_position("BTCUSDT", "LONG", 50000.0)

    mock_futures_client.open_long.assert_called_once()


def test_risk_manager_atr_2x_halves_position_size():
    """ATR 2× baseline → RiskManager scalar=0.5 → 사이즈 50% 축소 확인 (Task 13.4)."""
    from src.core.risk_manager import RiskManager

    rm = RiskManager()
    balance = 1000.0
    size_pct = 0.20
    atr_base = 100.0

    size_normal = rm.position_size(balance, size_pct, atr=atr_base, atr_baseline=atr_base)
    size_2x = rm.position_size(balance, size_pct, atr=atr_base * 2, atr_baseline=atr_base)

    assert size_2x == pytest.approx(size_normal * 0.5), (
        f"ATR 2× 시 사이즈가 50%로 축소되어야 합니다: normal={size_normal}, 2x={size_2x}"
    )


@pytest.mark.asyncio
async def test_open_position_calls_risk_manager_position_size(mock_futures_client):
    """_open_position이 RiskManager.position_size를 호출하는지 spy로 검증."""
    strategy = _AtrSignalStrategy(atr=0.0)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    mock_futures_client.get_balance.return_value = 1000.0

    db_patch = {
        "get_open_positions": MagicMock(return_value=[]),
        "get_today_pnl": MagicMock(return_value=(0.0, 0)),
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "open_position": MagicMock(),
    }

    with patch.object(
        engine._risk_manager, "position_size", wraps=engine._risk_manager.position_size
    ) as mock_ps, patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._open_position("BTCUSDT", "LONG", 50000.0)

    mock_ps.assert_called_once()
    call_kwargs = mock_ps.call_args
    assert call_kwargs is not None, "position_size가 호출되지 않았습니다"


# ── Task 17: 봇 재시작 내성 Tests ─────────────────────────────


# ── 17.1: 전략 상태 DB 왕복 저장/로드 ──────────────────────────


def test_save_and_load_strategy_state_roundtrip():
    """save_strategy_state → load_strategy_state 왕복 테스트 (DB mock)."""
    from src.core import database as database_module

    state = {
        "cooldown_remaining": 5,
        "consecutive_losses": 3,
        "trades_this_hour": 2,
        "last_hour": 14,
        "total_trades": 42,
        "wins": 25,
        "losses": 17,
    }

    with patch.object(database_module, "save_strategy_state") as mock_save, \
         patch.object(database_module, "load_strategy_state", return_value=state) as mock_load:
        database_module.save_strategy_state("pattern_scalper", state)
        loaded = database_module.load_strategy_state("pattern_scalper")

    mock_save.assert_called_once_with("pattern_scalper", state)
    mock_load.assert_called_once_with("pattern_scalper")
    assert loaded == state


@pytest.mark.asyncio
async def test_state_restored_after_restart(mock_futures_client):
    """재시작 후 cooldown_remaining / consecutive_losses가 DB에서 복원된다."""
    from src.core.futures_engine import FuturesEngine
    from src.strategies.pattern_scalper import PatternScalper

    strategy = PatternScalper()
    engine = FuturesEngine(mock_futures_client)

    saved_state = {
        "cooldown_remaining": 7,
        "consecutive_losses": 3,
        "trades_this_hour": 0,
        "last_hour": -1,
        "total_trades": 10,
        "wins": 6,
        "losses": 4,
    }

    db_patch = {
        "get_setting": MagicMock(return_value="pattern_scalper"),
        "get_settings_hash": MagicMock(return_value="abc"),
        "set_bot_running": MagicMock(),
        "is_bot_running": MagicMock(return_value=False),
        "load_strategy_state": MagicMock(return_value=saved_state),
        "save_strategy_state": MagicMock(),
        "get_position": MagicMock(return_value=None),
        "delete_position": MagicMock(),
    }

    # 거래소 포지션 없음
    mock_futures_client.get_position.return_value = None

    with patch.multiple("src.core.futures_engine.db", **db_patch), \
         patch("src.core.futures_engine.get_strategy", return_value=strategy):
        await engine.start_symbol("BTCUSDT")
        for task in list(engine._tasks.values()):
            task.cancel()
        if engine._paper_task:
            engine._paper_task.cancel()

    # 복원 확인
    assert strategy.state.cooldown_remaining == 7
    assert strategy.state.consecutive_losses == 3
    assert strategy.state.total_trades == 10


# ── 17.2: 재시작 시 거래소 포지션 동기화 ───────────────────────


@pytest.mark.asyncio
async def test_position_synced_from_exchange_on_start(mock_futures_client):
    """재시작 시 거래소에 열린 포지션이 있으면 전략 state에 반영된다."""
    from src.core.futures_engine import FuturesEngine
    from src.strategies.pattern_scalper import PatternScalper

    strategy = PatternScalper()
    engine = FuturesEngine(mock_futures_client)

    exchange_pos = {
        "side": "LONG",
        "entry_price": 85000.0,
        "quantity": 0.01,
    }
    mock_futures_client.get_position.return_value = exchange_pos

    db_patch = {
        "get_setting": MagicMock(return_value="pattern_scalper"),
        "get_settings_hash": MagicMock(return_value="abc"),
        "set_bot_running": MagicMock(),
        "is_bot_running": MagicMock(return_value=False),
        "load_strategy_state": MagicMock(return_value=None),
        "save_strategy_state": MagicMock(),
        "get_position": MagicMock(return_value=None),   # DB에는 없음
        "open_position": MagicMock(),
        "delete_position": MagicMock(),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch), \
         patch("src.core.futures_engine.get_strategy", return_value=strategy):
        await engine.start_symbol("BTCUSDT")
        for task in list(engine._tasks.values()):
            task.cancel()
        if engine._paper_task:
            engine._paper_task.cancel()

    # 전략 state에 포지션 반영됨
    assert strategy.state.position_side == "LONG"
    assert strategy.state.entry_price == 85000.0

    # DB에도 포지션 기록됨
    db_patch["open_position"].assert_called_once()


@pytest.mark.asyncio
async def test_no_exchange_position_clears_state_on_start(mock_futures_client):
    """재시작 시 거래소에 포지션 없으면 전략 state 포지션 필드 초기화 + DB 고아 정리."""
    from src.core.futures_engine import FuturesEngine
    from src.strategies.pattern_scalper import PatternScalper

    strategy = PatternScalper()
    # 이전에 포지션이 있었던 것처럼 state 세팅
    strategy.state.position_side = "SHORT"
    strategy.state.entry_price = 70000.0

    engine = FuturesEngine(mock_futures_client)
    mock_futures_client.get_position.return_value = None  # 거래소 포지션 없음

    mock_db_pos = MagicMock()
    db_patch = {
        "get_setting": MagicMock(return_value="pattern_scalper"),
        "get_settings_hash": MagicMock(return_value="abc"),
        "set_bot_running": MagicMock(),
        "is_bot_running": MagicMock(return_value=False),
        "load_strategy_state": MagicMock(return_value=None),
        "save_strategy_state": MagicMock(),
        "get_position": MagicMock(return_value=mock_db_pos),  # DB에는 있음 (고아)
        "delete_position": MagicMock(),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch), \
         patch("src.core.futures_engine.get_strategy", return_value=strategy):
        await engine.start_symbol("BTCUSDT")
        for task in list(engine._tasks.values()):
            task.cancel()
        if engine._paper_task:
            engine._paper_task.cancel()

    # 전략 state 포지션 초기화
    assert strategy.state.position_side == "NONE"
    assert strategy.state.entry_price == 0.0

    # DB 고아 레코드 삭제
    db_patch["delete_position"].assert_called_once_with("BTCUSDT")


# ── 17.3: 재시작 쿨다운 동안 신규 진입 차단 ─────────────────────


@pytest.mark.asyncio
async def test_startup_cooldown_blocks_new_entry(mock_futures_client):
    """재시작 쿨다운 > 0 이면 신규 진입을 차단한다 (포지션 없을 때)."""
    strategy = _SignalOnlyStrategy(SignalType.BUY)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    # 쿨다운 설정 (재시작 직후)
    engine._startup_cooldown_ticks["BTCUSDT"] = 5

    mock_futures_client.get_position.return_value = None  # 포지션 없음

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "get_settings_hash": MagicMock(return_value="abc"),
        "get_position": MagicMock(return_value=None),
        "record_trail": MagicMock(),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_signal_only("BTCUSDT", strategy)

    # 쿨다운 중이므로 진입 없음
    mock_futures_client.open_long.assert_not_called()
    mock_futures_client.open_short.assert_not_called()
    # 쿨다운 카운터가 감소됨
    assert engine._startup_cooldown_ticks["BTCUSDT"] == 4


@pytest.mark.asyncio
async def test_startup_cooldown_allows_entry_after_expiry(mock_futures_client):
    """쿨다운이 0이 되면 정상 진입한다."""
    strategy = _SignalOnlyStrategy(SignalType.BUY)
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    # 쿨다운 0 = 이미 만료
    engine._startup_cooldown_ticks["BTCUSDT"] = 0

    mock_futures_client.get_position.return_value = None

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "open_position": MagicMock(),
        "get_open_positions": MagicMock(return_value=[]),
        "get_today_pnl": MagicMock(return_value=(0.0, 0)),
        "get_settings_hash": MagicMock(return_value="abc"),
    }

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_signal_only("BTCUSDT", strategy)

    mock_futures_client.open_long.assert_called_once()


@pytest.mark.asyncio
async def test_startup_cooldown_does_not_block_position_management(mock_futures_client):
    """쿨다운 중에도 기존 포지션의 SL/TP 관리는 정상 동작한다."""
    strategy = _SignalOnlyStrategy(SignalType.HOLD)
    # FULL_TP_PCT 초과 → 청산이 먼저 발생
    strategy.FULL_TP_PCT = 0.001  # 0.1% — 테스트에서 쉽게 도달
    engine = _make_engine_with_strategy(mock_futures_client, strategy)

    # 쿨다운 설정
    engine._startup_cooldown_ticks["BTCUSDT"] = 10

    # 포지션 있음, 가격이 TP 근처
    pos = {"side": "LONG", "entry_price": 100.0, "quantity": 0.01}
    mock_futures_client.get_position.side_effect = [
        pos,     # _tick_signal_only 첫 조회 (포지션 체크)
        pos,     # _close_current 내부 조회
    ]

    db_patch = {
        "get_setting": MagicMock(return_value=""),
        "get_setting_float": MagicMock(return_value=0.005),
        "get_setting_int": MagicMock(return_value=5),
        "log_signal": MagicMock(),
        "close_position": MagicMock(return_value=None),
        "delete_position": MagicMock(),
        "record_trail": MagicMock(),
        "get_position": MagicMock(return_value=None),
        "link_trails_to_trade": MagicMock(),
        "get_settings_hash": MagicMock(return_value="abc"),
    }

    # 가격을 TP 초과로 설정 (entry 100 → price 102 = +2% > FULL_TP_PCT 0.1%)
    import pandas as pd
    import numpy as np
    close_prices = np.array([100.0] * 20 + [102.0])
    candles = pd.DataFrame({
        "open": close_prices,
        "high": close_prices * 1.001,
        "low": close_prices * 0.999,
        "close": close_prices,
        "volume": np.ones(len(close_prices)) * 1000,
    })
    mock_futures_client.get_candles.return_value = candles

    with patch.multiple("src.core.futures_engine.db", **db_patch):
        await engine._tick_signal_only("BTCUSDT", strategy)

    # 청산은 쿨다운과 무관하게 실행됨
    mock_futures_client.close_long.assert_called_once()
