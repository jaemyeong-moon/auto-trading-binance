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

    db_patch = {
        "get_setting": MagicMock(return_value="high_leverage_stub"),
        "get_settings_hash": MagicMock(return_value="abc"),
        "set_bot_running": MagicMock(),
        "is_bot_running": MagicMock(return_value=False),
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

    db_patch = {
        "get_setting": MagicMock(return_value="no_leverage_attr_stub"),
        "get_settings_hash": MagicMock(return_value="abc"),
        "set_bot_running": MagicMock(),
        "is_bot_running": MagicMock(return_value=False),
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

    db_patch = {
        "get_setting": MagicMock(return_value="high_leverage_stub"),
        "get_settings_hash": MagicMock(return_value="abc"),
        "set_bot_running": MagicMock(),
        "is_bot_running": MagicMock(return_value=False),
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
