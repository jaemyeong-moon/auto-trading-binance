"""Backtesting engine — simulates strategy on historical data."""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import structlog

from src.core.config import settings
from src.core.models import SignalType
from src.strategies.base import Strategy

logger = structlog.get_logger()


@dataclass
class BacktestTrade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    quantity: float
    pnl: float
    pnl_pct: float


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


class Backtester:
    """Run a strategy against historical candle data."""

    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.001,  # 0.1% Binance fee
    ) -> None:
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct

    def run(self, symbol: str, candles: pd.DataFrame) -> BacktestResult:
        """Execute backtest on historical candles."""
        capital = self.initial_capital
        position_qty = 0.0
        entry_price = 0.0
        entry_time = None

        trades: list[BacktestTrade] = []
        equity_curve: list[float] = [capital]
        peak_capital = capital

        position_size_pct = settings.trading.position_size_pct
        stop_loss_pct = settings.risk.stop_loss_pct
        take_profit_pct = settings.risk.take_profit_pct

        # Need enough rows for indicator warm-up
        min_rows = 50
        if len(candles) < min_rows:
            logger.warning("backtest.insufficient_data", rows=len(candles))
            return self._empty_result(symbol, candles)

        for i in range(min_rows, len(candles)):
            window = candles.iloc[: i + 1]
            current_price = candles.iloc[i]["close"]
            current_time = candles.index[i]

            # Check stop-loss / take-profit for open position
            if position_qty > 0:
                price_change = (current_price - entry_price) / entry_price

                if price_change <= -stop_loss_pct or price_change >= take_profit_pct:
                    # Close position
                    revenue = position_qty * current_price
                    commission = revenue * self.commission_pct
                    pnl = revenue - (position_qty * entry_price) - commission
                    capital += revenue - commission
                    trades.append(BacktestTrade(
                        symbol=symbol,
                        side="SELL",
                        entry_price=entry_price,
                        exit_price=current_price,
                        entry_time=entry_time,
                        exit_time=current_time,
                        quantity=position_qty,
                        pnl=pnl,
                        pnl_pct=price_change * 100,
                    ))
                    position_qty = 0.0
                    equity_curve.append(capital)
                    peak_capital = max(peak_capital, capital)
                    continue

            signal = self.strategy.evaluate(symbol, window)

            if signal.type == SignalType.BUY and position_qty == 0:
                # Open position
                invest = capital * position_size_pct
                commission = invest * self.commission_pct
                position_qty = (invest - commission) / current_price
                capital -= invest
                entry_price = current_price
                entry_time = current_time

            elif signal.type == SignalType.SELL and position_qty > 0:
                # Close position
                revenue = position_qty * current_price
                commission = revenue * self.commission_pct
                pnl = revenue - (position_qty * entry_price) - commission
                capital += revenue - commission
                trades.append(BacktestTrade(
                    symbol=symbol,
                    side="SELL",
                    entry_price=entry_price,
                    exit_price=current_price,
                    entry_time=entry_time,
                    exit_time=current_time,
                    quantity=position_qty,
                    pnl=pnl,
                    pnl_pct=((current_price - entry_price) / entry_price) * 100,
                ))
                position_qty = 0.0

            # Track equity (capital + open position value)
            total_equity = capital + (position_qty * current_price)
            equity_curve.append(total_equity)
            peak_capital = max(peak_capital, total_equity)

        # Close any remaining position at last price
        if position_qty > 0:
            last_price = candles.iloc[-1]["close"]
            revenue = position_qty * last_price
            capital += revenue - (revenue * self.commission_pct)

        final_capital = capital
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        # Max drawdown
        max_dd = 0.0
        peak = equity_curve[0]
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        # Sharpe ratio (annualized, assuming hourly candles)
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * (8760 ** 0.5)  # hourly -> annual
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        return BacktestResult(
            strategy_name=self.strategy.name,
            symbol=symbol,
            start_date=candles.index[0],
            end_date=candles.index[-1],
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning) / len(trades) * 100 if trades else 0.0,
            total_return_pct=((final_capital - self.initial_capital) / self.initial_capital) * 100,
            max_drawdown_pct=max_dd * 100,
            sharpe_ratio=sharpe,
            trades=trades,
            equity_curve=equity_curve,
        )

    def _empty_result(self, symbol: str, candles: pd.DataFrame) -> BacktestResult:
        return BacktestResult(
            strategy_name=self.strategy.name,
            symbol=symbol,
            start_date=candles.index[0] if len(candles) > 0 else datetime.now(),
            end_date=candles.index[-1] if len(candles) > 0 else datetime.now(),
            initial_capital=self.initial_capital,
            final_capital=self.initial_capital,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
        )
