"""Technical analysis strategy using RSI, MACD, and Bollinger Bands."""

import pandas as pd
import ta

from src.core.config import settings
from src.core.models import Signal, SignalType
from src.strategies.base import Strategy


class TechnicalStrategy(Strategy):
    """Generates signals based on technical indicators."""

    @property
    def name(self) -> str:
        return "technical"

    def evaluate(self, symbol: str, candles: pd.DataFrame,
                 htf_candles: pd.DataFrame | None = None) -> Signal:
        df = self._add_indicators(candles)
        latest = df.iloc[-1]

        buy_signals = 0
        sell_signals = 0
        total_indicators = 3

        # RSI
        cfg = settings.trading
        rsi = latest.get("rsi")
        if rsi is not None:
            if rsi < 30:
                buy_signals += 1
            elif rsi > 70:
                sell_signals += 1

        # MACD
        macd_diff = latest.get("macd_diff")
        if macd_diff is not None:
            if macd_diff > 0:
                buy_signals += 1
            else:
                sell_signals += 1

        # Bollinger Bands
        close = latest["close"]
        bb_low = latest.get("bb_low")
        bb_high = latest.get("bb_high")
        if bb_low is not None and bb_high is not None:
            if close <= bb_low:
                buy_signals += 1
            elif close >= bb_high:
                sell_signals += 1

        # Determine signal
        if buy_signals >= 2:
            return Signal(
                symbol=symbol,
                type=SignalType.BUY,
                confidence=buy_signals / total_indicators,
                source=self.name,
            )
        elif sell_signals >= 2:
            return Signal(
                symbol=symbol,
                type=SignalType.SELL,
                confidence=sell_signals / total_indicators,
                source=self.name,
            )
        return Signal(
            symbol=symbol,
            type=SignalType.HOLD,
            confidence=0.0,
            source=self.name,
        )

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators on the candle DataFrame."""
        df = df.copy()
        close = df["close"]

        # RSI
        df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

        # MACD
        macd = ta.trend.MACD(close)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close)
        df["bb_high"] = bb.bollinger_hband()
        df["bb_mid"] = bb.bollinger_mavg()
        df["bb_low"] = bb.bollinger_lband()

        return df
