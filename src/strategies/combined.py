"""Combined strategy that merges technical and ML signals."""

import pandas as pd

from src.core.models import Signal, SignalType
from src.strategies.base import Strategy
from src.strategies.ml_strategy import MLStrategy
from src.strategies.technical import TechnicalStrategy


class CombinedStrategy(Strategy):
    """Combines technical and ML signals with weighted voting."""

    def __init__(
        self,
        technical_weight: float = 0.4,
        ml_weight: float = 0.6,
    ) -> None:
        self.technical = TechnicalStrategy()
        self.ml = MLStrategy()
        self.technical_weight = technical_weight
        self.ml_weight = ml_weight

    @property
    def name(self) -> str:
        return "combined"

    def evaluate(self, symbol: str, candles: pd.DataFrame,
                 htf_candles: pd.DataFrame | None = None) -> Signal:
        tech_signal = self.technical.evaluate(symbol, candles)
        ml_signal = self.ml.evaluate(symbol, candles)

        # Convert signals to numeric scores: BUY=1, HOLD=0, SELL=-1
        score_map = {SignalType.BUY: 1.0, SignalType.HOLD: 0.0, SignalType.SELL: -1.0}
        tech_score = score_map[tech_signal.type] * tech_signal.confidence
        ml_score = score_map[ml_signal.type] * ml_signal.confidence

        combined = (
            tech_score * self.technical_weight + ml_score * self.ml_weight
        )

        if combined > 0.3:
            signal_type = SignalType.BUY
        elif combined < -0.3:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        return Signal(
            symbol=symbol,
            type=signal_type,
            confidence=abs(combined),
            source=self.name,
            metadata={
                "technical": {"type": tech_signal.type.value, "confidence": tech_signal.confidence},
                "ml": {"type": ml_signal.type.value, "confidence": ml_signal.confidence},
                "combined_score": combined,
            },
        )
