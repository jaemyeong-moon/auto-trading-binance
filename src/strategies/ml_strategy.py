"""Machine learning strategy using XGBoost for signal prediction."""

from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import ta

from src.core.models import Signal, SignalType
from src.strategies.base import Strategy

logger = structlog.get_logger()

MODEL_DIR = Path(__file__).parent.parent.parent / "models"


class MLStrategy(Strategy):
    """XGBoost-based trading strategy."""

    def __init__(self) -> None:
        self._model = None

    @property
    def name(self) -> str:
        return "ml"

    def evaluate(self, symbol: str, candles: pd.DataFrame,
                 htf_candles: pd.DataFrame | None = None) -> Signal:
        if self._model is None:
            logger.warning("ml.no_model", msg="Model not loaded, returning HOLD")
            return Signal(symbol=symbol, type=SignalType.HOLD, confidence=0.0, source=self.name)

        features = self._extract_features(candles)
        if features is None:
            return Signal(symbol=symbol, type=SignalType.HOLD, confidence=0.0, source=self.name)

        prediction = self._model.predict_proba(features.values[-1:])
        # prediction shape: [[prob_sell, prob_hold, prob_buy]]
        probs = prediction[0]
        action_idx = np.argmax(probs)
        confidence = float(probs[action_idx])

        signal_map = {0: SignalType.SELL, 1: SignalType.HOLD, 2: SignalType.BUY}
        signal_type = signal_map[action_idx]

        if confidence < 0.65:
            signal_type = SignalType.HOLD

        return Signal(
            symbol=symbol,
            type=signal_type,
            confidence=confidence,
            source=self.name,
            metadata={"probs": probs.tolist()},
        )

    def load_model(self, path: Path | None = None) -> None:
        """Load a trained XGBoost model from disk."""
        import joblib

        model_path = path or MODEL_DIR / "xgboost_model.joblib"
        if model_path.exists():
            self._model = joblib.load(model_path)
            logger.info("ml.model_loaded", path=str(model_path))
        else:
            logger.warning("ml.model_not_found", path=str(model_path))

    def train(self, candles: pd.DataFrame) -> None:
        """Train the XGBoost model on historical data."""
        from xgboost import XGBClassifier
        import joblib

        features = self._extract_features(candles)
        if features is None or len(features) < 100:
            logger.warning("ml.insufficient_data", rows=len(features) if features is not None else 0)
            return

        labels = self._generate_labels(candles, features.index)

        X = features.dropna()
        y = labels.loc[X.index]

        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=3,
        )
        model.fit(X, y)
        self._model = model

        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_DIR / "xgboost_model.joblib")
        logger.info("ml.model_trained", samples=len(X))

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Build feature matrix from candle data."""
        if len(df) < 50:
            return None

        features = pd.DataFrame(index=df.index)
        close = df["close"]

        # Price-based
        features["returns_1"] = close.pct_change(1)
        features["returns_5"] = close.pct_change(5)
        features["returns_10"] = close.pct_change(10)
        features["volatility_10"] = close.rolling(10).std() / close.rolling(10).mean()

        # Technical indicators
        features["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        macd = ta.trend.MACD(close)
        features["macd_diff"] = macd.macd_diff()

        bb = ta.volatility.BollingerBands(close)
        features["bb_position"] = (close - bb.bollinger_lband()) / (
            bb.bollinger_hband() - bb.bollinger_lband() + 1e-10
        )

        # Volume
        features["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

        return features.dropna()

    def _generate_labels(self, df: pd.DataFrame, index: pd.Index) -> pd.Series:
        """Generate labels: 0=sell, 1=hold, 2=buy based on future returns."""
        future_returns = df["close"].pct_change(5).shift(-5)
        labels = pd.Series(1, index=df.index)  # default HOLD
        labels[future_returns > 0.02] = 2   # BUY
        labels[future_returns < -0.02] = 0  # SELL
        return labels.loc[index]
