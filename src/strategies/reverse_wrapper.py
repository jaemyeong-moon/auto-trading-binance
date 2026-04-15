"""역신호 래퍼 — 임의의 Strategy 신호를 BUY↔SELL로 뒤집는 래퍼.

사용자 분석(Phase 18): 운영된 모든 전략이 페이퍼에서 WR 50% 미만.
즉 진입 방향이 체계적으로 틀리고 있다는 가설.

이 래퍼로 기존 전략들의 반대 신호를 페이퍼에 동시 투입하고,
paper_selector가 실제 승률을 측정해서 더 나은 쪽을 자동 선택하도록 한다.

뒤집히는 것:
- SignalType.BUY  → SignalType.SELL
- SignalType.SELL → SignalType.BUY

그대로 두는 것:
- SignalType.HOLD  (대기)
- SignalType.CLOSE (청산 — 방향 개념 없음)

매매 파라미터(SL_ATR_MULT, LEVERAGE 등)는 원본 전략 클래스에서 상속한다.
"""

from __future__ import annotations

import pandas as pd

from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.registry import _REGISTRY, register


_FLIP_MAP = {
    SignalType.BUY: SignalType.SELL,
    SignalType.SELL: SignalType.BUY,
    SignalType.HOLD: SignalType.HOLD,
    SignalType.CLOSE: SignalType.CLOSE,
}


class ReverseStrategy(Strategy):
    """베이스 역신호 래퍼. 서브클래스에서 `_INNER_CLS`와 `_NAME`만 지정."""

    _INNER_CLS: type[Strategy] = None  # type: ignore[assignment]
    _NAME: str = ""

    def __init__(self) -> None:
        self._inner: Strategy = self._INNER_CLS()  # type: ignore[misc]

    # ── 매매 파라미터: 원본에서 상속 ──
    @property
    def LEVERAGE(self) -> int:  # noqa: N802 — 기존 속성명 유지
        return getattr(self._inner, "LEVERAGE", 5)

    @property
    def POSITION_SIZE_PCT(self) -> float:  # noqa: N802
        return getattr(self._inner, "POSITION_SIZE_PCT", 0.20)

    @property
    def MAX_HOLD_HOURS(self) -> float:  # noqa: N802
        return getattr(self._inner, "MAX_HOLD_HOURS", 4.0)

    @property
    def TIMEFRAMES(self) -> list[str]:  # noqa: N802
        return getattr(self._inner, "TIMEFRAMES", ["15m", "1h"])

    @property
    def SL_ATR_MULT(self) -> float:  # noqa: N802
        return getattr(self._inner, "SL_ATR_MULT", 2.0)

    @property
    def TP_ATR_MULT(self) -> float:  # noqa: N802
        return getattr(self._inner, "TP_ATR_MULT", 4.0)

    @property
    def PARTIAL_TP_ATR_MULT(self) -> float:  # noqa: N802
        return getattr(self._inner, "PARTIAL_TP_ATR_MULT", 0.0)

    @property
    def TRAILING_ATR_MULT(self) -> float:  # noqa: N802
        return getattr(self._inner, "TRAILING_ATR_MULT", 0.0)

    @property
    def TRAILING_DIST_ATR(self) -> float:  # noqa: N802
        return getattr(self._inner, "TRAILING_DIST_ATR", 0.0)

    @property
    def state(self):
        """원본 전략의 state를 그대로 노출 — futures_engine이 entry_atr 등 참조."""
        return getattr(self._inner, "state", None)

    # ── 전략 메타 ──
    @property
    def name(self) -> str:
        return self._NAME

    @property
    def label(self) -> str:
        return f"{self._inner.label} (역신호)"

    @property
    def description(self) -> str:
        return f"Reverse of {self._inner.name}: BUY↔SELL 뒤집기. {self._inner.description}"

    @property
    def mode(self) -> ExecutionMode:
        return self._inner.mode

    # ── 신호 평가: 원본 평가 후 BUY/SELL만 뒤집기 ──
    def evaluate(self, symbol: str, candles: pd.DataFrame,
                 htf_candles: pd.DataFrame | None = None) -> Signal:
        inner_signal = self._inner.evaluate(symbol, candles, htf_candles)
        flipped_type = _FLIP_MAP.get(inner_signal.type, inner_signal.type)

        new_meta = dict(inner_signal.metadata or {})
        new_meta["reversed_from"] = self._inner.name
        new_meta["original_type"] = inner_signal.type.value

        return Signal(
            symbol=inner_signal.symbol,
            type=flipped_type,
            confidence=inner_signal.confidence,
            source=self._NAME,
            metadata=new_meta,
        )

    def record_result(self, pnl: float) -> None:
        # 원본 전략에 결과 전달 — 원본의 연패카운터/learning 로직이 계속 돌도록.
        self._inner.record_result(pnl)


def make_reverse(inner_name: str, reverse_name: str | None = None) -> type[ReverseStrategy]:
    """레지스트리에 등록된 전략의 역신호 래퍼를 동적으로 만들어 등록한다.

    Args:
        inner_name: 원본 전략 이름 (레지스트리 key)
        reverse_name: 역신호 전략 이름 (기본: "reverse_" + inner_name)

    Returns:
        생성·등록된 래퍼 클래스.
    """
    inner_cls = _REGISTRY.get(inner_name)
    if inner_cls is None:
        raise ValueError(
            f"Cannot make reverse: unknown inner strategy '{inner_name}'. "
            f"Register it first. Available: {list(_REGISTRY.keys())}"
        )

    rev_name = reverse_name or f"reverse_{inner_name}"

    # 동적으로 서브클래스 생성 (필수 클래스변수 주입) → @register로 등록.
    cls = type(
        f"Reverse_{inner_cls.__name__}",
        (ReverseStrategy,),
        {"_INNER_CLS": inner_cls, "_NAME": rev_name},
    )
    return register(cls)


# ── 실제 역신호 전략 등록 ──
# 페이퍼 성과 기준 (2026-04-15): 모두 WR 50% 미만 → 반대가 유효할 가능성
# momentum_flip  WR 38.3% (47거래) → reverse 기대 ~62%
# orderflow_v13  WR  0.0% ( 7거래) → reverse 기대 100%
# pattern_scalper WR 36.4% (11거래) → reverse 기대 ~64%
# data_driven    WR 28.6% ( 7거래) → reverse 기대 ~71%
make_reverse("momentum_flip_scalper")
make_reverse("orderflow_v13")
make_reverse("pattern_scalper")
make_reverse("data_driven_scalper")
