"""심볼 간 상관행렬 계산 스크립트 (Task 13.5).

바이낸스 30일 일봉 데이터를 가져와 심볼 쌍 간 피어슨 상관계수를 계산하고
JSON 파일로 저장한다.  실행은 오프라인(수동) 환경에서만 수행한다.

사용법:
    python scripts/correlation.py [--symbols SYM1 SYM2 ...] [--output PATH]

예:
    python scripts/correlation.py \
        --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT \
        --output config/correlation.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from itertools import combinations
from pathlib import Path

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
DEFAULT_OUTPUT = "config/correlation.json"
LOOKBACK_DAYS = 30  # 일봉 기준


async def _fetch_daily_closes(symbol: str, limit: int = LOOKBACK_DAYS) -> list[float]:
    """바이낸스 일봉 종가 조회."""
    from src.exchange.futures_client import FuturesClient
    from src.core.config import ExchangeConfig

    cfg = ExchangeConfig()
    client = FuturesClient(cfg)
    await client.connect()
    try:
        candles = await client.get_candles(symbol, interval="1d", limit=limit)
        if candles.empty:
            return []
        return candles["close"].tolist()
    finally:
        await client.disconnect()


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """두 시계열의 피어슨 상관계수 계산."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0

    xs = xs[-n:]
    ys = ys[-n:]

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = (sum((x - mean_x) ** 2 for x in xs) / n) ** 0.5
    std_y = (sum((y - mean_y) ** 2 for y in ys) / n) ** 0.5

    if std_x == 0 or std_y == 0:
        return 0.0

    return cov / (n * std_x * std_y)


async def _build_correlation_matrix(
    symbols: list[str],
) -> dict[str, float]:
    """심볼 쌍별 상관계수 딕셔너리 반환.

    Returns:
        {"BTCUSDT-ETHUSDT": 0.92, ...}  형식의 딕셔너리.
    """
    closes: dict[str, list[float]] = {}
    for sym in symbols:
        print(f"Fetching {sym} ({LOOKBACK_DAYS}d daily candles)...")
        closes[sym] = await _fetch_daily_closes(sym)

    result: dict[str, float] = {}
    for sym_a, sym_b in combinations(symbols, 2):
        xs = closes.get(sym_a, [])
        ys = closes.get(sym_b, [])
        if not xs or not ys:
            corr = 0.0
        else:
            corr = _pearson_correlation(xs, ys)
        key = f"{sym_a}-{sym_b}"
        result[key] = round(corr, 4)
        print(f"  {key}: {corr:.4f}")

    return result


def _save_json(data: dict, output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nCorrelation matrix saved to: {path.resolve()}")


async def main(symbols: list[str], output: str) -> None:
    print(f"Computing {LOOKBACK_DAYS}-day correlation matrix for: {symbols}")
    matrix = await _build_correlation_matrix(symbols)
    _save_json(matrix, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Binance symbol correlation matrix")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Space-separated list of Binance futures symbols",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output JSON file path (default: config/correlation.json)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.symbols, args.output))
