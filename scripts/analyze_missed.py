"""v12 Pattern Scalper 미진입 원인 분석 + 놓친 기회 탐지.

바이낸스 실제 가격 데이터로:
1. 패턴 스캐너가 감지한/못한 패턴 통계
2. HOLD 사유별 분포
3. 실제 가격 움직임 대비 놓친 기회 (가격이 2%+ 움직인 구간에서 미진입)
4. 패턴 조건 완화 시뮬레이션

Usage: python scripts/analyze_missed.py [--candles 2000] [--interval 15m]
"""

import asyncio
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import ta

from src.core.models import SignalType
from src.exchange.futures_client import FuturesClient
from src.strategies.patterns import (
    scan_all_patterns, analyze_volume,
    detect_double_bottom, detect_double_top,
    detect_inv_head_shoulders, detect_head_shoulders,
    detect_bull_flag, detect_bear_flag,
    detect_triangle_breakout,
    _local_minima, _local_maxima,
    LOOKBACK, PROXIMITY_PCT,
)
from src.strategies.registry import get_strategy


SEP = "=" * 90


# ─── 1. 틱별 HOLD 사유 분석 ──────────────────────────────────

def analyze_hold_reasons(strategy, symbol, candles, htf_candles):
    """매 캔들마다 evaluate()를 호출하여 HOLD 사유 수집."""
    reasons = Counter()
    signals_log = []
    warmup = 130

    for i in range(warmup, len(candles)):
        window = candles.iloc[:i + 1]
        htf_w = htf_candles if htf_candles is not None and not htf_candles.empty else None

        sig = strategy.evaluate(symbol, window, htf_w)

        reason = sig.metadata.get("reason", "unknown") if sig.metadata else "unknown"

        if sig.type == SignalType.HOLD:
            reasons[reason] += 1
        else:
            signals_log.append({
                "idx": i,
                "time": candles.index[i],
                "type": sig.type.value,
                "confidence": sig.confidence,
                "pattern": sig.metadata.get("pattern", ""),
                "direction": sig.metadata.get("direction", ""),
                "reason": reason,
            })

        # 포지션 상태 리셋 (분석용 — 매 틱 독립 평가)
        if hasattr(strategy, 'state'):
            strategy.state.position_side = "NONE"
            strategy.state.cooldown_remaining = 0
            strategy.state.trades_this_hour = 0

    return reasons, signals_log


# ─── 2. 놓친 기회 탐지 ─────────────────────────────────────────

def find_missed_opportunities(candles, signals_log, threshold_pct=1.5, lookahead=20):
    """가격이 threshold% 이상 움직인 구간에서 시그널이 없었던 곳 = 놓친 기회."""
    close = candles["close"].values.astype(float)
    missed = []
    signal_indices = set(s["idx"] for s in signals_log)

    for i in range(130, len(close) - lookahead):
        future = close[i + 1: i + lookahead + 1]
        if len(future) == 0:
            continue

        max_up = (future.max() - close[i]) / close[i] * 100
        max_down = (close[i] - future.min()) / close[i] * 100

        if max_up >= threshold_pct and i not in signal_indices:
            missed.append({
                "idx": i, "time": candles.index[i],
                "price": close[i], "direction": "LONG",
                "move_pct": round(max_up, 2),
                "best_price": round(future.max(), 2),
            })
        elif max_down >= threshold_pct and i not in signal_indices:
            missed.append({
                "idx": i, "time": candles.index[i],
                "price": close[i], "direction": "SHORT",
                "move_pct": round(max_down, 2),
                "best_price": round(future.min(), 2),
            })

    return missed


# ─── 3. 패턴 감지 실패 분석 (조건별) ──────────────────────────

def diagnose_pattern_failures(candles, missed_opportunities):
    """놓친 기회 시점에서 패턴 감지가 왜 실패했는지 진단."""
    close = candles["close"].values.astype(float)
    high = candles["high"].values.astype(float)
    low = candles["low"].values.astype(float)
    volume = candles["volume"].values.astype(float)

    atr_series = ta.volatility.AverageTrueRange(
        candles["high"], candles["low"], candles["close"], window=14
    ).average_true_range()

    failure_reasons = Counter()
    diagnosable = []

    for opp in missed_opportunities[:100]:  # 상위 100개만 분석
        i = opp["idx"]
        n = min(LOOKBACK, i + 1)
        lo = low[i - n + 1:i + 1]
        hi = high[i - n + 1:i + 1]
        cl = close[i - n + 1:i + 1]
        vol = volume[i - n + 1:i + 1]
        atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 1.0

        # 극점 확인
        mins = _local_minima(lo)
        maxs = _local_maxima(hi)

        reasons = []

        if len(mins) < 2:
            reasons.append("too_few_minima")
        if len(maxs) < 2:
            reasons.append("too_few_maxima")

        # 쌍바닥/쌍봉 근접도 체크
        if len(mins) >= 2:
            i2, i1 = mins[-1], mins[-2]
            l1, l2 = lo[i1], lo[i2]
            avg = (l1 + l2) / 2
            prox = abs(l1 - l2) / avg if avg > 0 else 999
            if prox > PROXIMITY_PCT:
                reasons.append(f"bottom_proximity_fail({prox:.4f}>{PROXIMITY_PCT})")
            else:
                # 넥라인 체크
                peaks = [m for m in maxs if i1 < m < i2]
                if not peaks:
                    reasons.append("no_neckline_peak")
                else:
                    neck = hi[max(peaks, key=lambda m: hi[m])]
                    price = cl[-1]
                    dist = (price - neck) / neck
                    if dist < -0.02:
                        reasons.append(f"price_below_neckline({dist:.4f})")
                    elif dist > 0.01:
                        reasons.append(f"price_too_far_above({dist:.4f})")

        # 전체 패턴 스캔 결과
        patterns = scan_all_patterns(lo, hi, cl, vol, atr)
        if patterns:
            reasons.append(f"patterns_found({','.join(p.name for p in patterns)})")
        else:
            reasons.append("no_pattern_detected")

        # 거래량 분석
        vol_sig = analyze_volume(cl, vol)
        reasons.append(f"vol_bias={vol_sig.bias}({vol_sig.strength:.2f})")

        for r in reasons:
            failure_reasons[r.split("(")[0]] += 1

        diagnosable.append({
            **opp,
            "reasons": reasons,
            "n_minima": len(mins),
            "n_maxima": len(maxs),
            "atr": round(atr, 2),
        })

    return failure_reasons, diagnosable


# ─── 4. 조건 완화 시뮬레이션 ──────────────────────────────────

def simulate_relaxed(candles, htf_candles, symbol, lookahead=20):
    """패턴 조건을 완화했을 때의 진입 횟수 + 수익성 시뮬레이션."""
    close = candles["close"].values.astype(float)
    high = candles["high"].values.astype(float)
    low = candles["low"].values.astype(float)
    volume = candles["volume"].values.astype(float)

    atr_series = ta.volatility.AverageTrueRange(
        candles["high"], candles["low"], candles["close"], window=14
    ).average_true_range()

    # 현재 조건 vs 완화 조건
    configs = {
        "현재": {"proximity": 0.005, "dist_lower": -0.02, "dist_upper": 0.01, "min_height": 0.001},
        "완화1_근접도": {"proximity": 0.01, "dist_lower": -0.02, "dist_upper": 0.01, "min_height": 0.001},
        "완화2_거리": {"proximity": 0.005, "dist_lower": -0.03, "dist_upper": 0.02, "min_height": 0.001},
        "완화3_높이": {"proximity": 0.005, "dist_lower": -0.02, "dist_upper": 0.01, "min_height": 0.0005},
        "완화_ALL": {"proximity": 0.01, "dist_lower": -0.03, "dist_upper": 0.02, "min_height": 0.0005},
    }

    results = {}

    for config_name, cfg in configs.items():
        entries = []
        for i in range(130, len(close) - lookahead):
            n = min(LOOKBACK, i + 1)
            lo = low[i - n + 1:i + 1]
            hi = high[i - n + 1:i + 1]
            cl = close[i - n + 1:i + 1]
            vol = volume[i - n + 1:i + 1]
            atr = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0
            if atr <= 0:
                continue

            # 완화된 쌍바닥 감지
            mins = _local_minima(lo)
            maxs = _local_maxima(hi)

            if len(mins) >= 2:
                i2, i1 = mins[-1], mins[-2]
                if i2 - i1 >= 5:
                    l1, l2 = lo[i1], lo[i2]
                    avg = (l1 + l2) / 2
                    if avg > 0 and abs(l1 - l2) / avg <= cfg["proximity"]:
                        peaks = [m for m in maxs if i1 < m < i2]
                        if peaks:
                            neck = hi[max(peaks, key=lambda m: hi[m])]
                            height = neck - avg
                            if height / avg >= cfg["min_height"]:
                                price = cl[-1]
                                dist = (price - neck) / neck
                                if cfg["dist_lower"] <= dist <= cfg["dist_upper"]:
                                    # 진입! 결과 추적
                                    future = close[i + 1:i + lookahead + 1]
                                    max_up = (future.max() - price) / price * 100
                                    max_down = (price - future.min()) / price * 100
                                    entries.append({
                                        "idx": i, "direction": "LONG",
                                        "price": price, "max_up": max_up, "max_down": max_down,
                                        "profitable": max_up > 1.0,
                                    })

            # 완화된 쌍봉 감지
            if len(maxs) >= 2:
                i2, i1 = maxs[-1], maxs[-2]
                if i2 - i1 >= 5:
                    h1, h2 = hi[i1], hi[i2]
                    avg = (h1 + h2) / 2
                    if avg > 0 and abs(h1 - h2) / avg <= cfg["proximity"]:
                        troughs = [m for m in mins if i1 < m < i2]
                        if troughs:
                            neck = lo[min(troughs, key=lambda m: lo[m])]
                            height = avg - neck
                            if height / avg >= cfg["min_height"]:
                                price = cl[-1]
                                dist = (neck - price) / neck
                                if cfg["dist_lower"] <= dist <= cfg["dist_upper"]:
                                    future = close[i + 1:i + lookahead + 1]
                                    max_up = (future.max() - price) / price * 100
                                    max_down = (price - future.min()) / price * 100
                                    entries.append({
                                        "idx": i, "direction": "SHORT",
                                        "price": price, "max_up": max_up, "max_down": max_down,
                                        "profitable": max_down > 1.0,
                                    })

        n = len(entries)
        wins = sum(1 for e in entries if e["profitable"])
        results[config_name] = {
            "entries": n,
            "wins": wins,
            "win_rate": wins / n * 100 if n > 0 else 0,
            "avg_max_favorable": np.mean([
                e["max_up"] if e["direction"] == "LONG" else e["max_down"]
                for e in entries
            ]) if entries else 0,
        }

    return results


# ─── 메인 ────────────────────────────────────────────────────

async def fetch_data(symbols, candle_count, interval):
    HTF_MAP = {"1m": "15m", "5m": "1h", "15m": "1h", "1h": "4h"}
    htf_interval = HTF_MAP.get(interval, "1h")

    client = FuturesClient()
    await client.connect()
    try:
        data = {}
        for sym in symbols:
            print(f"  {sym} 데이터 로딩...", end=" ", flush=True)
            candles = await client.get_candles(sym, interval=interval, limit=candle_count)
            htf = await client.get_candles(sym, interval=htf_interval, limit=500)
            data[sym] = {"candles": candles, "htf": htf}
            print(f"{interval}={len(candles)}개, {htf_interval}={len(htf)}개")
        return data
    finally:
        await client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="v12 미진입 원인 분석")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--candles", type=int, default=2000)
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--threshold", type=float, default=1.5,
                        help="놓친 기회 판정 기준 %% (기본 1.5%%)")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    tf_min = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}.get(args.interval, 15)
    hours = args.candles * tf_min / 60
    days = hours / 24

    print()
    print(SEP)
    print(f"  v12 Pattern Scalper 미진입 원인 분석")
    print(f"  캔들: {args.candles}개 ({args.interval}봉 = ~{days:.1f}일)")
    print(f"  놓친 기회 기준: {args.threshold}%+ 움직임")
    print(SEP)

    # 데이터 로드
    print()
    data = asyncio.run(fetch_data(symbols, args.candles, args.interval))

    for sym in symbols:
        candles = data[sym]["candles"]
        htf = data[sym]["htf"]
        if candles.empty or len(candles) < 200:
            print(f"\n  {sym}: 데이터 부족, 스킵")
            continue

        print()
        print(SEP)
        print(f"  === {sym} 분석 ===")
        print(SEP)

        # ── 1. HOLD 사유 분석 ──
        print(f"\n  [1] HOLD 사유 분석 (총 {len(candles) - 130}틱)...")
        strategy = get_strategy("pattern_scalper")
        reasons, signals = analyze_hold_reasons(strategy, sym, candles, htf)

        total_ticks = sum(reasons.values()) + len(signals)
        print(f"\n  총 틱: {total_ticks}  |  진입 시그널: {len(signals)}건  |  HOLD: {sum(reasons.values())}건")
        print(f"  진입률: {len(signals) / total_ticks * 100:.2f}%")
        print()
        print(f"  {'HOLD 사유':<35s} {'건수':>6s} {'비율':>7s}")
        print(f"  {'-' * 50}")
        for reason, cnt in reasons.most_common(20):
            print(f"  {reason:<35s} {cnt:>6d} {cnt / total_ticks * 100:>6.1f}%")

        if signals:
            print(f"\n  진입 시그널 상세 (최근 20건):")
            print(f"  {'시간':<20s} {'타입':<6s} {'방향':<6s} {'패턴':<20s} {'신뢰도':>6s}")
            print(f"  {'-' * 65}")
            for s in signals[-20:]:
                print(f"  {str(s['time']):<20s} {s['type']:<6s} {s['direction']:<6s} "
                      f"{s['pattern']:<20s} {s['confidence']:>6.3f}")

        # ── 2. 놓친 기회 탐지 ──
        print(f"\n  [2] 놓친 기회 탐지 ({args.threshold}%+ 움직임, {args.interval} {20}봉 선행)...")
        missed = find_missed_opportunities(candles, signals, args.threshold)

        long_missed = [m for m in missed if m["direction"] == "LONG"]
        short_missed = [m for m in missed if m["direction"] == "SHORT"]
        print(f"\n  놓친 기회: {len(missed)}건 (LONG {len(long_missed)}, SHORT {len(short_missed)})")

        if missed:
            print(f"\n  놓친 기회 TOP 15 (움직임 크기순):")
            print(f"  {'시간':<20s} {'방향':<6s} {'가격':>12s} {'움직임':>7s} {'최적가':>12s}")
            print(f"  {'-' * 65}")
            for m in sorted(missed, key=lambda x: x["move_pct"], reverse=True)[:15]:
                print(f"  {str(m['time']):<20s} {m['direction']:<6s} {m['price']:>12.2f} "
                      f"{m['move_pct']:>+6.2f}% {m['best_price']:>12.2f}")

        # ── 3. 패턴 실패 진단 ──
        print(f"\n  [3] 패턴 감지 실패 진단 (놓친 기회 상위 100건)...")
        failure_reasons, diagnostics = diagnose_pattern_failures(candles, missed)

        print(f"\n  {'실패 원인':<35s} {'건수':>6s}")
        print(f"  {'-' * 45}")
        for reason, cnt in failure_reasons.most_common(20):
            print(f"  {reason:<35s} {cnt:>6d}")

        # ── 4. 조건 완화 시뮬레이션 ──
        print(f"\n  [4] 조건 완화 시뮬레이션 (쌍바닥/쌍봉만)...")
        relaxed = simulate_relaxed(candles, htf, sym)

        print(f"\n  {'설정':<20s} {'진입수':>6s} {'승리':>5s} {'승률':>7s} {'평균유리%':>9s}")
        print(f"  {'-' * 55}")
        for name, r in relaxed.items():
            print(f"  {name:<20s} {r['entries']:>6d} {r['wins']:>5d} "
                  f"{r['win_rate']:>6.1f}% {r['avg_max_favorable']:>8.2f}%")

    # ── 5. 전체 백테스트 (현재 vs 이상적) ──
    print()
    print(SEP)
    print("  [5] 현재 v12 전체 백테스트")
    print(SEP)

    from scripts.backtest_all import simulate as bt_simulate, print_report

    all_results = []
    for sym in symbols:
        candles = data[sym]["candles"]
        htf = data[sym]["htf"]
        if candles.empty:
            continue

        strategy = get_strategy("pattern_scalper")
        result = bt_simulate(strategy, sym, candles, htf, capital=200, leverage=7)
        n = len(result["trades"])
        pnl = result["balance"] - 200
        print(f"  pattern_scalper / {sym}: {n}건, ${pnl:+.2f}")
        all_results.append({
            "strategy_name": "pattern_scalper",
            "strategy_label": "v12. Multi-Pattern Scalper",
            "symbol": sym,
            **result,
        })

    if all_results:
        print_report(all_results, 200)

    print()
    print(SEP)
    print("  분석 완료")
    print(SEP)


if __name__ == "__main__":
    main()
