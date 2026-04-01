"""전체 전략 백테스트 리포트.

Usage: python scripts/backtest_all.py [--symbols BTCUSDT,ETHUSDT] [--candles 1000]
"""

import asyncio
import io
import sys
from pathlib import Path

# Windows cp949 인코딩 문제 해결
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
import ta

from src.core.models import SignalType
from src.exchange.futures_client import FuturesClient
from src.strategies.registry import get_strategy, list_strategies


# ─── 시뮬레이션 엔진 ─────────────────────────────────────────

def simulate(strategy, symbol, candles, htf_candles, capital=200.0, leverage=7):
    """단일 전략 시뮬레이션. 타임프레임 무관."""
    sl_mult = getattr(strategy, "SL_ATR_MULT", 8.0)
    tp_mult = getattr(strategy, "TP_ATR_MULT", 12.0)

    atr_series = ta.volatility.AverageTrueRange(
        candles["high"], candles["low"], candles["close"], window=14
    ).average_true_range()

    balance = capital
    position = None
    trades = []
    equity = [capital]
    warmup = 70  # 지표 안정화

    for i in range(warmup, len(candles)):
        window = candles.iloc[:i + 1]
        price = float(window.iloc[-1]["close"])
        atr = float(atr_series.iloc[i]) if i < len(atr_series) and not pd.isna(atr_series.iloc[i]) else 0

        if atr <= 0:
            equity.append(balance)
            continue

        # 포지션 SL/TP 체크
        if position:
            p = position
            ea = p["entry_atr"]
            sl_d = ea * sl_mult
            tp_d = ea * tp_mult

            if p["side"] == "LONG":
                sl_p = p["entry"] - sl_d
                tp_p = p["entry"] + tp_d
            else:
                sl_p = p["entry"] + sl_d
                tp_p = p["entry"] - tp_d

            hit_sl = (p["side"] == "LONG" and price <= sl_p) or \
                     (p["side"] == "SHORT" and price >= sl_p)
            hit_tp = (p["side"] == "LONG" and price >= tp_p) or \
                     (p["side"] == "SHORT" and price <= tp_p)

            # CLOSE 시그널 체크
            htf_w = htf_candles if htf_candles is not None and not htf_candles.empty else None
            sig = strategy.evaluate(symbol, window, htf_w)
            hit_close = sig.type == SignalType.CLOSE

            if hit_sl or hit_tp or hit_close:
                if hit_sl:
                    exit_p = sl_p
                    reason = "SL"
                elif hit_tp:
                    exit_p = tp_p
                    reason = "TP"
                else:
                    exit_p = price
                    reason = "CLOSE"

                if p["side"] == "LONG":
                    pnl = (exit_p - p["entry"]) * p["qty"]
                else:
                    pnl = (p["entry"] - exit_p) * p["qty"]
                fee = p["entry"] * p["qty"] * 0.0008
                net = pnl - fee
                balance += net
                trades.append({
                    "side": p["side"], "entry": p["entry"], "exit": exit_p,
                    "pnl": round(net, 4), "reason": reason,
                    "entry_time": candles.index[p["entry_idx"]],
                    "exit_time": candles.index[i],
                    "duration_min": i - p["entry_idx"],
                })
                strategy.record_result(net)
                position = None

        # 진입
        if not position:
            htf_w = htf_candles if htf_candles is not None and not htf_candles.empty else None
            sig = strategy.evaluate(symbol, window, htf_w)

            if sig.type in (SignalType.BUY, SignalType.SELL):
                invest = balance * 0.3
                qty = (invest * leverage) / price
                if invest >= 5:
                    side = "LONG" if sig.type == SignalType.BUY else "SHORT"
                    position = {
                        "side": side, "entry": price,
                        "qty": qty, "entry_atr": atr,
                        "entry_idx": i,
                    }

        # equity
        if position:
            p = position
            if p["side"] == "LONG":
                unrealized = (price - p["entry"]) * p["qty"]
            else:
                unrealized = (p["entry"] - price) * p["qty"]
            equity.append(balance + unrealized)
        else:
            equity.append(balance)

    # 미청산 포지션 강제 종료
    if position:
        p = position
        final_price = float(candles.iloc[-1]["close"])
        if p["side"] == "LONG":
            pnl = (final_price - p["entry"]) * p["qty"]
        else:
            pnl = (p["entry"] - final_price) * p["qty"]
        fee = p["entry"] * p["qty"] * 0.0008
        net = pnl - fee
        balance += net
        trades.append({
            "side": p["side"], "entry": p["entry"], "exit": final_price,
            "pnl": round(net, 4), "reason": "END",
            "entry_time": candles.index[p["entry_idx"]],
            "exit_time": candles.index[-1],
            "duration_min": len(candles) - 1 - p["entry_idx"],
        })

    return {
        "balance": balance, "trades": trades, "equity": equity,
    }


# ─── 리포트 출력 ─────────────────────────────────────────────

def print_report(all_results, capital):
    """전체 백테스트 리포트 출력."""
    SEP = "=" * 90

    print()
    print(SEP)
    print("  전체 전략 백테스트 리포트")
    print(SEP)
    print()

    # ── 전략×심볼 요약 테이블 ──
    header = (f"{'전략':<32s} {'심볼':<10s} {'거래':>4s} {'승':>3s} {'패':>3s} "
              f"{'승률':>6s} {'순손익':>10s} {'ROI':>7s} {'MDD':>7s} {'TP':>3s} {'SL':>3s} "
              f"{'평균수익':>9s} {'평균손실':>9s}")
    print(header)
    print("-" * 90)

    strategy_totals = {}

    for r in all_results:
        sname = r["strategy_label"]
        sym = r["symbol"]
        trades = r["trades"]
        balance = r["balance"]
        equity = r["equity"]

        n = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        nw = len(wins)
        nl = len(losses)
        wr = nw / n * 100 if n > 0 else 0
        pnl = balance - capital
        roi = pnl / capital * 100
        tp_count = sum(1 for t in trades if t["reason"] == "TP")
        sl_count = sum(1 for t in trades if t["reason"] == "SL")
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0

        # MDD
        peak = equity[0]
        max_dd = 0
        for eq in equity:
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        mdd = max_dd * 100

        print(f"{sname:<32s} {sym:<10s} {n:>4d} {nw:>3d} {nl:>3d} "
              f"{wr:>5.1f}% {pnl:>+10.2f} {roi:>+6.1f}% {mdd:>6.1f}% {tp_count:>3d} {sl_count:>3d} "
              f"{avg_win:>+9.4f} {avg_loss:>+9.4f}")

        # 전략별 합산
        if sname not in strategy_totals:
            strategy_totals[sname] = {
                "trades": [], "pnl": 0, "equity_all": [],
                "symbols": 0,
            }
        strategy_totals[sname]["trades"].extend(trades)
        strategy_totals[sname]["pnl"] += pnl
        strategy_totals[sname]["equity_all"].extend(equity)
        strategy_totals[sname]["symbols"] += 1

    # ── 전략별 종합 요약 ──
    print()
    print(SEP)
    print("  전략별 종합 (전체 심볼 합산)")
    print(SEP)
    print()

    header2 = (f"{'전략':<32s} {'심볼수':>4s} {'거래':>4s} {'승률':>6s} "
               f"{'총손익':>12s} {'총ROI':>8s} {'평균거래':>10s} {'수익팩터':>8s}")
    print(header2)
    print("-" * 90)

    ranked = []
    for sname, data in strategy_totals.items():
        trades = data["trades"]
        n = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        wr = len(wins) / n * 100 if n > 0 else 0
        total_pnl = data["pnl"]
        total_roi = total_pnl / (capital * data["symbols"]) * 100
        avg_trade = np.mean([t["pnl"] for t in trades]) if trades else 0

        # profit factor
        gross_win = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.001
        pf = gross_win / gross_loss if gross_loss > 0 else 0

        print(f"{sname:<32s} {data['symbols']:>4d} {n:>4d} {wr:>5.1f}% "
              f"{total_pnl:>+12.2f} {total_roi:>+7.1f}% {avg_trade:>+10.4f} {pf:>8.2f}")

        ranked.append((sname, total_roi, wr, n, pf))

    # ── 랭킹 ──
    print()
    print(SEP)
    print("  전략 랭킹 (ROI 기준)")
    print(SEP)
    print()

    ranked.sort(key=lambda x: x[1], reverse=True)
    for i, (name, roi, wr, n, pf) in enumerate(ranked, 1):
        medal = {1: "[1st]", 2: "[2nd]", 3: "[3rd]"}.get(i, f"[{i}th]")
        status = "PROFIT" if roi > 0 else "LOSS" if n > 0 else "NO TRADE"
        print(f"  {medal:>5s}  {name:<32s}  ROI {roi:>+7.1f}%  승률 {wr:>5.1f}%  "
              f"거래 {n:>3d}건  PF {pf:.2f}  {status}")

    # ── 거래 상세 (상위 전략) ──
    if ranked and ranked[0][3] > 0:
        best = ranked[0][0]
        best_trades = strategy_totals[best]["trades"]
        print()
        print(SEP)
        print(f"  최고 전략 [{best}] 최근 거래 상세")
        print(SEP)
        print()
        print(f"  {'방향':<6s} {'진입가':>12s} {'청산가':>12s} {'손익':>10s} {'사유':<6s} {'시간':>6s}")
        print("  " + "-" * 60)
        for t in best_trades[-20:]:
            pnl_mark = "+" if t["pnl"] > 0 else ""
            print(f"  {t['side']:<6s} {t['entry']:>12.2f} {t['exit']:>12.2f} "
                  f"{pnl_mark}{t['pnl']:>9.4f} {t['reason']:<6s} {t['duration_min']:>4d}분")

    print()
    print(SEP)
    print("  리포트 종료")
    print(SEP)
    print()


# ─── 메인 ────────────────────────────────────────────────────

async def fetch_data(symbols, candle_count, interval):
    """바이낸스에서 멀티 심볼 캔들 데이터 가져오기."""
    # HTF 매핑: 주 타임프레임 → 상위 타임프레임
    HTF_MAP = {"1m": "15m", "5m": "1h", "15m": "1h", "1h": "4h", "1d": "1w"}
    htf_interval = HTF_MAP.get(interval, "1h")

    client = FuturesClient()
    await client.connect()
    try:
        data = {}
        for sym in symbols:
            print(f"  {sym} 데이터 로딩...", end=" ", flush=True)
            try:
                candles = await client.get_candles(sym, interval=interval, limit=candle_count)
                htf = await client.get_candles(sym, interval=htf_interval, limit=500)
                data[sym] = {"candles": candles, "htf": htf}
                print(f"{interval}={len(candles)}개, {htf_interval}={len(htf)}개")
            except Exception as e:
                print(f"실패: {e}")
                data[sym] = {"candles": pd.DataFrame(), "htf": pd.DataFrame()}
        return data
    finally:
        await client.disconnect()


# 타임프레임별 시간 환산
TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def main():
    parser = argparse.ArgumentParser(description="전체 전략 백테스트")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT",
                        help="심볼 목록 (콤마 구분)")
    parser.add_argument("--candles", type=int, default=1000,
                        help="캔들 수 (기본 1000)")
    parser.add_argument("--interval", default="15m",
                        help="타임프레임 (기본 15m). 1m/5m/15m/1h/1d")
    parser.add_argument("--capital", type=float, default=200.0,
                        help="초기 자본 (기본 $200)")
    parser.add_argument("--leverage", type=int, default=7,
                        help="레버리지 (기본 7)")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    strategies = list_strategies()
    tf_min = TF_MINUTES.get(args.interval, 15)
    hours = args.candles * tf_min / 60
    days = hours / 24

    print()
    print("=" * 90)
    print("  백테스트 설정")
    print("=" * 90)
    print(f"  심볼: {', '.join(symbols)}")
    print(f"  캔들: {args.candles}개 ({args.interval}봉 = ~{hours:.0f}시간 = ~{days:.1f}일)")
    print(f"  자본: ${args.capital:.0f} x{args.leverage} 레버리지")
    print(f"  전략: {len(strategies)}개")
    for s in strategies:
        print(f"    - {s['label']}")
    print()

    # 데이터 로드
    print("데이터 로딩 중...")
    data = asyncio.run(fetch_data(symbols, args.candles, args.interval))

    # 백테스트 실행
    print()
    print("백테스트 실행 중...")
    all_results = []

    for s_info in strategies:
        sname = s_info["name"]
        slabel = s_info["label"]

        for sym in symbols:
            candles = data[sym]["candles"]
            htf = data[sym]["htf"]

            if candles.empty or len(candles) < 100:
                print(f"  {slabel} / {sym}: 데이터 부족, 스킵")
                continue

            # 각 심볼×전략마다 독립 인스턴스
            strategy = get_strategy(sname)
            print(f"  {slabel:<35s} / {sym:<10s}", end=" ", flush=True)

            result = simulate(strategy, sym, candles, htf,
                              capital=args.capital, leverage=args.leverage)

            n_trades = len(result["trades"])
            pnl = result["balance"] - args.capital
            print(f"→ {n_trades:>3d}건, ${pnl:>+8.2f}")

            all_results.append({
                "strategy_name": sname,
                "strategy_label": slabel,
                "symbol": sym,
                **result,
            })

    # 리포트 출력
    print_report(all_results, args.capital)


if __name__ == "__main__":
    main()
