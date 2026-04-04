"""실거래 + 가상매매 종합 성과 분석 리포트.

Usage:
    python scripts/real_report.py                    # 로컬 DB
    python scripts/real_report.py data/trades_real_server.db  # 서버 DB
"""
import io
import sqlite3
import sys

# Windows 콘솔 인코딩 문제 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ─── DB 연결 ────────────────────────────────────────────

def connect(db_path: str | None = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = str(Path(__file__).parent.parent / "data" / "trades_real.db")
    if not Path(db_path).exists():
        print(f"DB not found: {db_path}")
        sys.exit(1)
    return sqlite3.connect(db_path)


def has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    c = conn.execute(f"PRAGMA table_info({table})")
    return col in [r[1] for r in c.fetchall()]


def has_table(conn: sqlite3.Connection, table: str) -> bool:
    c = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return table in [r[0] for r in c.fetchall()]


# ─── 6.1: 실거래 성과 종합 ──────────────────────────────

def report_real_trades(conn: sqlite3.Connection) -> list[dict]:
    has_reason = has_column(conn, "trades", "reason")
    has_fee = has_column(conn, "trades", "fee")
    has_sl = has_column(conn, "trades", "sl_price")

    pnl_col = "COALESCE(net_pnl, pnl, 0)" if has_fee else "COALESCE(pnl, 0)"

    rows = conn.execute(f"""
        SELECT id, symbol, side, entry_price, exit_price, quantity,
               pnl, pnl_pct, {f'fee, net_pnl,' if has_fee else ''} strategy,
               {'reason,' if has_reason else ''}
               {'sl_price, tp_price,' if has_sl else ''}
               opened_at, closed_at
        FROM trades ORDER BY opened_at
    """).fetchall()

    cols = [d[0] for d in conn.execute(f"""
        SELECT id, symbol, side, entry_price, exit_price, quantity,
               pnl, pnl_pct, {f'fee, net_pnl,' if has_fee else ''} strategy,
               {'reason,' if has_reason else ''}
               {'sl_price, tp_price,' if has_sl else ''}
               opened_at, closed_at
        FROM trades LIMIT 0
    """).description]

    trades = [dict(zip(cols, r)) for r in rows]

    if not trades:
        print("  (실거래 데이터 없음)")
        return []

    print(f"\n  총 {len(trades)}건 | "
          f"{trades[0]['opened_at'][:10]} ~ {trades[-1].get('closed_at', trades[-1]['opened_at'])[:10]}")

    # 전략별 요약
    by_strat: dict[str, list] = defaultdict(list)
    for t in trades:
        by_strat[t["strategy"] or "unknown"].append(t)

    print(f"\n  {'전략':<30s} {'거래':>4s} {'승':>3s} {'패':>3s} {'승률':>6s} "
          f"{'총PnL':>10s} {'평균':>8s} {'PF':>6s} {'MDD':>8s}")
    print("  " + "-" * 90)

    for strat, strades in sorted(by_strat.items()):
        _print_strategy_summary(strades, strat, pnl_key="net_pnl" if has_fee else "pnl")

    return trades


def _print_strategy_summary(trades: list[dict], name: str, pnl_key: str = "net_pnl"):
    pnls = [t.get(pnl_key) or t.get("pnl") or 0 for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    total = sum(pnls)
    avg = total / len(pnls) if pnls else 0
    wr = wins / len(pnls) * 100 if pnls else 0

    # Profit Factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # MDD (Maximum Drawdown)
    equity = 0
    peak = 0
    mdd = 0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        dd = peak - equity
        mdd = max(mdd, dd)

    print(f"  {name:<30s} {len(trades):>4d} {wins:>3d} {losses:>3d} {wr:>5.1f}% "
          f"{total:>+10.4f} {avg:>+8.4f} {pf:>6.2f} {mdd:>8.4f}")


# ─── 6.3: 손실 패턴 분석 ────────────────────────────────

def report_loss_patterns(trades: list[dict], pnl_key: str = "net_pnl"):
    if not trades:
        return

    print(f"\n  {'시간(KST)':<10s} {'거래':>4s} {'승':>3s} {'패':>3s} {'승률':>6s} {'PnL':>10s}")
    print("  " + "-" * 50)

    by_hour: dict[int, list] = defaultdict(list)
    for t in trades:
        opened = t.get("opened_at", "")
        if opened:
            try:
                h = int(opened[11:13])
            except (ValueError, IndexError):
                h = -1
            by_hour[h].append(t)

    for h in sorted(by_hour.keys()):
        ht = by_hour[h]
        pnls = [t.get(pnl_key) or t.get("pnl") or 0 for t in ht]
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100 if pnls else 0
        total = sum(pnls)
        marker = " << 위험" if wr < 40 and len(pnls) >= 3 else ""
        print(f"  {h:>2d}시        {len(pnls):>4d} {wins:>3d} {len(pnls)-wins:>3d} "
              f"{wr:>5.1f}% {total:>+10.4f}{marker}")

    # 방향별
    print(f"\n  {'방향':<10s} {'거래':>4s} {'승':>3s} {'패':>3s} {'승률':>6s} {'PnL':>10s}")
    print("  " + "-" * 50)
    for side in ["LONG", "SHORT"]:
        st = [t for t in trades if t.get("side") == side]
        if not st:
            continue
        pnls = [t.get(pnl_key) or t.get("pnl") or 0 for t in st]
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100 if pnls else 0
        print(f"  {side:<10s} {len(st):>4d} {wins:>3d} {len(st)-wins:>3d} "
              f"{wr:>5.1f}% {sum(pnls):>+10.4f}")

    # 연속 손실
    print("\n  ■ 연속 손실 구간:")
    pnls = [t.get(pnl_key) or t.get("pnl") or 0 for t in trades]
    streak = 0
    max_streak = 0
    streak_start = 0
    streaks = []
    for i, p in enumerate(pnls):
        if p <= 0:
            if streak == 0:
                streak_start = i
            streak += 1
        else:
            if streak >= 3:
                streaks.append((streak_start, streak, sum(pnls[streak_start:streak_start + streak])))
            max_streak = max(max_streak, streak)
            streak = 0
    if streak >= 3:
        streaks.append((streak_start, streak, sum(pnls[streak_start:streak_start + streak])))
    max_streak = max(max_streak, streak)

    if streaks:
        for start, length, loss in streaks:
            t = trades[start]
            print(f"    {length}연패  pnl={loss:>+.4f}  시작={t.get('opened_at', '?')[:16]}")
    print(f"    최대 연속 손실: {max_streak}연패")


# ─── 6.4: SL/TP 적중률 분석 ─────────────────────────────

def report_sltp_analysis(trades: list[dict], pnl_key: str = "net_pnl"):
    if not trades:
        return

    has_reason = any(t.get("reason") for t in trades)

    if has_reason:
        print(f"\n  {'청산사유':<15s} {'건수':>4s} {'총PnL':>10s} {'평균PnL':>8s} {'비율':>6s}")
        print("  " + "-" * 50)

        by_reason: dict[str, list] = defaultdict(list)
        for t in trades:
            reason = t.get("reason") or "unknown"
            by_reason[reason].append(t)

        for reason, rt in sorted(by_reason.items(), key=lambda x: -len(x[1])):
            pnls = [t.get(pnl_key) or t.get("pnl") or 0 for t in rt]
            pct = len(rt) / len(trades) * 100
            print(f"  {reason:<15s} {len(rt):>4d} {sum(pnls):>+10.4f} "
                  f"{sum(pnls)/len(pnls):>+8.4f} {pct:>5.1f}%")
    else:
        print("  (reason 컬럼 없음 -청산사유 분석 불가)")

    # SL/TP 가격 분석
    has_sl = any(t.get("sl_price") for t in trades)
    if has_sl:
        sl_trades = [t for t in trades if t.get("sl_price")]
        tp_trades = [t for t in trades if t.get("tp_price")]

        print(f"\n  SL 설정 거래: {len(sl_trades)}/{len(trades)}")
        print(f"  TP 설정 거래: {len(tp_trades)}/{len(trades)}")

        if sl_trades:
            # SL 거리 분석
            sl_dists = []
            for t in sl_trades:
                entry = t["entry_price"]
                sl = t["sl_price"]
                dist_pct = abs(sl - entry) / entry * 100
                sl_dists.append(dist_pct)
            print(f"  SL 평균 거리: {sum(sl_dists)/len(sl_dists):.3f}%")

        if tp_trades:
            tp_dists = []
            for t in tp_trades:
                entry = t["entry_price"]
                tp = t["tp_price"]
                dist_pct = abs(tp - entry) / entry * 100
                tp_dists.append(dist_pct)
            print(f"  TP 평균 거리: {sum(tp_dists)/len(tp_dists):.3f}%")
    else:
        print("  (sl_price/tp_price 컬럼 없음)")

    # 보유 시간 분석
    hold_times = []
    for t in trades:
        opened = t.get("opened_at", "")
        closed = t.get("closed_at", "")
        if opened and closed:
            try:
                fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in opened else "%Y-%m-%d %H:%M:%S"
                o = datetime.strptime(opened, fmt)
                fmt_c = "%Y-%m-%d %H:%M:%S.%f" if "." in closed else "%Y-%m-%d %H:%M:%S"
                c = datetime.strptime(closed, fmt_c)
                hold_times.append((c - o).total_seconds())
            except ValueError:
                pass

    if hold_times:
        avg_hold = sum(hold_times) / len(hold_times)
        max_hold = max(hold_times)
        min_hold = min(hold_times)

        def fmt_duration(s: float) -> str:
            if s < 60:
                return f"{s:.0f}초"
            if s < 3600:
                return f"{s/60:.1f}분"
            return f"{s/3600:.1f}시간"

        print(f"\n  ■ 보유 시간")
        print(f"    평균: {fmt_duration(avg_hold)}")
        print(f"    최단: {fmt_duration(min_hold)}")
        print(f"    최장: {fmt_duration(max_hold)}")

        # 보유시간 vs 수익 상관
        wins_hold = [h for h, t in zip(hold_times, trades) if (t.get(pnl_key) or t.get("pnl") or 0) > 0]
        loss_hold = [h for h, t in zip(hold_times, trades) if (t.get(pnl_key) or t.get("pnl") or 0) <= 0]
        if wins_hold and loss_hold:
            print(f"    수익 거래 평균: {fmt_duration(sum(wins_hold)/len(wins_hold))}")
            print(f"    손실 거래 평균: {fmt_duration(sum(loss_hold)/len(loss_hold))}")


# ─── 6.2: 가상매매 전략 비교 ────────────────────────────

def report_paper_trades(conn: sqlite3.Connection):
    if not has_table(conn, "paper_trades"):
        print("  (paper_trades 테이블 없음)")
        return []

    cnt = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    if cnt == 0:
        print("  (가상매매 데이터 없음)")
        return []

    has_reason = has_column(conn, "paper_trades", "reason")
    has_sl = has_column(conn, "paper_trades", "sl_price")

    rows = conn.execute("SELECT * FROM paper_trades ORDER BY opened_at").fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM paper_trades LIMIT 0").description]
    trades = [dict(zip(cols, r)) for r in rows]

    print(f"\n  총 {len(trades)}건")

    by_strat: dict[str, list] = defaultdict(list)
    for t in trades:
        by_strat[t["strategy"]].append(t)

    print(f"\n  {'전략':<30s} {'거래':>4s} {'승':>3s} {'패':>3s} {'승률':>6s} "
          f"{'총PnL':>10s} {'평균':>8s} {'PF':>6s}")
    print("  " + "-" * 82)

    for strat in sorted(by_strat.keys()):
        _print_strategy_summary(by_strat[strat], strat)

    # 잔고 테이블
    if has_table(conn, "paper_balances"):
        bals = conn.execute(
            "SELECT * FROM paper_balances ORDER BY balance DESC"
        ).fetchall()
        if bals:
            bcols = [d[0] for d in conn.execute("SELECT * FROM paper_balances LIMIT 0").description]
            print(f"\n  {'전략':<30s} {'잔고':>10s} {'초기':>10s} {'ROI':>7s}")
            print("  " + "-" * 62)
            for b in bals:
                d = dict(zip(bcols, b))
                roi = (d["balance"] - d["initial_balance"]) / d["initial_balance"] * 100
                print(f"  {d['strategy']:<30s} {d['balance']:>10.2f} "
                      f"{d['initial_balance']:>10.2f} {roi:>+6.1f}%")

    return trades


# ─── 메인 ───────────────────────────────────────────────

def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    conn = connect(db_path)

    print("=" * 70)
    print("  Auto-Trader 종합 성과 분석 리포트")
    print(f"  DB: {db_path or 'data/trades_real.db'}")
    print(f"  생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    has_fee = has_column(conn, "trades", "fee")
    pnl_key = "net_pnl" if has_fee else "pnl"

    # ── 6.1: 실거래 성과 종합
    print("\n" + "━" * 70)
    print("  [6.1] 실거래 성과 종합")
    print("━" * 70)
    real_trades = report_real_trades(conn)

    # ── 6.2: 가상매매 전략 비교
    print("\n" + "━" * 70)
    print("  [6.2] 가상매매 전략 비교")
    print("━" * 70)
    paper_trades = report_paper_trades(conn)

    # ── 6.3: 손실 패턴 분석
    all_trades = real_trades + paper_trades
    if all_trades:
        print("\n" + "━" * 70)
        print("  [6.3] 손실 패턴 분석 (실거래)")
        print("━" * 70)
        if real_trades:
            report_loss_patterns(real_trades, pnl_key)
        if paper_trades:
            print("\n  [6.3b] 손실 패턴 분석 (가상매매)")
            print("  " + "-" * 50)
            # 전략별 분석
            by_strat: dict[str, list] = defaultdict(list)
            for t in paper_trades:
                by_strat[t["strategy"]].append(t)
            for strat, st in sorted(by_strat.items()):
                if len(st) >= 5:
                    print(f"\n  > {strat}")
                    report_loss_patterns(st)

    # ── 6.4: SL/TP 적중률
    if all_trades:
        print("\n" + "━" * 70)
        print("  [6.4] SL/TP & 보유시간 분석 (실거래)")
        print("━" * 70)
        if real_trades:
            report_sltp_analysis(real_trades, pnl_key)
        if paper_trades:
            print("\n  [6.4b] SL/TP & 보유시간 분석 (가상매매)")
            print("  " + "-" * 50)
            report_sltp_analysis(paper_trades)

    # ── 결론
    print("\n" + "━" * 70)
    print("  [결론] 개선 포인트")
    print("━" * 70)

    if real_trades:
        pnls = [t.get(pnl_key) or t.get("pnl") or 0 for t in real_trades]
        total_pnl = sum(pnls)
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100

        print(f"\n  실거래 총 PnL: {total_pnl:+.4f} USDT")
        print(f"  승률: {wr:.1f}% ({wins}/{len(pnls)})")

        if wr < 50:
            print("  ⚠ 승률 50% 미만 -진입 조건 강화 필요")
        if total_pnl < 0:
            print("  ⚠ 순손실 -SL/TP 배수 또는 진입 타이밍 개선 필요")
    else:
        print("\n  ⚠ 실거래 데이터 부족 -봇 정상 가동 후 재분석 필요")

    print()
    conn.close()


if __name__ == "__main__":
    main()
