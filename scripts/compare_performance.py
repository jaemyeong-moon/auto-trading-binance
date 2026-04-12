"""Phase 8.2: 개선 전/후 실거래 + 가상매매 성과 비교 분석.

Phase 7 전략 개선 → Phase 8.1 서버 배포(2026-04-05) 전후 성과를 비교.

Usage:
    python scripts/compare_performance.py                          # 기본 서버 DB
    python scripts/compare_performance.py data/trades_real_server.db
    python scripts/compare_performance.py --cutoff 2026-04-05      # 기준일 변경
"""
import io
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CUTOFF_DATE = "2026-04-05"  # Phase 8.1 배포일


def connect(db_path: str | None = None) -> sqlite3.Connection:
    if db_path is None:
        db_path = str(Path(__file__).parent.parent / "data" / "trades_real_server.db")
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


def calc_metrics(trades: list[dict], pnl_key: str = "net_pnl") -> dict:
    """거래 리스트에서 핵심 지표 계산."""
    if not trades:
        return {
            "count": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl": 0, "avg_pnl": 0, "pf": 0, "mdd": 0,
            "avg_hold_min": 0,
        }

    pnls = [t.get(pnl_key) or t.get("pnl") or 0 for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    total = sum(pnls)
    avg = total / len(pnls) if pnls else 0
    wr = wins / len(pnls) * 100 if pnls else 0

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # MDD
    equity = 0
    peak = 0
    mdd = 0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        dd = peak - equity
        mdd = max(mdd, dd)

    # 평균 보유 시간 (분)
    hold_times = []
    for t in trades:
        opened = t.get("opened_at", "")
        closed = t.get("closed_at", "")
        if opened and closed:
            try:
                fmt_o = "%Y-%m-%d %H:%M:%S.%f" if "." in opened else "%Y-%m-%d %H:%M:%S"
                fmt_c = "%Y-%m-%d %H:%M:%S.%f" if "." in closed else "%Y-%m-%d %H:%M:%S"
                o = datetime.strptime(opened, fmt_o)
                c = datetime.strptime(closed, fmt_c)
                hold_times.append((c - o).total_seconds() / 60)
            except ValueError:
                pass

    avg_hold = max(0, sum(hold_times) / len(hold_times)) if hold_times else 0

    return {
        "count": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": wr,
        "total_pnl": total,
        "avg_pnl": avg,
        "pf": pf,
        "mdd": mdd,
        "avg_hold_min": avg_hold,
    }


def print_comparison_table(before: dict, after: dict, label: str):
    """Before/After 비교 테이블 출력."""
    def delta(key: str, fmt: str = "+.2f", higher_better: bool = True) -> str:
        b, a = before[key], after[key]
        if b == 0 and a == 0:
            return "-"
        if b == float("inf") or a == float("inf"):
            return "N/A"
        diff = a - b
        arrow = "^" if (diff > 0) == higher_better else "v"
        if diff == 0:
            arrow = "="
        return f"{diff:{fmt}} {arrow}"

    print(f"\n  {'지표':<18s} {'개선 전':>12s} {'개선 후':>12s} {'변화':>14s}")
    print("  " + "-" * 60)
    print(f"  {'거래 수':<18s} {before['count']:>12d} {after['count']:>12d} {delta('count', '+d'):>14s}")
    print(f"  {'승':<18s} {before['wins']:>12d} {after['wins']:>12d} {delta('wins', '+d'):>14s}")
    print(f"  {'패':<18s} {before['losses']:>12d} {after['losses']:>12d} {delta('losses', '+d', False):>14s}")
    print(f"  {'승률 (%)':<18s} {before['win_rate']:>11.1f}% {after['win_rate']:>11.1f}% {delta('win_rate', '+.1f'):>14s}")
    print(f"  {'총 PnL (USDT)':<18s} {before['total_pnl']:>+12.4f} {after['total_pnl']:>+12.4f} {delta('total_pnl'):>14s}")
    print(f"  {'평균 PnL':<18s} {before['avg_pnl']:>+12.4f} {after['avg_pnl']:>+12.4f} {delta('avg_pnl'):>14s}")
    def fmt_pf(v):
        return f"{'inf':>12s}" if v == float("inf") else f"{v:>12.2f}"
    print(f"  {'Profit Factor':<18s} {fmt_pf(before['pf'])} {fmt_pf(after['pf'])} {delta('pf'):>14s}")
    print(f"  {'MDD (USDT)':<18s} {before['mdd']:>12.4f} {after['mdd']:>12.4f} {delta('mdd', '+.4f', False):>14s}")
    print(f"  {'평균보유 (분)':<18s} {before['avg_hold_min']:>12.1f} {after['avg_hold_min']:>12.1f} {delta('avg_hold_min', '+.1f', False):>14s}")


def load_trades(conn: sqlite3.Connection, table: str) -> list[dict]:
    """테이블에서 전체 거래 로드."""
    if not has_table(conn, table):
        return []
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY opened_at").fetchall()
    cols = [d[0] for d in conn.execute(f"SELECT * FROM {table} LIMIT 0").description]
    return [dict(zip(cols, r)) for r in rows]


def split_by_cutoff(trades: list[dict], cutoff: str) -> tuple[list[dict], list[dict]]:
    """기준일 전후로 거래 분리."""
    before, after = [], []
    for t in trades:
        opened = t.get("opened_at", "")
        if isinstance(opened, datetime):
            opened = opened.isoformat()
        if opened and opened[:10] >= cutoff:
            after.append(t)
        else:
            before.append(t)
    return before, after


def report_by_reason(trades: list[dict], pnl_key: str = "net_pnl"):
    """청산사유별 분석."""
    if not trades or not any(t.get("reason") for t in trades):
        return

    by_reason: dict[str, list] = defaultdict(list)
    for t in trades:
        reason = t.get("reason") or "unknown"
        by_reason[reason].append(t)

    print(f"\n  {'청산사유':<15s} {'건수':>4s} {'승':>3s} {'패':>3s} {'승률':>6s} {'PnL':>10s}")
    print("  " + "-" * 45)
    for reason, rt in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        pnls = [t.get(pnl_key) or t.get("pnl") or 0 for t in rt]
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(rt) * 100
        print(f"  {reason:<15s} {len(rt):>4d} {wins:>3d} {len(rt)-wins:>3d} "
              f"{wr:>5.1f}% {sum(pnls):>+10.4f}")


def main():
    # 인자 파싱
    db_path = None
    cutoff = CUTOFF_DATE
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--cutoff" and i + 1 < len(args):
            cutoff = args[i + 1]
        elif not arg.startswith("--"):
            db_path = arg

    conn = connect(db_path)
    has_fee = has_column(conn, "trades", "fee") if has_table(conn, "trades") else False
    pnl_key = "net_pnl" if has_fee else "pnl"

    print("=" * 70)
    print("  Phase 8.2: 개선 전/후 성과 비교 분석")
    print(f"  DB: {db_path or 'data/trades_real_server.db'}")
    print(f"  기준일: {cutoff} (Phase 8.1 배포)")
    print(f"  생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # ── 실거래 비교
    real_trades = load_trades(conn, "trades")
    real_before, real_after = split_by_cutoff(real_trades, cutoff)

    print("\n" + "━" * 70)
    print(f"  [1] 실거래 성과 비교 (기준: {cutoff})")
    print("━" * 70)
    print(f"  개선 전: {len(real_before)}건 | 개선 후: {len(real_after)}건")

    if real_before or real_after:
        m_before = calc_metrics(real_before, pnl_key)
        m_after = calc_metrics(real_after, pnl_key)
        print_comparison_table(m_before, m_after, "실거래")

        if real_after:
            print("\n  ■ 개선 후 청산사유 분석")
            report_by_reason(real_after, pnl_key)
    else:
        print("  (실거래 데이터 없음)")

    # ── 가상매매 비교
    paper_trades = load_trades(conn, "paper_trades")
    paper_before, paper_after = split_by_cutoff(paper_trades, cutoff)

    print("\n" + "━" * 70)
    print(f"  [2] 가상매매 성과 비교 (기준: {cutoff})")
    print("━" * 70)
    print(f"  개선 전: {len(paper_before)}건 | 개선 후: {len(paper_after)}건")

    if paper_before or paper_after:
        m_before = calc_metrics(paper_before, pnl_key)
        m_after = calc_metrics(paper_after, pnl_key)
        print_comparison_table(m_before, m_after, "가상매매")

    # ── 전략별 비교 (가상매매)
    if paper_before or paper_after:
        print("\n" + "━" * 70)
        print("  [3] 전략별 성과 비교 (가상매매)")
        print("━" * 70)

        all_strategies = set()
        for t in paper_trades:
            all_strategies.add(t["strategy"])

        print(f"\n  {'전략':<28s} │ {'개선 전':^28s} │ {'개선 후':^28s}")
        print(f"  {'':<28s} │ {'건수':>4s} {'승률':>6s} {'PnL':>9s} {'PF':>6s} │ "
              f"{'건수':>4s} {'승률':>6s} {'PnL':>9s} {'PF':>6s}")
        print("  " + "-" * 88)

        for strat in sorted(all_strategies):
            sb = [t for t in paper_before if t["strategy"] == strat]
            sa = [t for t in paper_after if t["strategy"] == strat]
            mb = calc_metrics(sb, pnl_key)
            ma = calc_metrics(sa, pnl_key)

            def fmt(m):
                if m["count"] == 0:
                    return f"{'--':>4s} {'--':>6s} {'--':>9s} {'--':>6s}"
                return (f"{m['count']:>4d} {m['win_rate']:>5.1f}% "
                        f"{m['total_pnl']:>+9.2f} {m['pf']:>6.2f}")

            print(f"  {strat:<28s} │ {fmt(mb)} │ {fmt(ma)}")

    # ── 잔고 현황
    if has_table(conn, "paper_balances"):
        bals = conn.execute(
            "SELECT strategy, balance, initial_balance FROM paper_balances ORDER BY balance DESC"
        ).fetchall()
        if bals:
            print("\n" + "━" * 70)
            print("  [4] 가상매매 잔고 현황")
            print("━" * 70)
            print(f"\n  {'전략':<28s} {'잔고':>10s} {'초기':>10s} {'ROI':>8s}")
            print("  " + "-" * 60)
            for b in bals:
                roi = (b[1] - b[2]) / b[2] * 100
                marker = " << 수익" if roi > 0 else (" << 손실" if roi < -10 else "")
                print(f"  {b[0]:<28s} {b[1]:>10.2f} {b[2]:>10.2f} {roi:>+7.1f}%{marker}")

    # ── 데이터 품질 경고
    print("\n" + "━" * 70)
    print("  [5] 데이터 품질 & 운영 이슈")
    print("━" * 70)

    warnings = []

    if len(real_after) < 20:
        warnings.append(f"실거래 개선 후 데이터 부족: {len(real_after)}건 (통계적 유의성 낮음)")
    if len(real_before) < 5:
        warnings.append(f"실거래 개선 전 데이터 부족: {len(real_before)}건")
    if len(paper_after) < 10:
        warnings.append(f"가상매매 개선 후 데이터 부족: {len(paper_after)}건")

    # 봇 동결 감지 — 마지막 거래 이후 24시간 이상 경과
    if real_trades:
        last_close = real_trades[-1].get("closed_at", "")
        if last_close:
            try:
                fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in last_close else "%Y-%m-%d %H:%M:%S"
                last_dt = datetime.strptime(last_close, fmt)
                gap_hours = (datetime.now() - last_dt).total_seconds() / 3600
                if gap_hours > 24:
                    warnings.append(
                        f"봇 동결 의심: 마지막 거래 {last_close[:16]} 이후 "
                        f"{gap_hours:.0f}시간 경과 (약 {gap_hours/24:.1f}일)"
                    )
            except ValueError:
                pass

    # 오픈 포지션 장기 방치
    if has_table(conn, "positions"):
        positions = conn.execute("SELECT symbol, side, entry_price, opened_at FROM positions").fetchall()
        for pos in positions:
            opened = pos[3]
            if opened:
                opened_str = opened if isinstance(opened, str) else str(opened)
                try:
                    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in opened_str else "%Y-%m-%d %H:%M:%S"
                    o = datetime.strptime(opened_str, fmt)
                    days = (datetime.now() - o).total_seconds() / 86400
                    if days > 1:
                        warnings.append(
                            f"오픈 포지션 장기 방치: {pos[0]} {pos[1]} "
                            f"entry={pos[2]:.2f} ({days:.1f}일째)"
                        )
                except ValueError:
                    pass

    if warnings:
        for w in warnings:
            print(f"  !! {w}")
    else:
        print("  (이슈 없음)")

    # ── 결론
    print("\n" + "━" * 70)
    print("  [결론] 종합 평가")
    print("━" * 70)

    if real_after:
        m = calc_metrics(real_after, pnl_key)
        verdict = "수익" if m["total_pnl"] > 0 else "손실"
        print(f"\n  실거래 (개선 후): {m['count']}건, 승률 {m['win_rate']:.1f}%, "
              f"PnL {m['total_pnl']:+.4f}, PF {m['pf']:.2f} → {verdict}")

    if paper_after:
        m = calc_metrics(paper_after, pnl_key)
        verdict = "수익" if m["total_pnl"] > 0 else "손실"
        print(f"  가상매매 (개선 후): {m['count']}건, 승률 {m['win_rate']:.1f}%, "
              f"PnL {m['total_pnl']:+.4f}, PF {m['pf']:.2f} → {verdict}")

    if warnings:
        print(f"\n  ** 데이터 품질 경고 {len(warnings)}건 — 결과 해석 시 주의 필요")
        print("  ** 봇 재시작 후 충분한 데이터 수집 뒤 재분석 권장")

    print()
    conn.close()


if __name__ == "__main__":
    main()
