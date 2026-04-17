"""일일 종합 분석 리포트 — 서버에서 실행, JSON 출력.

퀀트 관점 자동 분석:
  1. 서버 상태 (프로세스/메모리/디스크)
  2. 실거래 성과 (24h/7d/전체)
  3. 페이퍼 전략 랭킹 (승률/PF/ROI)
  4. 시그널 품질 (confidence 분포)
  5. 포지션 현황 (실거래 + 페이퍼)
  6. 자동 전략 추천

Usage:
    python scripts/daily_analysis.py           # 기본 DB
    python scripts/daily_analysis.py /path/db  # 지정 DB
"""
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))


def _connect(db_path=None):
    if db_path is None:
        candidates = [
            "/home/ubuntu/data/trades_real.db",
            str(Path(__file__).parent.parent / "data" / "trades_real.db"),
        ]
        for p in candidates:
            if Path(p).exists():
                db_path = p
                break
    if not db_path or not Path(db_path).exists():
        print(json.dumps({"error": f"DB not found: {db_path}"}))
        sys.exit(1)
    return sqlite3.connect(db_path), db_path


def server_health():
    """서버 상태: 프로세스/메모리/디스크."""
    result = {}
    try:
        mem = subprocess.check_output(["free", "-m"], text=True).split("\n")[1].split()
        result["ram_total_mb"] = int(mem[1])
        result["ram_used_mb"] = int(mem[2])
        result["ram_pct"] = round(int(mem[2]) / int(mem[1]) * 100, 1)
    except Exception:
        result["ram_pct"] = -1

    try:
        disk = subprocess.check_output(["df", "-h", "/"], text=True).split("\n")[1].split()
        result["disk_used"] = disk[2]
        result["disk_avail"] = disk[3]
        result["disk_pct"] = disk[4]
    except Exception:
        pass

    # 봇 프로세스
    try:
        ps = subprocess.check_output(["pgrep", "-af", "python.*src.main"], text=True).strip()
        result["bot_running"] = bool(ps)
    except subprocess.CalledProcessError:
        result["bot_running"] = False

    try:
        ps = subprocess.check_output(["pgrep", "-af", "python.*dashboard.mobile"], text=True).strip()
        result["dashboard_running"] = bool(ps)
    except subprocess.CalledProcessError:
        result["dashboard_running"] = False

    return result


def real_trades_analysis(conn):
    """실거래 분석: 24h/7d/전체."""
    c = conn.cursor()
    now = datetime.now(KST)
    periods = {
        "24h": (now - timedelta(hours=24)).isoformat(),
        "7d": (now - timedelta(days=7)).isoformat(),
        "all": "2020-01-01",
    }
    result = {}
    for label, since in periods.items():
        c.execute("""SELECT COUNT(*),
            COALESCE(SUM(CASE WHEN pnl_pct>0 THEN 1 ELSE 0 END), 0),
            COALESCE(ROUND(SUM(pnl_pct), 4), 0),
            COALESCE(ROUND(AVG(pnl_pct), 4), 0),
            COALESCE(ROUND(MAX(pnl_pct), 4), 0),
            COALESCE(ROUND(MIN(pnl_pct), 4), 0)
            FROM trades WHERE opened_at >= ?""", (since,))
        r = c.fetchone()
        total = r[0] or 0
        wins = r[1] or 0
        result[label] = {
            "trades": total,
            "wins": wins,
            "losses": total - wins,
            "winrate": round(wins / total * 100, 1) if total > 0 else 0,
            "net_pnl_pct": r[2],
            "avg_pnl_pct": r[3],
            "best_pnl_pct": r[4],
            "worst_pnl_pct": r[5],
        }
    # 전략별 요약
    c.execute("""SELECT strategy, COUNT(*),
        SUM(CASE WHEN pnl_pct>0 THEN 1 ELSE 0 END),
        ROUND(SUM(pnl_pct), 4)
        FROM trades GROUP BY strategy ORDER BY SUM(pnl_pct) DESC""")
    result["by_strategy"] = []
    for r in c.fetchall():
        wr = round(r[2] / r[1] * 100, 1) if r[1] > 0 else 0
        result["by_strategy"].append({
            "strategy": r[0], "trades": r[1],
            "wins": r[2], "winrate": wr, "net_pnl_pct": r[3],
        })
    return result


def paper_trades_analysis(conn):
    """페이퍼 전략 랭킹."""
    c = conn.cursor()
    c.execute("""SELECT strategy, COUNT(*) as cnt,
        SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) as wins,
        ROUND(SUM(CASE WHEN net_pnl > 0 THEN net_pnl ELSE 0 END), 2) as gp,
        ROUND(SUM(CASE WHEN net_pnl < 0 THEN net_pnl ELSE 0 END), 2) as gl,
        ROUND(SUM(net_pnl), 2) as net,
        ROUND(AVG(net_pnl), 2) as avg
        FROM paper_trades GROUP BY strategy ORDER BY SUM(net_pnl) DESC""")
    strategies = []
    for r in c.fetchall():
        wr = round(r[2] / r[1] * 100, 1) if r[1] > 0 else 0
        pf = round(abs(r[3] / r[4]), 2) if r[4] != 0 else 999
        strategies.append({
            "strategy": r[0], "trades": r[1], "wins": r[2],
            "winrate": wr, "pf": pf, "net_pnl": r[5], "avg_pnl": r[6],
        })

    # 잔고
    c.execute("""SELECT strategy, balance, initial_balance, total_trades, wins, losses
        FROM paper_balances ORDER BY balance DESC""")
    balances = []
    for r in c.fetchall():
        roi = round((r[1] - r[2]) / r[2] * 100, 1) if r[2] > 0 else 0
        balances.append({
            "strategy": r[0], "balance": round(r[1], 2),
            "roi_pct": roi, "total_trades": r[3], "wins": r[4], "losses": r[5],
        })

    return {"ranked": strategies, "balances": balances}


def signal_quality(conn):
    """최근 24h 시그널 confidence 분포."""
    c = conn.cursor()
    since = (datetime.now(KST) - timedelta(hours=24)).isoformat()
    c.execute("""SELECT strategy,
        COUNT(*) as total,
        SUM(CASE WHEN signal_type IN ('buy','sell') THEN 1 ELSE 0 END) as entries,
        ROUND(AVG(CASE WHEN signal_type IN ('buy','sell') THEN confidence END), 3) as avg_conf,
        MAX(CASE WHEN signal_type IN ('buy','sell') THEN confidence END) as max_conf
        FROM signal_logs WHERE recorded_at >= ?
        GROUP BY strategy ORDER BY entries DESC""", (since,))
    result = []
    for r in c.fetchall():
        result.append({
            "strategy": r[0], "total_signals": r[1],
            "entry_signals": r[2], "avg_confidence": r[3], "max_confidence": r[4],
        })
    return result


def open_positions(conn):
    """열린 포지션 (실거래 + 페이퍼)."""
    c = conn.cursor()
    result = {"real": [], "paper": []}

    c.execute("SELECT symbol, side, entry_price, quantity, opened_at FROM positions")
    for r in c.fetchall():
        result["real"].append({
            "symbol": r[0], "side": r[1], "entry": r[2],
            "qty": r[3], "opened_at": r[4],
        })

    c.execute("SELECT strategy, symbol, side, entry_price, sl_price, tp_price, opened_at FROM paper_positions")
    for r in c.fetchall():
        result["paper"].append({
            "strategy": r[0], "symbol": r[1], "side": r[2],
            "entry": r[3], "sl": r[4], "tp": r[5], "opened_at": r[6],
        })
    return result


def current_settings(conn):
    """현재 봇 설정."""
    c = conn.cursor()
    c.execute("SELECT key, value FROM bot_settings")
    return {r[0]: r[1] for r in c.fetchall()}


def strategy_recommendation(paper, real):
    """데이터 기반 전략 추천.

    기준:
    - 페이퍼 5건 이상, 승률 45%+, PF 1.0+ → eligible
    - 실거래 검증 전략 우선
    - eligible 중 net_pnl 최고 선택
    """
    eligible = []
    for s in paper.get("ranked", []):
        if s["trades"] >= 5 and s["winrate"] >= 45 and s["pf"] >= 1.0:
            eligible.append(s)

    # 실거래 검증 가산점
    real_strategies = {s["strategy"] for s in real.get("by_strategy", [])}
    for e in eligible:
        if e["strategy"] in real_strategies:
            e["real_verified"] = True
        else:
            e["real_verified"] = False

    eligible.sort(key=lambda x: (x["real_verified"], x["net_pnl"]), reverse=True)

    if not eligible:
        # 전략 없으면 실거래에서 가장 좋았던 전략
        if real.get("by_strategy"):
            best_real = real["by_strategy"][0]
            return {
                "action": "keep_current",
                "reason": f"페이퍼 적격 전략 없음. 실거래 최고: {best_real['strategy']} (WR {best_real['winrate']}%)",
                "eligible": [],
            }
        return {"action": "pause", "reason": "적격 전략 없음", "eligible": []}

    best = eligible[0]
    return {
        "action": "recommend",
        "strategy": best["strategy"],
        "reason": f"WR {best['winrate']}%, PF {best['pf']}, net ${best['net_pnl']}"
                  + (" (실거래 검증)" if best.get("real_verified") else " (페이퍼만)"),
        "eligible": eligible,
    }


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    conn, actual_path = _connect(db_path)

    report = {
        "generated_at": datetime.now(KST).isoformat(),
        "db_path": actual_path,
        "server": server_health(),
        "settings": current_settings(conn),
        "real_trades": real_trades_analysis(conn),
        "paper_trades": paper_trades_analysis(conn),
        "signals_24h": signal_quality(conn),
        "positions": open_positions(conn),
    }
    report["recommendation"] = strategy_recommendation(
        report["paper_trades"], report["real_trades"]
    )

    conn.close()
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
