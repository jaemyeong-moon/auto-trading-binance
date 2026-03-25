"""Paper Trading 정시 리포트 생성 + 웹훅 전송.

매 정시마다 호출되어 가상매매 현황을 요약하고 웹훅으로 전송한다.
사용: python -m src.core.paper_report
"""

import asyncio
import sys
from datetime import timedelta

import httpx
import structlog

from src.core import database as db
from src.core.database import (
    PaperBalance, PaperPosition, PaperTrade, get_session,
)
from src.utils.timezone import now_kst

logger = structlog.get_logger()


def _gather_stats() -> dict:
    """DB에서 전략별 가상매매 통계 수집."""
    now = now_kst()
    hour_ago = now - timedelta(hours=1)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    with get_session() as session:
        balances = session.query(PaperBalance).all()
        positions = session.query(PaperPosition).all()

        # 최근 1시간 거래
        recent_trades = (
            session.query(PaperTrade)
            .filter(PaperTrade.closed_at >= hour_ago)
            .order_by(PaperTrade.closed_at.desc())
            .all()
        )

        # 오늘 전체 거래
        today_trades = (
            session.query(PaperTrade)
            .filter(PaperTrade.closed_at >= today_start)
            .all()
        )

        # 전체 거래 (최근 100건)
        all_trades = (
            session.query(PaperTrade)
            .order_by(PaperTrade.closed_at.desc())
            .limit(100)
            .all()
        )

    strategies = {}
    for bal in balances:
        roi = ((bal.balance - bal.initial_balance) / bal.initial_balance * 100
               if bal.initial_balance > 0 else 0)
        winrate = (bal.wins / bal.total_trades * 100
                   if bal.total_trades > 0 else 0)
        strategies[bal.strategy] = {
            "balance": bal.balance,
            "initial": bal.initial_balance,
            "roi_pct": roi,
            "total_trades": bal.total_trades,
            "wins": bal.wins,
            "losses": bal.losses,
            "winrate": winrate,
        }

    open_positions = []
    for pos in positions:
        open_positions.append({
            "strategy": pos.strategy,
            "symbol": pos.symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "quantity": pos.quantity,
            "sl": pos.sl_price,
            "tp": pos.tp_price,
        })

    recent = []
    for t in recent_trades[:10]:
        recent.append({
            "strategy": t.strategy,
            "symbol": t.symbol,
            "side": t.side,
            "net_pnl": t.net_pnl or 0,
            "reason": t.reason,
        })

    today_pnl = sum(t.net_pnl or 0 for t in today_trades)
    today_count = len(today_trades)
    today_wins = sum(1 for t in today_trades if (t.net_pnl or 0) >= 0)

    hour_pnl = sum(t.net_pnl or 0 for t in recent_trades)
    hour_count = len(recent_trades)

    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M KST"),
        "strategies": strategies,
        "open_positions": open_positions,
        "recent_trades": recent,
        "today_pnl": today_pnl,
        "today_trades": today_count,
        "today_wins": today_wins,
        "hour_pnl": hour_pnl,
        "hour_trades": hour_count,
    }


def format_text_report(stats: dict) -> str:
    """터미널/로그용 텍스트 리포트."""
    lines = [
        f"{'='*40}",
        f"  Paper Trading Report",
        f"  {stats['timestamp']}",
        f"{'='*40}",
        "",
        f"  Last Hour: {stats['hour_trades']} trades, PnL ${stats['hour_pnl']:+.2f}",
        f"  Today:     {stats['today_trades']} trades, PnL ${stats['today_pnl']:+.2f}"
        + (f" (WR {stats['today_wins']}/{stats['today_trades']})" if stats['today_trades'] > 0 else ""),
        "",
    ]

    if stats["strategies"]:
        lines.append("  [ Strategy Performance ]")
        for name, s in stats["strategies"].items():
            lines.append(
                f"  {name}: ${s['balance']:.2f} "
                f"(ROI {s['roi_pct']:+.1f}%) "
                f"W/L {s['wins']}/{s['losses']} "
                f"WR {s['winrate']:.0f}%"
            )
        lines.append("")

    if stats["open_positions"]:
        lines.append("  [ Open Positions ]")
        for p in stats["open_positions"]:
            lines.append(
                f"  {p['strategy']} | {p['symbol']} {p['side']} "
                f"@ ${p['entry_price']:,.2f} "
                f"SL ${p['sl']:.2f} TP ${p['tp']:.2f}"
            )
        lines.append("")

    if stats["recent_trades"]:
        lines.append("  [ Recent Trades (1h) ]")
        for t in stats["recent_trades"][:5]:
            emoji = "+" if t["net_pnl"] >= 0 else ""
            lines.append(
                f"  {t['strategy']} | {t['symbol']} {t['side']} "
                f"${emoji}{t['net_pnl']:.4f} ({t['reason']})"
            )

    lines.append(f"{'='*40}")
    return "\n".join(lines)


def format_discord_report(stats: dict) -> dict:
    """Discord embed 형식 리포트."""
    # 전략별 성과 요약
    strat_lines = []
    for name, s in stats["strategies"].items():
        emoji = "🟢" if s["roi_pct"] >= 0 else "🔴"
        strat_lines.append(
            f"{emoji} **{name}**\n"
            f"  잔고: ${s['balance']:.2f} (ROI {s['roi_pct']:+.1f}%)\n"
            f"  승/패: {s['wins']}/{s['losses']} (WR {s['winrate']:.0f}%)"
        )

    # 포지션 요약
    pos_lines = []
    for p in stats["open_positions"]:
        side_emoji = "🟩" if p["side"] == "LONG" else "🟥"
        pos_lines.append(
            f"{side_emoji} {p['symbol']} {p['side']} @ ${p['entry_price']:,.2f}"
        )

    fields = [
        {
            "name": "Last Hour",
            "value": f"{stats['hour_trades']} trades | ${stats['hour_pnl']:+.2f}",
            "inline": True,
        },
        {
            "name": "Today",
            "value": (
                f"{stats['today_trades']} trades | ${stats['today_pnl']:+.2f}"
                + (f" | WR {stats['today_wins']}/{stats['today_trades']}"
                   if stats['today_trades'] > 0 else "")
            ),
            "inline": True,
        },
    ]

    if strat_lines:
        fields.append({
            "name": "Strategy Performance",
            "value": "\n".join(strat_lines) or "N/A",
            "inline": False,
        })

    if pos_lines:
        fields.append({
            "name": "Open Positions",
            "value": "\n".join(pos_lines),
            "inline": False,
        })

    # 최근 거래
    if stats["recent_trades"]:
        trade_lines = []
        for t in stats["recent_trades"][:5]:
            pnl_emoji = "✅" if t["net_pnl"] >= 0 else "❌"
            trade_lines.append(
                f"{pnl_emoji} {t['symbol']} {t['side']} ${t['net_pnl']:+.4f} ({t['reason']})"
            )
        fields.append({
            "name": "Recent Trades",
            "value": "\n".join(trade_lines),
            "inline": False,
        })

    color = 0x26A69A if stats["today_pnl"] >= 0 else 0xEF5350

    return {
        "embeds": [{
            "title": "📊 Paper Trading Report",
            "color": color,
            "fields": fields,
            "timestamp": now_kst().isoformat(),
            "footer": {"text": stats["timestamp"]},
        }]
    }


async def send_report() -> str:
    """리포트 생성 → 웹훅 전송. 전송된 텍스트 리포트를 반환."""
    db.init_db()
    stats = _gather_stats()
    text_report = format_text_report(stats)

    url = db.get_setting("webhook_url")
    if not url:
        logger.warning("paper_report.no_webhook_url")
        return text_report

    # 플랫폼 감지
    if "discord.com" in url or "discordapp.com" in url:
        payload = format_discord_report(stats)
    elif "hooks.slack.com" in url:
        payload = {"text": f"```\n{text_report}\n```"}
    else:
        payload = {"message": text_report, "timestamp": now_kst().isoformat()}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code < 400:
                logger.info("paper_report.sent", status=resp.status_code)
            else:
                logger.warning("paper_report.send_failed",
                               status=resp.status_code, body=resp.text[:200])
    except Exception:
        logger.exception("paper_report.send_error")

    return text_report


def main():
    """CLI 엔트리포인트."""
    report = asyncio.run(send_report())
    print(report)


if __name__ == "__main__":
    main()
