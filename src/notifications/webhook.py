"""Webhook — 매수/매도/청산 이벤트를 외부 URL로 전송.

Discord, Slack, 일반 URL을 자동 감지하여 각 플랫폼에 맞는 형식으로 전송.
"""

from datetime import datetime, timezone

import httpx
import structlog

from src.core import database as db
from src.utils.timezone import now_kst, KST

logger = structlog.get_logger()


def _detect_platform(url: str) -> str:
    if "discord.com" in url or "discordapp.com" in url:
        return "discord"
    if "hooks.slack.com" in url:
        return "slack"
    return "generic"


def _format_discord(event: str, data: dict) -> dict:
    """Discord embed 형식으로 변환."""
    colors = {
        "open": 0x26A69A,       # 초록
        "close": 0xEF5350,      # 빨강
        "partial_tp": 0xFFA726,  # 주황
    }

    emoji = {
        "open": "📈",
        "close": "📉",
        "partial_tp": "💰",
    }

    title = f"{emoji.get(event, '🔔')} {event.upper()}"

    fields = []
    # 주요 필드만 보기 좋게
    field_map = {
        "symbol": "심볼",
        "direction": "방향",
        "side": "방향",
        "price": "가격",
        "entry_price": "진입가",
        "exit_price": "청산가",
        "quantity": "수량",
        "invest_usdt": "투자금",
        "pnl_usdt": "손익(USDT)",
        "fee_usdt": "수수료",
        "net_pnl_usdt": "순수익",
        "reason": "사유",
        "change_pct": "변동(%)",
        "closed_quantity": "청산수량",
        "balance_usdt": "계좌잔고",
        "today_pnl_usdt": "오늘 수익",
        "today_pnl_pct": "오늘 수익률",
    }

    for key, label in field_map.items():
        val = data.get(key)
        if val is not None:
            if isinstance(val, float):
                if "price" in key or "usdt" in key.lower():
                    display = f"${val:,.2f}"
                elif "pct" in key:
                    display = f"{val:+.2f}%"
                else:
                    display = f"{val:.6f}"
            else:
                display = str(val)

            # 손익에 이모지
            if key == "pnl_usdt":
                pnl_emoji = "✅" if val >= 0 else "❌"
                display = f"{pnl_emoji} {display}"

            fields.append({"name": label, "value": display, "inline": True})

    embed = {
        "title": title,
        "color": colors.get(event, 0x607D8B),
        "fields": fields,
        "timestamp": now_kst().isoformat(),
        "footer": {"text": "Auto-Trader Bot (KST)"},
    }

    return {"embeds": [embed]}


def _format_slack(event: str, data: dict) -> dict:
    """Slack Block Kit 형식."""
    emoji = {"open": "📈", "close": "📉", "partial_tp": "💰"}
    lines = [f"{emoji.get(event, '🔔')} *{event.upper()}*"]

    for key, val in data.items():
        if key == "timestamp":
            continue
        if isinstance(val, float):
            val = f"${val:,.2f}" if "price" in key or "usdt" in key.lower() else f"{val}"
        lines.append(f"• {key}: `{val}`")

    return {"text": "\n".join(lines)}


def _format_generic(event: str, data: dict) -> dict:
    """일반 JSON."""
    return {
        "event": event,
        "timestamp": now_kst().isoformat(),
        **data,
    }


async def send_webhook(event: str, data: dict) -> None:
    """웹훅 URL로 POST 전송. 플랫폼 자동 감지."""
    url = db.get_setting("webhook_url")
    if not url:
        return

    platform = _detect_platform(url)

    if platform == "discord":
        payload = _format_discord(event, data)
    elif platform == "slack":
        payload = _format_slack(event, data)
    else:
        payload = _format_generic(event, data)

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                logger.warning("webhook.failed",
                               status=resp.status_code, platform=platform,
                               body=resp.text[:200])
    except Exception:
        logger.exception("webhook.error", url=url[:50])


# ─── 편의 함수 ────────────────────────────────────────────

async def notify_open(symbol: str, direction: str, price: float,
                      quantity: float, invest: float) -> None:
    await send_webhook("open", {
        "symbol": symbol,
        "direction": direction,
        "price": price,
        "quantity": quantity,
        "invest_usdt": round(invest, 2),
    })


async def notify_close(symbol: str, side: str, entry_price: float,
                        exit_price: float, pnl: float, reason: str = "",
                        balance: float = 0, fee: float = 0,
                        net_pnl: float = 0) -> None:
    data = {
        "symbol": symbol,
        "side": side,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_usdt": round(pnl, 2),
        "fee_usdt": round(fee, 4),
        "net_pnl_usdt": round(net_pnl, 2),
        "reason": reason,
    }
    if balance > 0:
        data["balance_usdt"] = round(balance, 2)
        # 오늘 누적 수익률 계산
        trades = db.get_trades(limit=50)
        today_pnl = sum(
            t.pnl for t in trades
            if t.pnl is not None and t.closed_at
            and t.closed_at.date() == now_kst().date()
        )
        data["today_pnl_usdt"] = round(today_pnl, 2)
        data["today_pnl_pct"] = f"{today_pnl / balance * 100:+.2f}%" if balance > 0 else "0%"
    await send_webhook("close", data)


async def notify_partial_tp(symbol: str, price: float,
                             closed_qty: float, change_pct: float) -> None:
    await send_webhook("partial_tp", {
        "symbol": symbol,
        "price": price,
        "closed_quantity": closed_qty,
        "change_pct": round(change_pct * 100, 2),
    })


async def send_raw(message: str) -> None:
    """단순 텍스트 메시지를 웹훅으로 전송."""
    url = db.get_setting("webhook_url")
    if not url:
        return

    platform = _detect_platform(url)

    if platform == "discord":
        payload = {"content": message}
    elif platform == "slack":
        payload = {"text": message}
    else:
        payload = {"message": message, "timestamp": now_kst().isoformat()}

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                logger.warning("webhook.send_raw_failed", status=resp.status_code)
    except Exception:
        logger.exception("webhook.send_raw_error")
