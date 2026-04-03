"""경량 모바일 대시보드 — FastAPI + 순수 HTML/CSS.

Streamlit 대비 ~30MB로 1core/1GB 서버에서도 부담 없이 실행.
모바일 최적화 UI, PWA 지원, 자동 새로고침.

실행: python -m src.dashboard.mobile
"""

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import uvicorn

from src.core import database as db
from src.exchange.futures_client import FuturesClient
from src.strategies.registry import list_strategies

db.init_db()

app = FastAPI(title="Auto-Trader Mobile")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

# ─── Auth ─────────────────────────────────────────────────
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")
SESSION_SECRET = secrets.token_hex(32)  # 서버 재시작 시 세션 만료
SESSION_MAX_AGE = 7 * 24 * 3600  # 7일


def _make_token(timestamp: int) -> str:
    msg = f"{timestamp}:{DASHBOARD_PASSWORD}"
    return hmac.new(SESSION_SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()


def _verify_session(request: Request) -> bool:
    """세션 쿠키 검증. 비밀번호 미설정 시 인증 비활성화."""
    if not DASHBOARD_PASSWORD:
        return True
    cookie = request.cookies.get("session", "")
    if not cookie or ":" not in cookie:
        return False
    try:
        ts_str, token = cookie.split(":", 1)
        ts = int(ts_str)
        if time.time() - ts > SESSION_MAX_AGE:
            return False
        return hmac.compare_digest(token, _make_token(ts))
    except (ValueError, TypeError):
        return False


@app.post("/api/login")
async def api_login(request: Request):
    body = await request.json()
    pw = body.get("password", "")
    if not DASHBOARD_PASSWORD:
        return JSONResponse({"ok": False, "error": "비밀번호가 설정되지 않았습니다"})
    if not hmac.compare_digest(pw, DASHBOARD_PASSWORD):
        return JSONResponse({"ok": False, "error": "비밀번호가 틀렸습니다"}, status_code=401)
    ts = int(time.time())
    token = _make_token(ts)
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        "session", f"{ts}:{token}",
        max_age=SESSION_MAX_AGE, httponly=True, samesite="strict",
    )
    return resp


@app.post("/api/logout")
async def api_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("session")
    return resp


def _auth_guard(request: Request) -> JSONResponse | None:
    """인증 실패 시 401 반환, 성공 시 None."""
    if not _verify_session(request):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    return None


# ─── API endpoints ────────────────────────────────────────

async def _fetch_live():
    """거래소 실시간 데이터 조회."""
    client = FuturesClient()
    await client.connect()
    try:
        account = await client.get_account_summary()
        positions = {}
        for s in SYMBOLS:
            try:
                price = await client.get_price(s)
                pos = await client.get_position(s)
                positions[s] = {"price": price, "position": pos}
            except Exception:
                positions[s] = {"price": 0, "position": None}
        return {"account": account, "positions": positions}
    finally:
        await client.disconnect()


@app.get("/api/status")
async def api_status(request: Request):
    if err := _auth_guard(request): return err
    try:
        data = await _fetch_live()
        bot_states = db.get_all_bot_states()
        settings = db.get_all_settings()

        # DB 포지션의 SL/TP를 거래소 포지션에 병합
        with db.get_session() as session:
            db_positions = {p.symbol: p for p in session.query(db.PositionRecord).all()}
        for sym, info in data["positions"].items():
            db_pos = db_positions.get(sym)
            if db_pos and info.get("position"):
                info["position"]["sl_price"] = db_pos.sl_price
                info["position"]["tp_price"] = db_pos.tp_price

        return JSONResponse({
            "ok": True,
            "account": data["account"],
            "positions": data["positions"],
            "bots": bot_states,
            "settings": {
                "strategy": settings.get("strategy", ""),
                "leverage": settings.get("leverage", "5"),
                "position_size_pct": settings.get("position_size_pct", "0.1"),
                "tp_pct": settings.get("tp_pct", "0.01"),
                "sl_pct": settings.get("sl_pct", "0.005"),
                "tick_interval": settings.get("tick_interval", "15"),
            },
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e)})


@app.get("/api/trades")
async def api_trades(request: Request, symbol: str | None = None, limit: int = 50):
    if err := _auth_guard(request): return err
    trades = db.get_trades(symbol=symbol, limit=limit)
    return JSONResponse([{
        "id": t.id, "symbol": t.symbol, "side": t.side,
        "entry": t.entry_price, "exit": t.exit_price,
        "qty": t.quantity, "pnl": round(t.pnl or 0, 2),
        "fee": round(t.fee or 0, 4),
        "net_pnl": round(t.net_pnl if t.net_pnl is not None else (t.pnl or 0), 2),
        "pnl_pct": round(t.pnl_pct or 0, 2),
        "strategy": t.strategy or "",
        "reason": getattr(t, "reason", "") or "",
        "sl_price": getattr(t, "sl_price", None),
        "tp_price": getattr(t, "tp_price", None),
        "opened_at": t.opened_at.isoformat() if t.opened_at else "",
        "closed_at": t.closed_at.isoformat() if t.closed_at else "",
    } for t in trades])


@app.get("/api/paper")
async def api_paper(request: Request):
    if err := _auth_guard(request): return err
    with db.get_session() as session:
        balances = session.query(db.PaperBalance).all()
        positions = session.query(db.PaperPosition).all()
        return JSONResponse({
            "balances": [{
                "strategy": b.strategy, "balance": round(b.balance, 2),
                "initial": b.initial_balance, "trades": b.total_trades,
                "wins": b.wins, "losses": b.losses,
            } for b in balances],
            "positions": [{
                "strategy": p.strategy, "symbol": p.symbol, "side": p.side,
                "entry": p.entry_price, "sl": p.sl_price, "tp": p.tp_price,
            } for p in positions],
        })


@app.get("/api/candles/{symbol}")
async def api_candles(request: Request, symbol: str, interval: str = "15m", limit: int = 200):
    """캔들 데이터 + 해당 심볼 페이퍼 트레이드 마커."""
    if err := _auth_guard(request): return err
    try:
        client = FuturesClient()
        await client.connect()
        try:
            candles = await client.get_candles(symbol, interval=interval, limit=limit)
        finally:
            await client.disconnect()

        if candles.empty:
            return JSONResponse({"candles": [], "trades": []})

        candle_data = [{
            "time": int(row.name.timestamp()) if hasattr(row.name, 'timestamp') else int(idx),
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row["volume"]),
        } for idx, row in candles.iterrows()]

        # 페이퍼 트레이드 마커 (최근 거래)
        with db.get_session() as session:
            paper_trades = session.query(db.PaperTrade).filter_by(
                symbol=symbol
            ).order_by(db.PaperTrade.closed_at.desc()).limit(50).all()

            trade_markers = [{
                "time": int(t.opened_at.timestamp()) if t.opened_at else 0,
                "exit_time": int(t.closed_at.timestamp()) if t.closed_at else 0,
                "side": t.side, "entry": t.entry_price, "exit": t.exit_price,
                "sl": t.sl_price, "tp": t.tp_price,
                "pnl": round(t.net_pnl or 0, 4), "reason": t.reason or "",
                "strategy": t.strategy,
            } for t in paper_trades]

            # 열린 포지션도 마커로
            open_positions = session.query(db.PaperPosition).filter_by(
                symbol=symbol).all()
            open_markers = [{
                "time": int(p.opened_at.timestamp()) if p.opened_at else 0,
                "side": p.side, "entry": p.entry_price,
                "sl": p.sl_price, "tp": p.tp_price,
                "strategy": p.strategy, "open": True,
            } for p in open_positions]

        return JSONResponse({
            "candles": candle_data,
            "trades": trade_markers,
            "positions": open_markers,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"candles": [], "trades": [], "error": str(e)})


@app.get("/api/paper/trades")
async def api_paper_trades(request: Request, strategy: str | None = None, limit: int = 50):
    """페이퍼 트레이드 내역."""
    if err := _auth_guard(request): return err
    with db.get_session() as session:
        q = session.query(db.PaperTrade).order_by(db.PaperTrade.closed_at.desc())
        if strategy:
            q = q.filter_by(strategy=strategy)
        trades = q.limit(limit).all()
        return JSONResponse([{
            "id": t.id, "strategy": t.strategy, "symbol": t.symbol,
            "side": t.side, "entry": t.entry_price, "exit": t.exit_price,
            "pnl": round(t.net_pnl or 0, 4), "fee": round(t.fee or 0, 4),
            "reason": t.reason or "", "sl": t.sl_price, "tp": t.tp_price,
            "closed_at": t.closed_at.isoformat() if t.closed_at else "",
        } for t in trades])


@app.get("/api/strategies")
async def api_strategies(request: Request):
    if err := _auth_guard(request): return err
    strategies = list_strategies()
    return JSONResponse(strategies)


@app.post("/api/bot/{symbol}/start")
async def api_bot_start(request: Request, symbol: str):
    if err := _auth_guard(request): return err
    db.set_bot_running(symbol, True)
    return JSONResponse({"ok": True, "symbol": symbol, "running": True})


@app.post("/api/bot/{symbol}/stop")
async def api_bot_stop(request: Request, symbol: str):
    if err := _auth_guard(request): return err
    db.set_bot_running(symbol, False)
    return JSONResponse({"ok": True, "symbol": symbol, "running": False})


@app.post("/api/settings")
async def api_settings_save(request: Request):
    if err := _auth_guard(request): return err
    body = await request.json()
    for key, value in body.items():
        db.set_setting(key, str(value))
    return JSONResponse({"ok": True})


@app.get("/api/trade/{trade_id}/chart")
async def api_trade_chart(request: Request, trade_id: int, source: str = "real"):
    """포지션 진입→청산 구간의 캔들 + SL/TP + 진입/청산 마커."""
    if err := _auth_guard(request): return err
    try:
        with db.get_session() as session:
            if source == "paper":
                trade = session.query(db.PaperTrade).get(trade_id)
            else:
                trade = session.query(db.TradeRecord).get(trade_id)
            if not trade:
                return JSONResponse({"error": "trade not found"}, status_code=404)

            info = {
                "symbol": trade.symbol, "side": trade.side,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "sl_price": getattr(trade, "sl_price", None),
                "tp_price": getattr(trade, "tp_price", None),
                "pnl": round(getattr(trade, "net_pnl", None) or getattr(trade, "pnl", 0) or 0, 4),
                "reason": getattr(trade, "reason", "") or "",
                "opened_at": trade.opened_at.isoformat() if trade.opened_at else None,
                "closed_at": trade.closed_at.isoformat() if trade.closed_at else None,
            }

        # 구간 캔들 조회 (opened_at ~ closed_at + 여유)
        if not info["opened_at"] or not info["closed_at"]:
            return JSONResponse({"trade": info, "candles": []})

        from datetime import datetime, timedelta
        opened = datetime.fromisoformat(info["opened_at"])
        closed = datetime.fromisoformat(info["closed_at"])
        duration = (closed - opened).total_seconds()

        # 캔들 간격: 포지션 보유 시간에 따라 자동 선택
        if duration < 3600:          # < 1h → 1m 캔들
            interval, limit = "1m", min(int(duration / 60) + 20, 500)
        elif duration < 86400:       # < 24h → 5m 캔들
            interval, limit = "5m", min(int(duration / 300) + 20, 500)
        else:                        # >= 24h → 15m 캔들
            interval, limit = "15m", min(int(duration / 900) + 20, 500)

        # 시작 시간 여유 (앞뒤 10% 추가)
        margin = timedelta(seconds=duration * 0.1 + 60)
        start_ms = int((opened - margin).timestamp() * 1000)

        client = FuturesClient()
        await client.connect()
        try:
            candles = await client.get_candles(
                info["symbol"], interval=interval, limit=limit,
            )
        finally:
            await client.disconnect()

        if candles.empty:
            return JSONResponse({"trade": info, "candles": []})

        # 시간 범위 필터링
        candle_data = []
        end_ts = (closed + margin).timestamp()
        start_ts = (opened - margin).timestamp()
        for idx, row in candles.iterrows():
            ts = row.name.timestamp() if hasattr(row.name, 'timestamp') else float(idx)
            if start_ts <= ts <= end_ts:
                candle_data.append({
                    "time": int(ts),
                    "open": float(row["open"]), "high": float(row["high"]),
                    "low": float(row["low"]), "close": float(row["close"]),
                    "volume": float(row["volume"]),
                })

        return JSONResponse({"trade": info, "candles": candle_data})

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# ─── Mobile HTML SPA ──────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if DASHBOARD_PASSWORD and not _verify_session(request):
        return LOGIN_HTML
    return MOBILE_HTML


LOGIN_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="theme-color" content="#0a0a0f">
<title>Auto-Trader Login</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #0a0a0f; color: #e0e0e0;
  display: flex; align-items: center; justify-content: center;
  min-height: 100vh; padding: 20px;
}
.login-box {
  width: 100%; max-width: 360px; background: #141420;
  border-radius: 16px; padding: 32px 24px;
  border: 1px solid #1e1e30;
}
.login-box h1 { text-align: center; font-size: 22px; margin-bottom: 8px; }
.login-box p { text-align: center; font-size: 13px; color: #888; margin-bottom: 24px; }
.login-box input {
  width: 100%; padding: 14px 16px; border-radius: 10px;
  border: 1px solid #1e1e30; background: #0a0a0f; color: #e0e0e0;
  font-size: 16px; margin-bottom: 12px; outline: none;
}
.login-box input:focus { border-color: #6c5ce7; }
.login-box button {
  width: 100%; padding: 14px; border: none; border-radius: 10px;
  background: #6c5ce7; color: white; font-size: 16px; font-weight: 600;
  cursor: pointer;
}
.login-box button:active { opacity: 0.8; }
.error { color: #ef5350; font-size: 13px; text-align: center; margin-top: 12px; display: none; }
</style>
</head>
<body>
<div class="login-box">
  <h1>Auto-Trader</h1>
  <p>대시보드 접속을 위해 비밀번호를 입력하세요</p>
  <form onsubmit="doLogin(event)">
    <input type="password" id="pw" placeholder="비밀번호" autocomplete="current-password" autofocus>
    <button type="submit">로그인</button>
  </form>
  <div class="error" id="err"></div>
</div>
<script>
async function doLogin(e) {
  e.preventDefault();
  const pw = document.getElementById('pw').value;
  const errEl = document.getElementById('err');
  errEl.style.display = 'none';
  try {
    const r = await fetch('/api/login', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({password: pw}),
    });
    const d = await r.json();
    if (d.ok) { location.href = '/'; }
    else { errEl.textContent = d.error || '로그인 실패'; errEl.style.display = 'block'; }
  } catch(e) { errEl.textContent = '서버 연결 실패'; errEl.style.display = 'block'; }
}
</script>
</body>
</html>"""

MOBILE_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#0a0a0f">
<title>Auto-Trader</title>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root {
  --bg: #0a0a0f; --card: #141420; --border: #1e1e30;
  --text: #e0e0e0; --dim: #888; --accent: #6c5ce7;
  --green: #00d4aa; --red: #ef5350; --orange: #ffa726;
  --radius: 12px;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg); color: var(--text);
  min-height: 100vh; padding-bottom: 70px;
  -webkit-font-smoothing: antialiased;
}
.top-bar {
  position: sticky; top: 0; z-index: 100;
  background: rgba(10,10,15,0.92); backdrop-filter: blur(12px);
  padding: 12px 16px; display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid var(--border);
}
.top-bar h1 { font-size: 18px; font-weight: 700; }
.top-bar .status { font-size: 12px; color: var(--dim); }
.logout-btn { background:none; border:1px solid var(--border); color:var(--dim); padding:4px 10px;
  border-radius:6px; font-size:11px; cursor:pointer; margin-left:8px; }
.refresh-dot { width:8px; height:8px; border-radius:50%; background:var(--green); display:inline-block; margin-right:6px; }
.refresh-dot.off { background:var(--red); }

/* Tab bar */
.tab-bar {
  position: fixed; bottom: 0; left: 0; right: 0; z-index: 100;
  background: rgba(10,10,15,0.95); backdrop-filter: blur(12px);
  display: flex; border-top: 1px solid var(--border);
  padding-bottom: env(safe-area-inset-bottom, 0);
}
.tab-bar button {
  flex: 1; border: none; background: none; color: var(--dim);
  padding: 10px 0 8px; font-size: 10px; display: flex; flex-direction: column;
  align-items: center; gap: 2px; cursor: pointer; transition: color 0.2s;
}
.tab-bar button.active { color: var(--accent); }
.tab-bar button .icon { font-size: 20px; }

/* Cards */
.page { display: none; padding: 12px; }
.page.active { display: block; }
.card {
  background: var(--card); border-radius: var(--radius);
  padding: 14px; margin-bottom: 10px;
  border: 1px solid var(--border);
}
.card-title { font-size: 13px; color: var(--dim); margin-bottom: 8px; font-weight: 600; }

/* Metrics row */
.metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.metrics.cols-3 { grid-template-columns: 1fr 1fr 1fr; }
.metric { text-align: center; }
.metric .value { font-size: 20px; font-weight: 700; }
.metric .label { font-size: 11px; color: var(--dim); margin-top: 2px; }
.metric.sm .value { font-size: 16px; }

/* Colors */
.positive { color: var(--green); }
.negative { color: var(--red); }
.warn { color: var(--orange); }

/* Position card */
.pos-card { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; }
.pos-card + .pos-card { border-top: 1px solid var(--border); }
.pos-symbol { font-weight: 700; font-size: 15px; }
.pos-side { font-size: 12px; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
.pos-side.long { background: rgba(0,212,170,0.15); color: var(--green); }
.pos-side.short { background: rgba(239,83,80,0.15); color: var(--red); }
.pos-pnl { font-size: 16px; font-weight: 700; text-align: right; }
.pos-details { font-size: 11px; color: var(--dim); text-align: right; }

/* Bot pills */
.bot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.bot-pill {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 12px; border-radius: 10px; border: 1px solid var(--border);
  background: var(--card);
}
.bot-pill .name { font-weight: 600; font-size: 14px; }
.bot-pill .dot { width:8px; height:8px; border-radius:50%; margin-right:6px; }
.bot-pill .dot.on { background:var(--green); }
.bot-pill .dot.off { background:var(--red); }
.bot-pill button {
  border:none; border-radius:6px; padding:6px 12px; font-size:12px; font-weight:600; cursor:pointer;
}
.bot-pill button.start { background:rgba(0,212,170,0.2); color:var(--green); }
.bot-pill button.stop { background:rgba(239,83,80,0.2); color:var(--red); }

/* Trade list */
.trade-item { padding: 10px 0; display: flex; justify-content: space-between; align-items: center; }
.trade-item + .trade-item { border-top: 1px solid var(--border); }
.trade-info .trade-sym { font-weight: 600; font-size: 14px; }
.trade-info .trade-meta { font-size: 11px; color: var(--dim); }
.trade-pnl { font-weight: 700; font-size: 15px; text-align: right; }
.trade-pnl-pct { font-size: 11px; text-align: right; }

/* Paper trading */
.paper-row { padding: 10px 0; }
.paper-row + .paper-row { border-top: 1px solid var(--border); }
.paper-header { display: flex; justify-content: space-between; align-items: center; }
.paper-name { font-weight: 600; font-size: 14px; }
.paper-bal { font-weight: 700; font-size: 16px; }
.paper-stats { display: flex; gap: 12px; margin-top: 4px; font-size: 12px; color: var(--dim); }

/* Settings */
.setting-row { padding: 12px 0; }
.setting-row + .setting-row { border-top: 1px solid var(--border); }
.setting-row label { display: block; font-size: 13px; color: var(--dim); margin-bottom: 6px; }
.setting-row input, .setting-row select {
  width: 100%; padding: 10px 12px; border-radius: 8px;
  border: 1px solid var(--border); background: var(--bg); color: var(--text);
  font-size: 15px;
}
.btn-primary {
  width: 100%; padding: 14px; border: none; border-radius: 10px;
  background: var(--accent); color: white; font-size: 15px; font-weight: 600;
  cursor: pointer; margin-top: 12px;
}
.btn-primary:active { opacity: 0.8; }

/* Loading */
.loading { text-align: center; padding: 40px; color: var(--dim); }
.spinner { display: inline-block; width: 24px; height: 24px; border: 3px solid var(--border);
  border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Pull to refresh hint */
.pull-hint { text-align: center; font-size: 12px; color: var(--dim); padding: 8px; }
</style>
</head>
<body>

<!-- Top bar -->
<div class="top-bar">
  <div>
    <h1>Auto-Trader</h1>
  </div>
  <div class="status">
    <span class="refresh-dot" id="connDot"></span>
    <span id="lastUpdate">-</span>
    <button class="logout-btn" onclick="doLogout()">로그아웃</button>
  </div>
</div>

<!-- Pages -->
<div class="page active" id="page-home">
  <div id="accountCard" class="card">
    <div class="card-title">계좌 현황</div>
    <div class="metrics">
      <div class="metric"><div class="value" id="mBalance">-</div><div class="label">총 잔고</div></div>
      <div class="metric"><div class="value" id="mAvailable">-</div><div class="label">가용</div></div>
      <div class="metric"><div class="value" id="mMargin">-</div><div class="label">증거금</div></div>
      <div class="metric"><div class="value" id="mUpnl">-</div><div class="label">미실현 PnL</div></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">포지션</div>
    <div id="positionList"><div class="loading"><div class="spinner"></div></div></div>
  </div>
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <div class="card-title" style="margin:0;">실시간 차트</div>
      <select id="chartSymbol" style="background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:12px;">
        <option value="BTCUSDT">BTC</option>
        <option value="ETHUSDT">ETH</option>
        <option value="BNBUSDT">BNB</option>
        <option value="SOLUSDT">SOL</option>
        <option value="XRPUSDT">XRP</option>
      </select>
    </div>
    <div id="chartContainer" style="width:100%;height:350px;border-radius:8px;overflow:hidden;"></div>
  </div>
  <div class="card">
    <div class="card-title">실거래 내역</div>
    <div id="recentTrades"><div class="loading"><div class="spinner"></div></div></div>
  </div>
</div>

<div class="page" id="page-bots">
  <div class="card">
    <div class="card-title">활성 전략</div>
    <div id="activeStrategy" style="font-size:15px;font-weight:600;margin-bottom:8px;">-</div>
    <div id="strategyDesc" style="font-size:12px;color:var(--dim);"></div>
  </div>
  <div class="card">
    <div class="card-title">봇 제어</div>
    <div class="bot-grid" id="botGrid"></div>
  </div>
  <div class="card">
    <div class="card-title">현재 설정</div>
    <div id="settingSummary"></div>
  </div>
</div>

<div class="page" id="page-trades">
  <div class="card">
    <div class="card-title">전략별 성과</div>
    <div id="paperStrategyList"><div class="loading"><div class="spinner"></div></div></div>
  </div>
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <div class="card-title" style="margin:0;">차트</div>
      <select id="tradeChartSymbol" style="background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:12px;">
        <option value="BTCUSDT">BTC</option>
        <option value="ETHUSDT">ETH</option>
        <option value="BNBUSDT">BNB</option>
        <option value="SOLUSDT">SOL</option>
        <option value="XRPUSDT">XRP</option>
      </select>
    </div>
    <div id="tvTradeChartWrap" style="width:100%;height:400px;border-radius:8px;overflow:hidden;"></div>
    <div id="tradeChartContainer" style="width:100%;height:300px;border-radius:8px;overflow:hidden;display:none;"></div>
  </div>
  <div class="card">
    <div class="card-title">최근 거래 내역</div>
    <div id="tradeList"><div class="loading"><div class="spinner"></div></div></div>
  </div>
</div>


<div class="page" id="page-settings">
  <div class="card">
    <div class="card-title">전략 선택</div>
    <div class="setting-row">
      <label>매매 전략</label>
      <select id="sStrategy"></select>
    </div>
    <div id="strategyInfo" style="font-size:12px;color:var(--dim);padding:4px 0;"></div>
  </div>
  <div class="card">
    <div class="card-title">거래 설정</div>
    <div class="setting-row">
      <label>레버리지</label>
      <select id="sLeverage">
        <option value="1">x1</option><option value="2">x2</option>
        <option value="3">x3</option><option value="5">x5</option>
        <option value="7">x7</option><option value="10">x10</option>
        <option value="15">x15</option><option value="20">x20</option>
      </select>
    </div>
    <div class="setting-row">
      <label>투자 비율 (%)</label>
      <input type="number" id="sSizePct" min="5" max="100" step="5">
    </div>
    <div class="setting-row">
      <label>익절 (%)</label>
      <input type="number" id="sTp" min="0.1" max="10" step="0.1">
    </div>
    <div class="setting-row">
      <label>손절 (%)</label>
      <input type="number" id="sSl" min="0.1" max="10" step="0.1">
    </div>
    <div class="setting-row">
      <label>분석 주기 (초)</label>
      <input type="number" id="sTick" min="5" max="120" step="5">
    </div>
    <button class="btn-primary" onclick="saveSettings()">설정 저장</button>
    <div style="font-size:11px;color:var(--dim);margin-top:8px;text-align:center;">전략 변경 시 봇 재시작 필요</div>
  </div>
</div>

<!-- Tab bar -->
<div class="tab-bar">
  <button class="active" onclick="showPage('home',this)">
    <span class="icon">📊</span>현황
  </button>
  <button onclick="showPage('bots',this)">
    <span class="icon">🤖</span>봇
  </button>
  <button onclick="showPage('trades',this)">
    <span class="icon">📋</span>가상매매
  </button>
  <button onclick="showPage('settings',this)">
    <span class="icon">⚙️</span>설정
  </button>
</div>

<!-- 포지션 차트 모달 -->
<div id="tradeChartModal" style="display:none;position:fixed;top:0;left:0;right:0;bottom:0;z-index:1000;background:var(--bg);">
  <div style="display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid var(--border);">
    <div id="tradeChartTitle" style="font-weight:700;font-size:14px;"></div>
    <button onclick="closeTradeChart()" style="background:none;border:none;color:var(--text);font-size:20px;cursor:pointer;padding:4px 8px;">✕</button>
  </div>
  <div id="tradeChartInfo" style="padding:8px 16px;font-size:12px;color:var(--dim);"></div>
  <div id="tradeChartArea" style="width:100%;height:calc(100vh - 120px);"></div>
</div>

<script>
// ─── Auth helpers ────────────────────────────
async function authFetch(url, opts={}) {
  const r = await fetch(url, opts);
  if (r.status === 401) { location.href = '/'; return null; }
  return r;
}
async function doLogout() {
  await fetch('/api/logout', {method:'POST'});
  location.href = '/';
}

// ─── Strategies cache ────────────────────────
let strategiesCache = [];
async function loadStrategies() {
  try {
    const r = await authFetch('/api/strategies');
    if (!r) return;
    strategiesCache = await r.json();
  } catch(e) { console.error('loadStrategies:', e); }
}

// ─── Navigation ──────────────────────────────
function showPage(name, btn) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-bar button').forEach(b => b.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  if (btn) btn.classList.add('active');
  // Load data for page
  if (name==='home') loadStatus();
  else if (name==='trades') { loadTrades(); loadTradeChart(); }
  else if (name==='bots') loadStatus();
  else if (name==='settings') loadSettingsForm();
}

// ─── Formatting ──────────────────────────────
const fmt = (n,d=2) => n!=null ? '$'+Number(n).toLocaleString('en',{minimumFractionDigits:d,maximumFractionDigits:d}) : '-';
const pct = (n) => n!=null ? (n>=0?'+':'')+n.toFixed(2)+'%' : '-';
const cls = (n) => n>=0 ? 'positive' : 'negative';

// ─── Load Status ─────────────────────────────
let statusCache = null;

async function loadStatus() {
  try {
    const r = await authFetch('/api/status');
    if (!r) return;
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || 'API 응답 오류');
    statusCache = d;
    renderStatus(d);
    document.getElementById('connDot').className = 'refresh-dot';
    document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString('ko');
  } catch(e) {
    document.getElementById('connDot').className = 'refresh-dot off';
    document.getElementById('lastUpdate').textContent = e.message || '연결 실패';
    console.error('loadStatus error:', e);
  }
}

function renderStatus(d) {
  const a = d.account || {};
  document.getElementById('mBalance').textContent = fmt(a.balance);
  document.getElementById('mAvailable').textContent = fmt(a.available);

  let totalMargin=0, totalUpnl=0;
  const posHtml = [];
  for (const [sym, info] of Object.entries(d.positions || {})) {
    const pos = info.position;
    if (!pos) continue;
    totalMargin += pos.margin || 0;
    totalUpnl += pos.unrealized_pnl || 0;
    const entry = pos.entry_price || 0;
    const mark = pos.mark_price || 0;
    const pnlPct = entry > 0 ? (pos.side==='LONG'
      ? ((mark-entry)/entry*100)
      : ((entry-mark)/entry*100)) : 0;
    posHtml.push(`
      <div class="pos-card">
        <div>
          <div class="pos-symbol">${sym.replace('USDT','')}</div>
          <span class="pos-side ${pos.side.toLowerCase()}">${pos.side}</span>
          <span style="font-size:11px;color:var(--dim);margin-left:6px;">${fmt(pos.entry_price)}</span>
        </div>
        <div>
          <div class="pos-pnl ${cls(pos.unrealized_pnl)}">${fmt(pos.unrealized_pnl)}</div>
          <div class="pos-details ${cls(pnlPct)}">${pct(pnlPct)}</div>
        </div>
      </div>
    `);
  }
  document.getElementById('mMargin').textContent = fmt(totalMargin);
  const upnlEl = document.getElementById('mUpnl');
  upnlEl.textContent = fmt(totalUpnl);
  upnlEl.className = 'value ' + cls(totalUpnl);

  document.getElementById('positionList').innerHTML =
    posHtml.length ? posHtml.join('') : '<div style="text-align:center;padding:16px;color:var(--dim)">포지션 없음</div>';

  // Recent trades (last 5)
  loadRecentTrades();

  // Bot grid
  renderBots(d.bots);

  // Setting summary + active strategy
  const s = d.settings;
  const stratLabel = strategiesCache.find(x=>x.name===s.strategy);
  const stratName = stratLabel ? stratLabel.label : s.strategy;
  const stratDesc = stratLabel ? stratLabel.description : '';
  document.getElementById('activeStrategy').textContent = stratName;
  document.getElementById('strategyDesc').textContent = stratDesc;
  document.getElementById('settingSummary').innerHTML = `
    <div style="font-size:13px;line-height:2;">
      전략: <b>${stratName}</b><br>
      레버리지: <b>x${s.leverage}</b><br>
      투자비율: <b>${(parseFloat(s.position_size_pct)*100).toFixed(0)}%</b><br>
      익절/손절: <b>${(parseFloat(s.tp_pct)*100).toFixed(1)}% / ${(parseFloat(s.sl_pct)*100).toFixed(1)}%</b><br>
      분석주기: <b>${s.tick_interval}초</b>
    </div>
  `;

  // Chart — 현황 탭은 lightweight-charts + 실거래 포지션 마커
  loadChart();
}

async function loadRecentTrades() {
  try {
  const r = await authFetch('/api/trades?limit=20');
  if (!r) return;
  const trades = await r.json();
  if (!trades.length) {
    document.getElementById('recentTrades').innerHTML =
      '<div style="text-align:center;padding:16px;color:var(--dim)">실거래 내역 없음</div>';
    return;
  }
  document.getElementById('recentTrades').innerHTML = trades.map(t => `
    <div class="trade-item" onclick="openTradeChart(${t.id},'real')" style="cursor:pointer;">
      <div class="trade-info">
        <div class="trade-sym">${t.symbol.replace('USDT','')} <span class="pos-side ${t.side==='BUY'||t.side==='LONG'?'long':'short'}" style="font-size:10px">${t.side}</span>
          ${t.reason ? '<span style="font-size:10px;color:var(--dim);margin-left:4px;">'+t.reason+'</span>' : ''}
        </div>
        <div class="trade-meta">${fmt(t.entry)} → ${fmt(t.exit)} · ${t.strategy || ''}</div>
        <div class="trade-meta">${t.closed_at ? new Date(t.closed_at).toLocaleString('ko') : ''}</div>
      </div>
      <div>
        <div class="trade-pnl ${cls(t.net_pnl)}">${fmt(t.net_pnl)}</div>
        <div class="trade-pnl-pct ${cls(t.pnl_pct)}">${pct(t.pnl_pct)}</div>
      </div>
    </div>
  `).join('');
  } catch(e) { console.error('loadRecentTrades error:', e); }
}

// ─── Bots ────────────────────────────────────
function renderBots(bots) {
  const grid = document.getElementById('botGrid');
  const symbols = ['BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT'];
  grid.innerHTML = symbols.map(s => {
    const running = bots[s] || false;
    return `
      <div class="bot-pill">
        <div style="display:flex;align-items:center">
          <span class="dot ${running?'on':'off'}"></span>
          <span class="name">${s.replace('USDT','')}</span>
        </div>
        <button class="${running?'stop':'start'}" onclick="toggleBot('${s}',${!running})">
          ${running?'중지':'시작'}
        </button>
      </div>
    `;
  }).join('');
}

async function toggleBot(symbol, start) {
  await authFetch(`/api/bot/${symbol}/${start?'start':'stop'}`, {method:'POST'});
  loadStatus();
}

// ─── Trades ──────────────────────────────────
async function loadTrades() {
  // 1) 전략별 성과 요약
  try {
    const r = await authFetch('/api/paper');
    if (r) {
      const d = await r.json();
      const el = document.getElementById('paperStrategyList');
      if (!d.balances || !d.balances.length) {
        el.innerHTML = '<div style="text-align:center;padding:16px;color:var(--dim)">가상매매 데이터 없음</div>';
      } else {
        el.innerHTML = d.balances.map(b => {
          const pnl = b.balance - b.initial;
          const pnlPct = (pnl / b.initial * 100);
          const wr = b.trades > 0 ? (b.wins/b.trades*100).toFixed(1) : '0';
          const pos = d.positions.filter(p=>p.strategy===b.strategy);
          const label = strategiesCache.find(x=>x.name===b.strategy);
          const name = label ? label.label : b.strategy;
          return `
            <div class="paper-row">
              <div class="paper-header">
                <span class="paper-name">${name}</span>
                <span class="paper-bal ${cls(pnl)}">${fmt(b.balance)} <small>(${pnl>=0?'+':''}${pnlPct.toFixed(1)}%)</small></span>
              </div>
              <div class="paper-stats">
                <span>${b.trades}건</span>
                <span>승률 ${wr}%</span>
                <span>${b.wins}W/${b.losses}L</span>
              </div>
              ${pos.map(p => `
                <div style="margin-top:6px;font-size:12px;padding:6px 8px;background:var(--bg);border-radius:6px;">
                  <span class="pos-side ${p.side.toLowerCase()}" style="font-size:10px">${p.side}</span>
                  ${p.symbol.replace('USDT','')} @ ${fmt(p.entry)}
                  <span style="color:var(--dim)">SL ${fmt(p.sl)} / TP ${fmt(p.tp)}</span>
                </div>
              `).join('')}
            </div>
          `;
        }).join('');
      }
    }
  } catch(e) { console.error('loadTrades paper:', e); }

  // 2) 최근 거래 내역
  try {
    const r2 = await authFetch('/api/paper/trades?limit=30');
    if (!r2) return;
    const trades = await r2.json();
    if (!trades.length) {
      document.getElementById('tradeList').innerHTML =
        '<div style="text-align:center;padding:16px;color:var(--dim)">거래 없음</div>';
      return;
    }
    document.getElementById('tradeList').innerHTML = trades.map(t => `
      <div class="trade-item" onclick="openTradeChart(${t.id},'paper')" style="cursor:pointer;">
        <div class="trade-info">
          <div class="trade-sym">${t.symbol.replace('USDT','')} <span class="pos-side ${t.side==='BUY'||t.side==='LONG'?'long':'short'}" style="font-size:10px">${t.side}</span>
            <span style="font-size:10px;color:var(--accent);margin-left:4px;">${(strategiesCache.find(x=>x.name===t.strategy)||{}).label||t.strategy}</span>
            ${t.reason ? '<span style="font-size:10px;color:var(--dim);margin-left:2px;">'+t.reason+'</span>' : ''}
          </div>
          <div class="trade-meta">${fmt(t.entry)} → ${fmt(t.exit)} · ${t.closed_at ? new Date(t.closed_at).toLocaleString('ko',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'}) : ''}</div>
        </div>
        <div>
          <div class="trade-pnl ${cls(t.pnl)}">${fmt(t.pnl)}</div>
        </div>
      </div>
    `).join('');
  } catch(e) { console.error('loadTrades list:', e); }
}

// ─── Paper Trading ───────────────────────────
async function loadPaper() {
  const r = await authFetch('/api/paper');
  if (!r) return;
  const d = await r.json();

  if (!d.balances.length) {
    document.getElementById('paperList').innerHTML =
      '<div style="text-align:center;padding:16px;color:var(--dim)">가상매매 데이터 없음</div>';
    return;
  }

  document.getElementById('paperList').innerHTML = d.balances.map(b => {
    const pnl = b.balance - b.initial;
    const pnlPct = (pnl / b.initial * 100);
    const wr = b.trades > 0 ? (b.wins/b.trades*100).toFixed(1) : '0';
    const pos = d.positions.filter(p=>p.strategy===b.strategy);
    const label = strategiesCache.find(x=>x.name===b.strategy);
    const name = label ? label.label : b.strategy;
    return `
      <div class="paper-row">
        <div class="paper-header">
          <span class="paper-name">${name}</span>
          <span class="paper-bal ${cls(pnl)}">${fmt(b.balance)} <small>(${pnl>=0?'+':''}${pnlPct.toFixed(1)}%)</small></span>
        </div>
        <div class="paper-stats">
          <span>${b.trades}건</span>
          <span>승률 ${wr}%</span>
          <span>${b.wins}W/${b.losses}L</span>
        </div>
        ${pos.map(p => `
          <div style="margin-top:6px;font-size:12px;padding:6px 8px;background:var(--bg);border-radius:6px;">
            <span class="pos-side ${p.side.toLowerCase()}" style="font-size:10px">${p.side}</span>
            ${p.symbol.replace('USDT','')} @ ${fmt(p.entry)}
            <span style="color:var(--dim)">SL ${fmt(p.sl)} / TP ${fmt(p.tp)}</span>
          </div>
        `).join('')}
      </div>
    `;
  }).join('');
}

// ─── Paper summary (bots page) ───────────────
async function loadPaperSummaryBots() {
  try {
    const r = await authFetch('/api/paper');
    if (!r) return;
    const d = await r.json();
    const el = document.getElementById('paperSummaryBots');
    if (!d.balances.length) {
      el.innerHTML = '<div style="text-align:center;padding:12px;color:var(--dim)">봇 가동 시 자동 시작됩니다</div>';
      return;
    }
    el.innerHTML = d.balances.map(b => {
      const pnl = b.balance - b.initial;
      const pnlPct = (pnl / b.initial * 100);
      const wr = b.trades > 0 ? (b.wins/b.trades*100).toFixed(1) : '0';
      const label = strategiesCache.find(x=>x.name===b.strategy);
      const name = label ? label.label : b.strategy;
      return `
        <div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;${b !== d.balances[0] ? 'border-top:1px solid var(--border);' : ''}">
          <div>
            <div style="font-weight:600;font-size:13px;">${name}</div>
            <div style="font-size:11px;color:var(--dim);">${b.trades}건 · 승률 ${wr}%</div>
          </div>
          <div style="text-align:right;">
            <div class="${cls(pnl)}" style="font-weight:700;font-size:14px;">${fmt(b.balance)}</div>
            <div class="${cls(pnl)}" style="font-size:11px;">${pnl>=0?'+':''}${pnlPct.toFixed(1)}%</div>
          </div>
        </div>
      `;
    }).join('');
  } catch(e) { console.error('loadPaperSummaryBots:', e); }
}

// ─── TradingView Chart (바이낸스 전체 기간) ────────────
function loadTvChart(containerId, symbol) {
  const sym = symbol || document.getElementById('chartSymbol').value;
  const tvSym = `BINANCE:${sym}.P`;
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = '';

  // TradingView 위젯 스크립트 동적 로드
  if (!window._tvScriptLoaded) {
    const sc = document.createElement('script');
    sc.src = 'https://s3.tradingview.com/tv.js';
    sc.async = true;
    sc.onload = () => { window._tvScriptLoaded = true; _createTvWidget(containerId, tvSym); };
    document.head.appendChild(sc);
  } else {
    _createTvWidget(containerId, tvSym);
  }
}

function _createTvWidget(containerId, tvSym) {
  new TradingView.widget({
    autosize: true,
    symbol: tvSym,
    interval: '15',
    timezone: 'Asia/Seoul',
    theme: 'dark',
    style: '1',
    locale: 'kr',
    toolbar_bg: '#131722',
    enable_publishing: false,
    allow_symbol_change: true,
    container_id: containerId,
    hide_side_toolbar: true,
    studies: ['MAExp@tv-basicstudies','Volume@tv-basicstudies'],
    save_image: false,
  });
}

// 거래 마커 목록 로드 (TradingView 모드용)
async function loadTradeMarkers(sym) {
  const symbol = sym || document.getElementById('chartSymbol').value;
  const el = document.getElementById('tradeMarkersList');
  if (!el) return;
  try {
    const r = await authFetch(`/api/candles/${symbol}?interval=15m&limit=200`);
    if (!r) return;
    const d = await r.json();
    const trades = d.trades || [];
    const positions = d.positions || [];
    if (!trades.length && !positions.length) {
      el.innerHTML = '<div style="color:var(--dim);">거래 기록 없음</div>';
      return;
    }
    let html = '';
    // 열린 포지션
    positions.forEach(p => {
      const c = p.side === 'LONG' ? '#2196f3' : '#ff5722';
      html += `<div style="color:${c};padding:4px 0;border-bottom:1px solid var(--border);">
        <b>${p.strategy.substring(0,12)}</b> ${p.side} @ $${p.entry.toFixed(2)}
        ${p.sl ? ' SL $'+p.sl.toFixed(2) : ''} ${p.tp ? ' TP $'+p.tp.toFixed(2) : ''}
        <span style="color:var(--dim);font-size:10px;">[OPEN]</span>
      </div>`;
    });
    // 최근 거래 (시간표)
    trades.slice(0, 15).forEach(t => {
      const c = t.pnl >= 0 ? '#26a69a' : '#ef5350';
      const dt = t.exit_time ? new Date(t.exit_time * 1000).toLocaleString('ko') : '-';
      html += `<div style="padding:3px 0;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;">
        <span>${t.strategy.substring(0,8)} ${t.side}</span>
        <span style="color:${c};font-weight:600;">$${t.pnl.toFixed(2)} ${t.reason}</span>
        <span style="color:var(--dim);font-size:10px;">${dt}</span>
      </div>`;
    });
    el.innerHTML = html;
  } catch(e) { console.error('loadTradeMarkers:', e); }
}

// 차트 모드 전환
function switchChartMode() {
  loadChart();
}

// ─── Lightweight Chart (거래 마커 오버레이) ────
let chart = null;
let candleSeries = null;
let volumeSeries = null;
let markers = [];

function initChart() {
  const container = document.getElementById('chartContainer');
  if (chart) { chart.remove(); }
  chart = LightweightCharts.createChart(container, {
    layout: { background: { color: '#141420' }, textColor: '#888' },
    grid: { vertLines: { color: '#1e1e30' }, horzLines: { color: '#1e1e30' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Magnet },
    rightPriceScale: { borderColor: '#1e1e30' },
    timeScale: { borderColor: '#1e1e30', timeVisible: true, secondsVisible: false },
    width: container.clientWidth,
    height: 300,
  });
  candleSeries = chart.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350',
    borderUpColor: '#26a69a', borderDownColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
  });
  volumeSeries = chart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: '',
    scaleMargins: { top: 0.85, bottom: 0 },
  });
  // Resize
  new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth });
  }).observe(container);
}

async function loadChart() {
  const sym = document.getElementById('chartSymbol').value;
  try {
    const r = await authFetch(`/api/candles/${sym}?limit=200`);
    if (!r) return;
    const d = await r.json();
    if (!d.candles || !d.candles.length) return;

    if (!chart) initChart();

    candleSeries.setData(d.candles.map(c => ({
      time: c.time, open: c.open, high: c.high, low: c.low, close: c.close,
    })));
    volumeSeries.setData(d.candles.map(c => ({
      time: c.time, value: c.volume,
      color: c.close >= c.open ? 'rgba(38,166,154,0.3)' : 'rgba(239,83,80,0.3)',
    })));

    // 실거래 현재 포지션 → 진입가/SL/TP 수평선
    if (statusCache && statusCache.positions && statusCache.positions[sym]) {
      const info = statusCache.positions[sym];
      const pos = info.position;
      if (pos && pos.entry_price > 0) {
        candleSeries.createPriceLine({
          price: pos.entry_price, color: '#ffd54f', lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Solid,
          axisLabelVisible: true, title: `Entry ${pos.side}`,
        });
        if (pos.sl_price && pos.sl_price > 0) {
          candleSeries.createPriceLine({
            price: pos.sl_price, color: '#ef5350', lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            axisLabelVisible: true, title: 'SL',
          });
        }
        if (pos.tp_price && pos.tp_price > 0) {
          candleSeries.createPriceLine({
            price: pos.tp_price, color: '#26a69a', lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            axisLabelVisible: true, title: 'TP',
          });
        }
      }
    }

    // 실거래 최근 체결 마커
    try {
      const tr = await authFetch(`/api/trades?symbol=${sym}&limit=20`);
      if (tr) {
        const trades = await tr.json();
        markers = [];
        trades.forEach(t => {
          if (t.opened_at) {
            const ts = Math.floor(new Date(t.opened_at).getTime() / 1000);
            const isLong = t.side === 'LONG' || t.side === 'BUY';
            markers.push({
              time: ts, position: isLong ? 'belowBar' : 'aboveBar',
              color: isLong ? '#26a69a' : '#ef5350',
              shape: isLong ? 'arrowUp' : 'arrowDown',
              text: t.side,
            });
          }
          if (t.closed_at) {
            const ts = Math.floor(new Date(t.closed_at).getTime() / 1000);
            markers.push({
              time: ts, position: 'inBar',
              color: t.net_pnl >= 0 ? '#26a69a' : '#ef5350',
              shape: 'circle',
              text: `${t.reason||'close'} ${fmt(t.net_pnl)}`,
            });
          }
        });
        markers.sort((a,b) => a.time - b.time);
        if (markers.length) candleSeries.setMarkers(markers);
      }
    } catch(e) {}

    chart.timeScale().fitContent();
  } catch(e) { console.error('loadChart:', e); }
}

// 심볼 변경 시 차트 리로드
document.getElementById('chartSymbol').addEventListener('change', () => loadChart());

// ─── Trade Chart (거래 현황 페이지) — TradingView ──────────
function loadTradeChart() {
  const sym = document.getElementById('tradeChartSymbol').value;
  loadTvChart('tvTradeChartWrap', sym);
}
document.getElementById('tradeChartSymbol').addEventListener('change', loadTradeChart);

// ─── Paper Trades (거래내역) ─────────────────
async function loadPaperTrades() {
  try {
    const r = await authFetch('/api/paper/trades?limit=30');
    if (!r) return;
    const trades = await r.json();
    const el = document.getElementById('paperTradeList');

    if (!trades.length) {
      el.innerHTML = '<div style="text-align:center;padding:16px;color:var(--dim)">거래 없음</div>';
      return;
    }

    el.innerHTML = trades.map(t => {
      const label = strategiesCache.find(x=>x.name===t.strategy);
      const sname = label ? label.label : t.strategy;
      return `
        <div class="trade-item">
          <div class="trade-info">
            <div class="trade-sym">
              ${t.symbol.replace('USDT','')}
              <span class="pos-side ${t.side==='LONG'?'long':'short'}" style="font-size:10px">${t.side}</span>
              <span style="font-size:10px;color:var(--dim);margin-left:4px;">${t.reason}</span>
            </div>
            <div class="trade-meta">${sname} · ${fmt(t.entry)} → ${fmt(t.exit)}</div>
            <div class="trade-meta">${t.closed_at ? new Date(t.closed_at).toLocaleString('ko',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'}) : ''}</div>
          </div>
          <div>
            <div class="trade-pnl ${cls(t.pnl)}">${fmt(t.pnl, 4)}</div>
          </div>
        </div>
      `;
    }).join('');
  } catch(e) { console.error('loadPaperTrades:', e); }
}

// ─── Settings ────────────────────────────────
async function loadSettingsForm() {
  // Load strategies into dropdown
  if (!strategiesCache.length) await loadStrategies();
  const sel = document.getElementById('sStrategy');
  if (sel.options.length <= 1) {
    sel.innerHTML = strategiesCache.map(s =>
      `<option value="${s.name}">${s.label}</option>`
    ).join('');
  }

  // Fill form with current values
  if (!statusCache) await loadStatus();
  if (statusCache && statusCache.settings) {
    const s = statusCache.settings;
    sel.value = s.strategy;
    document.getElementById('sLeverage').value = s.leverage || '5';
    document.getElementById('sSizePct').value = (parseFloat(s.position_size_pct) * 100).toFixed(0);
    document.getElementById('sTp').value = (parseFloat(s.tp_pct) * 100).toFixed(1);
    document.getElementById('sSl').value = (parseFloat(s.sl_pct) * 100).toFixed(1);
    document.getElementById('sTick').value = s.tick_interval;
  }

  // Strategy description on change
  sel.onchange = () => {
    const info = strategiesCache.find(x=>x.name===sel.value);
    document.getElementById('strategyInfo').textContent = info ? info.description : '';
  };
  sel.onchange();
}

async function saveSettings() {
  const body = {
    strategy: document.getElementById('sStrategy').value,
    leverage: document.getElementById('sLeverage').value,
    position_size_pct: String(Number(document.getElementById('sSizePct').value)/100),
    tp_pct: String(Number(document.getElementById('sTp').value)/100),
    sl_pct: String(Number(document.getElementById('sSl').value)/100),
    tick_interval: document.getElementById('sTick').value,
  };
  await authFetch('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  alert('설정 저장 완료. 전략 변경 시 봇을 재시작하세요.');
  loadStatus();
}

// ─── Position Chart Modal ────────────────────
let _tradeChart = null;

function closeTradeChart() {
  document.getElementById('tradeChartModal').style.display = 'none';
  if (_tradeChart) { _tradeChart.remove(); _tradeChart = null; }
}

async function openTradeChart(tradeId, source='real') {
  const modal = document.getElementById('tradeChartModal');
  const area = document.getElementById('tradeChartArea');
  const titleEl = document.getElementById('tradeChartTitle');
  const infoEl = document.getElementById('tradeChartInfo');

  modal.style.display = 'block';
  area.innerHTML = '<div style="text-align:center;padding:40px;color:var(--dim)">로딩 중...</div>';
  titleEl.textContent = '';
  infoEl.textContent = '';

  try {
    const r = await authFetch(`/api/trade/${tradeId}/chart?source=${source}`);
    if (!r) return;
    const data = await r.json();
    if (data.error) { area.innerHTML = `<div style="text-align:center;padding:40px;color:#f44">${data.error}</div>`; return; }
    if (!data.candles || !data.candles.length) { area.innerHTML = '<div style="text-align:center;padding:40px;color:var(--dim)">캔들 데이터 없음</div>'; return; }

    const t = data.trade;
    const isLong = t.side === 'LONG' || t.side === 'BUY';
    const pnlCls = t.pnl >= 0 ? 'up' : 'down';
    titleEl.innerHTML = `${t.symbol.replace('USDT','')} <span class="pos-side ${isLong?'long':'short'}" style="font-size:11px">${t.side}</span>`;
    infoEl.innerHTML = `${fmt(t.entry_price)} → ${fmt(t.exit_price)} · <span class="${pnlCls}">${fmt(t.pnl)}</span> · ${t.reason || ''}`;

    // 차트 생성
    area.innerHTML = '';
    if (_tradeChart) _tradeChart.remove();
    const isDark = true;
    _tradeChart = LightweightCharts.createChart(area, {
      width: area.clientWidth, height: area.clientHeight,
      layout: { background: { color: '#1a1a2e' }, textColor: '#aaa' },
      grid: { vertLines: { color: '#2a2a3e' }, horzLines: { color: '#2a2a3e' } },
      crosshair: { mode: 0 },
      timeScale: { timeVisible: true, secondsVisible: false },
    });

    const candleSeries = _tradeChart.addCandlestickSeries({
      upColor: '#26a69a', downColor: '#ef5350',
      borderUpColor: '#26a69a', borderDownColor: '#ef5350',
      wickUpColor: '#26a69a', wickDownColor: '#ef5350',
    });
    candleSeries.setData(data.candles);

    // SL 수평선 (빨간 점선)
    if (t.sl_price) {
      candleSeries.createPriceLine({
        price: t.sl_price, color: '#ef5350', lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true, title: 'SL',
      });
    }
    // TP 수평선 (초록 점선)
    if (t.tp_price) {
      candleSeries.createPriceLine({
        price: t.tp_price, color: '#26a69a', lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true, title: 'TP',
      });
    }
    // 진입가 수평선 (노란 실선)
    candleSeries.createPriceLine({
      price: t.entry_price, color: '#ffd54f', lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Solid,
      axisLabelVisible: true, title: 'Entry',
    });
    // 청산가 수평선 (흰색 점선)
    if (t.exit_price) {
      candleSeries.createPriceLine({
        price: t.exit_price, color: '#ffffff88', lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted,
        axisLabelVisible: true, title: 'Exit',
      });
    }

    // 마커: 진입 & 청산
    const markers = [];
    if (t.opened_at) {
      const openTs = Math.floor(new Date(t.opened_at).getTime() / 1000);
      markers.push({
        time: openTs, position: isLong ? 'belowBar' : 'aboveBar',
        color: isLong ? '#26a69a' : '#ef5350',
        shape: isLong ? 'arrowUp' : 'arrowDown',
        text: isLong ? 'LONG' : 'SHORT',
      });
    }
    if (t.closed_at && t.exit_price) {
      const closeTs = Math.floor(new Date(t.closed_at).getTime() / 1000);
      markers.push({
        time: closeTs, position: isLong ? 'aboveBar' : 'belowBar',
        color: '#ffd54f', shape: 'circle',
        text: (t.reason || 'close').toUpperCase(),
      });
    }
    if (markers.length) candleSeries.setMarkers(markers.sort((a,b) => a.time - b.time));

    _tradeChart.timeScale().fitContent();

    // 리사이즈 대응
    const ro = new ResizeObserver(() => {
      if (_tradeChart) _tradeChart.applyOptions({ width: area.clientWidth, height: area.clientHeight });
    });
    ro.observe(area);
  } catch(e) {
    area.innerHTML = `<div style="text-align:center;padding:40px;color:#f44">${e.message}</div>`;
  }
}

// ─── Auto refresh ────────────────────────────
loadStrategies().then(() => loadStatus());
setInterval(() => {
  const activePage = document.querySelector('.page.active');
  if (activePage.id === 'page-home' || activePage.id === 'page-bots') loadStatus();
  else if (activePage.id === 'page-trades') loadTrades();
  else if (activePage.id === 'page-paper') loadPaper();
}, 10000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(
        "src.dashboard.mobile:app",
        host="0.0.0.0",
        port=8503,
        reload=False,
    )
