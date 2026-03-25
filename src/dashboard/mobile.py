"""경량 모바일 대시보드 — FastAPI + 순수 HTML/CSS.

Streamlit 대비 ~30MB로 1core/1GB 서버에서도 부담 없이 실행.
모바일 최적화 UI, PWA 지원, 자동 새로고침.

실행: python -m src.dashboard.mobile
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from src.core import database as db
from src.exchange.futures_client import FuturesClient

db.init_db()

app = FastAPI(title="Auto-Trader Mobile")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]


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
async def api_status():
    try:
        data = await _fetch_live()
        bot_states = db.get_all_bot_states()
        settings = db.get_all_settings()
        return JSONResponse({
            "ok": True,
            "account": data["account"],
            "positions": data["positions"],
            "bots": bot_states,
            "settings": {
                "strategy": settings.get("strategy", ""),
                "leverage": settings.get("leverage", "5"),
                "tick_interval": settings.get("tick_interval", "15"),
            },
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e)})


@app.get("/api/trades")
async def api_trades(symbol: str | None = None, limit: int = 50):
    trades = db.get_trades(symbol=symbol, limit=limit)
    return JSONResponse([{
        "id": t.id, "symbol": t.symbol, "side": t.side,
        "entry": t.entry_price, "exit": t.exit_price,
        "qty": t.quantity, "pnl": round(t.pnl or 0, 2),
        "fee": round(t.fee or 0, 4),
        "net_pnl": round(t.net_pnl if t.net_pnl is not None else (t.pnl or 0), 2),
        "pnl_pct": round(t.pnl_pct or 0, 2),
        "strategy": t.strategy or "",
        "closed_at": t.closed_at.isoformat() if t.closed_at else "",
    } for t in trades])


@app.get("/api/paper")
async def api_paper():
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


@app.post("/api/bot/{symbol}/start")
async def api_bot_start(symbol: str):
    db.set_bot_running(symbol, True)
    return JSONResponse({"ok": True, "symbol": symbol, "running": True})


@app.post("/api/bot/{symbol}/stop")
async def api_bot_stop(symbol: str):
    db.set_bot_running(symbol, False)
    return JSONResponse({"ok": True, "symbol": symbol, "running": False})


@app.post("/api/settings")
async def api_settings_save(request: Request):
    body = await request.json()
    for key, value in body.items():
        db.set_setting(key, str(value))
    return JSONResponse({"ok": True})


# ─── Mobile HTML SPA ──────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return MOBILE_HTML


MOBILE_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="theme-color" content="#0a0a0f">
<title>Auto-Trader</title>
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
    <div class="card-title">최근 거래</div>
    <div id="recentTrades"><div class="loading"><div class="spinner"></div></div></div>
  </div>
</div>

<div class="page" id="page-bots">
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
    <div class="card-title">거래 요약</div>
    <div class="metrics" id="tradeSummary"></div>
  </div>
  <div class="card">
    <div class="card-title">거래 내역</div>
    <div id="tradeList"><div class="loading"><div class="spinner"></div></div></div>
  </div>
</div>

<div class="page" id="page-paper">
  <div class="card">
    <div class="card-title">가상매매 현황</div>
    <div id="paperList"><div class="loading"><div class="spinner"></div></div></div>
  </div>
</div>

<div class="page" id="page-settings">
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
    <span class="icon">📋</span>거래
  </button>
  <button onclick="showPage('paper',this)">
    <span class="icon">🧪</span>가상매매
  </button>
  <button onclick="showPage('settings',this)">
    <span class="icon">⚙️</span>설정
  </button>
</div>

<script>
// ─── Navigation ──────────────────────────────
function showPage(name, btn) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-bar button').forEach(b => b.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  if (btn) btn.classList.add('active');
  // Load data for page
  if (name==='home') loadStatus();
  else if (name==='trades') loadTrades();
  else if (name==='paper') loadPaper();
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
    const r = await fetch('/api/status');
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

  // Setting summary
  const s = d.settings;
  document.getElementById('settingSummary').innerHTML = `
    <div style="font-size:13px;line-height:2;">
      전략: <b>${s.strategy}</b><br>
      레버리지: <b>x${s.leverage}</b><br>
      분석주기: <b>${s.tick_interval}초</b>
    </div>
  `;
}

async function loadRecentTrades() {
  try {
  const r = await fetch('/api/trades?limit=5');
  const trades = await r.json();
  if (!trades.length) {
    document.getElementById('recentTrades').innerHTML =
      '<div style="text-align:center;padding:16px;color:var(--dim)">거래 없음</div>';
    return;
  }
  document.getElementById('recentTrades').innerHTML = trades.map(t => `
    <div class="trade-item">
      <div class="trade-info">
        <div class="trade-sym">${t.symbol.replace('USDT','')} <span class="pos-side ${t.side==='BUY'||t.side==='LONG'?'long':'short'}" style="font-size:10px">${t.side}</span></div>
        <div class="trade-meta">${t.strategy || ''} · ${t.closed_at ? new Date(t.closed_at).toLocaleString('ko',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'}) : ''}</div>
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
  await fetch(`/api/bot/${symbol}/${start?'start':'stop'}`, {method:'POST'});
  loadStatus();
}

// ─── Trades ──────────────────────────────────
async function loadTrades() {
  const r = await fetch('/api/trades?limit=50');
  const trades = await r.json();

  if (!trades.length) {
    document.getElementById('tradeSummary').innerHTML = '';
    document.getElementById('tradeList').innerHTML =
      '<div style="text-align:center;padding:16px;color:var(--dim)">거래 없음</div>';
    return;
  }

  const totalNet = trades.reduce((s,t)=>s+t.net_pnl,0);
  const totalFee = trades.reduce((s,t)=>s+t.fee,0);
  const wins = trades.filter(t=>t.net_pnl>0).length;
  const winRate = (wins/trades.length*100);

  document.getElementById('tradeSummary').innerHTML = `
    <div class="metric sm"><div class="value ${cls(totalNet)}">${fmt(totalNet)}</div><div class="label">순손익</div></div>
    <div class="metric sm"><div class="value">${fmt(totalFee,4)}</div><div class="label">수수료</div></div>
    <div class="metric sm"><div class="value">${winRate.toFixed(1)}%</div><div class="label">승률</div></div>
    <div class="metric sm"><div class="value">${trades.length}건</div><div class="label">거래수</div></div>
  `;

  document.getElementById('tradeList').innerHTML = trades.map(t => `
    <div class="trade-item">
      <div class="trade-info">
        <div class="trade-sym">${t.symbol.replace('USDT','')} <span class="pos-side ${t.side==='BUY'||t.side==='LONG'?'long':'short'}" style="font-size:10px">${t.side}</span></div>
        <div class="trade-meta">${fmt(t.entry)} → ${fmt(t.exit)} · ${t.strategy || ''}</div>
        <div class="trade-meta">${t.closed_at ? new Date(t.closed_at).toLocaleString('ko') : ''}</div>
      </div>
      <div>
        <div class="trade-pnl ${cls(t.net_pnl)}">${fmt(t.net_pnl)}</div>
        <div class="trade-pnl-pct ${cls(t.pnl_pct)}">${pct(t.pnl_pct)}</div>
      </div>
    </div>
  `).join('');
}

// ─── Paper Trading ───────────────────────────
async function loadPaper() {
  const r = await fetch('/api/paper');
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
    return `
      <div class="paper-row">
        <div class="paper-header">
          <span class="paper-name">${b.strategy}</span>
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

// ─── Settings ────────────────────────────────
async function loadSettingsForm() {
  if (!statusCache) await loadStatus();
  // Also load full settings from current DB
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    if (d.ok) {
      const s = d.settings;
      document.getElementById('sLeverage').value = s.leverage || '5';
    }
  } catch(e) {}
}

async function saveSettings() {
  const body = {
    leverage: document.getElementById('sLeverage').value,
    position_size_pct: String(Number(document.getElementById('sSizePct').value)/100),
    tp_pct: String(Number(document.getElementById('sTp').value)/100),
    sl_pct: String(Number(document.getElementById('sSl').value)/100),
    tick_interval: document.getElementById('sTick').value,
  };
  await fetch('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  alert('설정 저장 완료');
}

// ─── Auto refresh ────────────────────────────
loadStatus();
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
