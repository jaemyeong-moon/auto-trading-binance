"""Streamlit dashboard — 선물 스캘핑 봇 제어 + 실시간 분석 모니터링."""

import asyncio
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import database as db
from src.core.models import SignalType
from src.exchange.futures_client import FuturesClient
from src.core.futures_engine import FuturesEngine
from src.strategies.scalper import MomentumFlipScalper
from src.strategies.technical import TechnicalStrategy
from src.backtesting.backtest import Backtester

st.set_page_config(page_title="Auto-Trader Futures", page_icon="⚡", layout="wide")
db.init_db()

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─── Background bot ───────────────────────────────────────
def _bot_thread(symbol: str):
    async def _run():
        client = FuturesClient()
        await client.connect()
        engine = FuturesEngine(client=client)
        await engine.start_symbol(symbol)
        while db.is_bot_running(symbol):
            await asyncio.sleep(1)
        await engine.stop_symbol(symbol)
        await client.disconnect()
    asyncio.run(_run())


def start_bot(symbol: str):
    db.set_bot_running(symbol, True)
    t = threading.Thread(target=_bot_thread, args=(symbol,), daemon=True)
    t.start()


def stop_bot(symbol: str):
    db.set_bot_running(symbol, False)


# ─── Live data ─────────────────────────────────────────────
def fetch_all(symbols: list[str]) -> dict:
    """계좌 요약 + 심볼별 가격/포지션 한번에 조회."""
    async def _fetch():
        client = FuturesClient()
        await client.connect()
        try:
            account = await client.get_account_summary()
            sym_data = {}
            for s in symbols:
                try:
                    price = await client.get_price(s)
                    pos = await client.get_position(s)
                    sym_data[s] = {"price": price, "position": pos}
                except Exception:
                    sym_data[s] = {"price": 0.0, "position": None}
            return {"account": account, "symbols": sym_data}
        finally:
            await client.disconnect()
    return run_async(_fetch())


def fetch_analysis(symbol: str) -> dict:
    """1분봉 + EMA + 전략 신호."""
    async def _fetch():
        client = FuturesClient()
        await client.connect()
        try:
            return await client.get_candles(symbol, interval="1m", limit=100)
        finally:
            await client.disconnect()

    candles = run_async(_fetch())
    if candles.empty:
        return {"candles": candles, "signal": None, "indicators": {}}

    df = candles.copy()
    close = df["close"]
    volume = df["volume"]

    ema3 = close.ewm(span=3, adjust=False).mean()
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()

    vol_avg = volume.rolling(20).mean()
    vol_ratio = (volume / vol_avg).iloc[-1]
    ema_gap = ((ema3.iloc[-1] - ema8.iloc[-1]) / ema8.iloc[-1]) * 100

    cross_up = ema3.iloc[-2] <= ema8.iloc[-2] and ema3.iloc[-1] > ema8.iloc[-1]
    cross_down = ema3.iloc[-2] >= ema8.iloc[-2] and ema3.iloc[-1] < ema8.iloc[-1]

    trend = "상승" if ema3.iloc[-1] > ema8.iloc[-1] > ema20.iloc[-1] else \
            "하락" if ema3.iloc[-1] < ema8.iloc[-1] < ema20.iloc[-1] else "혼조"

    strategy = MomentumFlipScalper()
    signal = strategy.evaluate(symbol, candles)

    df["ema3"] = ema3
    df["ema8"] = ema8
    df["ema20"] = ema20

    return {
        "candles": df, "signal": signal,
        "indicators": {
            "ema3": ema3.iloc[-1], "ema8": ema8.iloc[-1], "ema20": ema20.iloc[-1],
            "ema_gap_pct": ema_gap, "cross_up": cross_up, "cross_down": cross_down,
            "trend": trend, "vol_ratio": vol_ratio, "vol_strong": vol_ratio > 1.2,
            "price": close.iloc[-1],
        },
    }


def load_trades_df(symbol: str | None = None) -> pd.DataFrame:
    trades = db.get_trades(symbol=symbol, limit=200)
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([{
        "ID": t.id, "심볼": t.symbol, "방향": t.side,
        "진입가": t.entry_price, "청산가": t.exit_price, "수량": t.quantity,
        "손익(USDT)": round(t.pnl, 2) if t.pnl is not None else 0,
        "수수료": round(t.fee, 4) if t.fee is not None else 0,
        "순손익": round(t.net_pnl if t.net_pnl is not None else (t.pnl or 0), 2),
        "손익(%)": round(t.pnl_pct, 2) if t.pnl_pct is not None else 0,
        "전략": t.strategy, "청산시간": t.closed_at,
    } for t in trades])


# ─── Sidebar ───────────────────────────────────────────────
st.sidebar.title("⚡ Futures Scalper")

# 현재 설정 로드
current_settings = db.get_all_settings()

st.sidebar.caption(
    f"x{current_settings['leverage']} 레버리지 · "
    f"잔고 {float(current_settings['position_size_pct'])*100:.0f}% 투자 · "
    f"1분봉"
)
st.sidebar.divider()
page = st.sidebar.radio("페이지", ["실시간 현황", "시뮬레이션", "백테스팅", "거래 내역", "설정"])
auto_refresh = st.sidebar.checkbox("자동 새로고침 (10초)", value=False)

# 사이드바 하단에 현재 설정 요약
st.sidebar.divider()
st.sidebar.markdown("**현재 설정**")
st.sidebar.markdown(
    f"- 투자 비율: **{float(current_settings['position_size_pct'])*100:.0f}%**\n"
    f"- 레버리지: **x{current_settings['leverage']}**\n"
    f"- 익절: **{float(current_settings['tp_pct'])*100:.1f}%**\n"
    f"- 손절: **{float(current_settings['sl_pct'])*100:.1f}%**\n"
    f"- 분석주기: **{current_settings['tick_interval']}초**"
)


# ═══════════════════════════════════════════════════════════
#  실시간 현황
# ═══════════════════════════════════════════════════════════
if page == "실시간 현황":
    st.title("⚡ 실시간 현황")

    # ── 봇 제어 ──
    st.subheader("봇 제어")
    bot_states = db.get_all_bot_states()
    cols = st.columns(len(SYMBOLS))
    for i, symbol in enumerate(SYMBOLS):
        with cols[i]:
            running = bot_states.get(symbol, False)
            if running:
                st.markdown(f"**{symbol}**  \n🟢 실행중")
                if st.button("중지", key=f"stop_{symbol}", type="secondary"):
                    stop_bot(symbol)
                    st.rerun()
            else:
                st.markdown(f"**{symbol}**  \n🔴 중지")
                if st.button("시작", key=f"start_{symbol}", type="primary"):
                    start_bot(symbol)
                    st.rerun()

    st.divider()

    # ── 계좌 + 포지션 조회 ──
    active_symbols = [s for s, r in bot_states.items() if r]
    display_symbols = active_symbols if active_symbols else SYMBOLS[:2]

    with st.spinner("거래소 데이터 조회 중..."):
        data = fetch_all(display_symbols)

    account = data["account"]
    sym_data = data["symbols"]

    # ── 계좌 요약 (상단 고정) ──
    st.subheader("💰 계좌 현황")

    total_margin = sum(
        sym_data[s]["position"]["margin"]
        for s in display_symbols
        if sym_data[s]["position"]
    )
    total_notional = sum(
        sym_data[s]["position"]["notional"]
        for s in display_symbols
        if sym_data[s]["position"]
    )
    total_upnl = sum(
        sym_data[s]["position"]["unrealized_pnl"]
        for s in display_symbols
        if sym_data[s]["position"]
    )
    active_count = sum(1 for s in display_symbols if sym_data[s]["position"])

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("총 잔고", f"${account['balance']:,.2f}")
    a2.metric("가용 잔고", f"${account['available']:,.2f}")
    a3.metric("투입 증거금", f"${total_margin:,.2f}")
    a4.metric("포지션 규모 (x5)", f"${total_notional:,.2f}")

    upnl_color = "normal" if total_upnl >= 0 else "inverse"
    a5.metric("미실현 손익", f"${total_upnl:+,.2f}", delta=f"{active_count}개 포지션")

    st.divider()

    # ── 심볼별 상세 ──
    for symbol in display_symbols:
        sd = sym_data.get(symbol, {})
        price = sd.get("price", 0)
        pos = sd.get("position")

        st.subheader(f"📈 {symbol}")

        # 분석 데이터
        analysis = fetch_analysis(symbol)
        ind = analysis["indicators"]
        signal = analysis["signal"]
        candles = analysis["candles"]

        # ── 3열: 가격+포지션 | 투자현황 | 분석판단 ──
        col_left, col_mid, col_right = st.columns([2, 2, 2])

        with col_left:
            st.markdown(f"### ${price:,.2f}")
            if pos:
                side = pos["side"]
                entry = pos["entry_price"]
                mark = pos["mark_price"]
                upnl = pos["unrealized_pnl"]
                qty = pos["quantity"]

                if side == "LONG":
                    pnl_pct = ((mark - entry) / entry) * 100
                else:
                    pnl_pct = ((entry - mark) / entry) * 100

                side_label = "🟢 LONG" if side == "LONG" else "🔴 SHORT"
                pnl_color = "#26a69a" if upnl >= 0 else "#ef5350"

                st.markdown(f"**{side_label}** · 수량 `{qty}`")
                st.markdown(
                    f'<div style="font-size:1.5em; color:{pnl_color}; font-weight:bold;">'
                    f'${upnl:+,.2f} ({pnl_pct:+.2f}%)</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("포지션 없음")

        with col_mid:
            if pos:
                st.markdown("**투자 현황**")
                # ATR 기반 SL/TP 가격 계산
                _atr_val = None
                if not candles.empty and len(candles) > 14:
                    import ta as _ta
                    _atr_val = _ta.volatility.AverageTrueRange(
                        candles["high"], candles["low"], candles["close"], window=14
                    ).average_true_range().iloc[-1]

                sl_mult = float(current_settings.get("auto_sl_mult", "8.0"))
                tp_mult = float(current_settings.get("auto_tp_mult", "12.0"))

                sl_price_str = "—"
                tp_price_str = "—"
                if _atr_val and _atr_val > 0:
                    if side == "LONG":
                        sl_price = entry - _atr_val * sl_mult
                        tp_price = entry + _atr_val * tp_mult
                    else:
                        sl_price = entry + _atr_val * sl_mult
                        tp_price = entry - _atr_val * tp_mult
                    sl_price_str = f"${sl_price:,.2f}"
                    tp_price_str = f"${tp_price:,.2f}"

                lev = int(float(current_settings['leverage']))
                st.markdown(f"""
| 항목 | 값 |
|------|------|
| 진입가 | `${entry:,.2f}` |
| 현재가(Mark) | `${pos['mark_price']:,.2f}` |
| 🟢 익절가 | `{tp_price_str}` ({tp_mult:.0f}x ATR) |
| 🔴 손절가 | `{sl_price_str}` ({sl_mult:.0f}x ATR) |
| 증거금 | `${pos['margin']:,.2f}` |
| 포지션 규모 | `${pos['notional']:,.2f}` |
| 레버리지 효과 | x{lev} → 수익률 `{pnl_pct*lev:+.1f}%` |
""")
            else:
                st.markdown("**투자 현황**")
                st.caption("진입 대기중")

        with col_right:
            if ind:
                trend_color = {"상승": "#26a69a", "하락": "#ef5350", "혼조": "#ffa726"}
                tc = trend_color.get(ind["trend"], "gray")

                vol_icon = "🔥" if ind["vol_strong"] else "💤"
                cross_text = "⬆️ 골든" if ind["cross_up"] else \
                             "⬇️ 데드" if ind["cross_down"] else "➡️ 없음"

                if signal and signal.type.value != "hold":
                    sig_text = "🟢 LONG" if signal.type.value == "buy" else "🔴 SHORT"
                    contrarian = " (역추세)" if signal.metadata.get("contrarian") else ""
                    st.markdown(
                        f'**다음 판단:** <span style="color:{tc}; font-size:1.2em;">'
                        f'{sig_text}{contrarian}</span> ({signal.confidence:.0%})',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("**다음 판단:** ⏸️ 유지")

                st.markdown(f"""
| 분석 | 값 |
|------|------|
| 추세 | <span style="color:{tc}">**{ind['trend']}**</span> |
| EMA(3/8) 갭 | `{ind['ema_gap_pct']:+.3f}%` |
| EMA 크로스 | {cross_text} |
| 거래량 | {vol_icon} x`{ind['vol_ratio']:.1f}` |
""", unsafe_allow_html=True)

        # ── 1분봉 차트 ──
        if not candles.empty and "ema3" in candles.columns:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.03, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(
                x=candles.index, open=candles["open"], high=candles["high"],
                low=candles["low"], close=candles["close"], name="Price",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=candles.index, y=candles["ema3"],
                name="EMA(3)", line=dict(color="#ffeb3b", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=candles.index, y=candles["ema8"],
                name="EMA(8)", line=dict(color="#ff9800", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=candles.index, y=candles["ema20"],
                name="EMA(20)", line=dict(color="#9c27b0", width=1, dash="dash")), row=1, col=1)

            # 진입가 / SL / TP 라인
            if pos:
                fig.add_hline(y=pos["entry_price"], line_dash="dot",
                              line_color="white", annotation_text=f"진입 ${pos['entry_price']:,.0f}",
                              row=1, col=1)
                if _atr_val and _atr_val > 0:
                    fig.add_hline(y=tp_price, line_dash="dash",
                                  line_color="#26a69a", annotation_text=f"TP ${tp_price:,.0f}",
                                  row=1, col=1)
                    fig.add_hline(y=sl_price, line_dash="dash",
                                  line_color="#ef5350", annotation_text=f"SL ${sl_price:,.0f}",
                                  row=1, col=1)

            vol_colors = ["#ef5350" if c < o else "#26a69a"
                          for c, o in zip(candles["close"], candles["open"])]
            fig.add_trace(go.Bar(x=candles.index, y=candles["volume"],
                marker_color=vol_colors, opacity=0.6, name="Volume"), row=2, col=1)

            fig.update_layout(template="plotly_dark", height=400,
                              xaxis_rangeslider_visible=False,
                              margin=dict(l=0, r=0, t=10, b=0),
                              legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

    # ── 최근 거래 ──
    st.subheader("최근 거래")
    trades_df = load_trades_df()
    if trades_df.empty:
        st.info("체결된 거래가 없습니다.")
    else:
        total_pnl = trades_df["손익(USDT)"].sum()
        total_fee = trades_df["수수료"].sum()
        total_net = trades_df["순손익"].sum()
        total_count = len(trades_df)
        win_count = len(trades_df[trades_df["순손익"] > 0])
        lose_count = len(trades_df[trades_df["순손익"] < 0])
        win_rate = (win_count / total_count * 100) if total_count > 0 else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("순손익", f"${total_net:+,.2f}")
        c2.metric("총 수수료", f"${total_fee:,.4f}")
        c3.metric("승률", f"{win_rate:.1f}%")
        c4.metric("거래 수", f"{total_count}회")
        c5.metric("수익/손실", f"{win_count}W / {lose_count}L")
        st.dataframe(trades_df, use_container_width=True)

    if auto_refresh:
        time.sleep(10)
        st.rerun()


# ═══════════════════════════════════════════════════════════
#  시뮬레이션 (페이퍼 트레이딩)
# ═══════════════════════════════════════════════════════════
elif page == "시뮬레이션":
    st.title("🧪 시뮬레이션")

    from src.strategies.registry import get_strategy, list_strategies
    from src.core.database import PaperBalance, PaperPosition, PaperTrade, get_session
    import ta as _ta

    sim_tab1, sim_tab2 = st.tabs(["📊 실시간 가상매매", "🔬 원샷 시뮬레이션"])

    # ══════════════════════════════════════════════════════
    #  탭 1: 실시간 가상매매 (페이퍼 트레이딩) — 지속적 데이터 축적
    # ══════════════════════════════════════════════════════
    with sim_tab1:
        st.subheader("실시간 가상매매")
        st.caption(
            "모든 전략이 $200 가상자본으로 실시간 매매 중. "
            "봇 가동 중 자동으로 데이터가 쌓입니다."
        )

        # ── 전략별 잔고 요약 ──
        with get_session() as session:
            balances = session.query(PaperBalance).all()
            positions = session.query(PaperPosition).all()
            trades_all = session.query(PaperTrade).order_by(
                PaperTrade.closed_at.desc()).limit(200).all()

            bal_data = {b.strategy: b for b in balances}
            pos_data = {}
            for p in positions:
                pos_data.setdefault(p.strategy, []).append(p)
            trade_data = {}
            for t in trades_all:
                trade_data.setdefault(t.strategy, []).append(t)

        if not bal_data:
            st.info("아직 가상매매 데이터가 없습니다. 봇이 가동되면 자동으로 시작됩니다.")
        else:
            # 전체 비교 테이블
            strats = list_strategies()
            compare_rows = []
            for s_info in strats:
                sname = s_info["name"]
                b = bal_data.get(sname)
                if not b:
                    continue
                pnl = b.balance - b.initial_balance
                pnl_pct = pnl / b.initial_balance * 100
                wr = (b.wins / b.total_trades * 100) if b.total_trades > 0 else 0
                compare_rows.append({
                    "전략": s_info["label"],
                    "잔고": f"${b.balance:,.2f}",
                    "손익": f"${pnl:+,.2f}",
                    "수익률": f"{pnl_pct:+.1f}%",
                    "거래수": b.total_trades,
                    "승률": f"{wr:.1f}%",
                    "W/L": f"{b.wins}/{b.losses}",
                })

            if compare_rows:
                st.dataframe(pd.DataFrame(compare_rows), use_container_width=True,
                             hide_index=True)

            # ── 전략별 상세 ──
            for s_info in strats:
                sname = s_info["name"]
                b = bal_data.get(sname)
                if not b:
                    continue

                pnl = b.balance - b.initial_balance
                pnl_pct = pnl / b.initial_balance * 100
                wr = (b.wins / b.total_trades * 100) if b.total_trades > 0 else 0

                with st.expander(
                    f"{'🟢' if pnl >= 0 else '🔴'} {s_info['label']} — "
                    f"${b.balance:,.2f} ({pnl_pct:+.1f}%)",
                    expanded=False,
                ):
                    # 메트릭
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("잔고", f"${b.balance:,.2f}",
                               delta=f"${pnl:+,.2f}")
                    mc2.metric("거래수", f"{b.total_trades}건")
                    mc3.metric("승률", f"{wr:.1f}%")
                    mc4.metric("W/L", f"{b.wins}/{b.losses}")

                    # 현재 포지션
                    s_positions = pos_data.get(sname, [])
                    if s_positions:
                        st.markdown("**열린 포지션:**")
                        for p in s_positions:
                            st.markdown(
                                f"- {p.symbol} **{p.side}** @ `${p.entry_price:,.2f}` "
                                f"| SL `${p.sl_price:,.2f}` | TP `${p.tp_price:,.2f}`"
                            )

                    # 최근 거래
                    s_trades = trade_data.get(sname, [])
                    if s_trades:
                        st.markdown("**최근 거래:**")
                        tdf = pd.DataFrame([{
                            "심볼": t.symbol, "방향": t.side,
                            "진입": f"${t.entry_price:,.2f}",
                            "청산": f"${t.exit_price:,.2f}" if t.exit_price else "—",
                            "SL": f"${t.sl_price:,.2f}" if t.sl_price else "—",
                            "TP": f"${t.tp_price:,.2f}" if t.tp_price else "—",
                            "손익": f"${t.net_pnl:+,.4f}" if t.net_pnl is not None else "—",
                            "사유": t.reason or "—",
                            "시간": t.closed_at,
                        } for t in s_trades[:20]])
                        st.dataframe(tdf, use_container_width=True, hide_index=True)
                    else:
                        st.caption("아직 완료된 거래 없음")

            # 리셋 버튼
            st.divider()
            if st.button("🔄 가상매매 초기화 ($200 리셋)", type="secondary"):
                with get_session() as session:
                    session.query(PaperTrade).delete()
                    session.query(PaperPosition).delete()
                    session.query(PaperBalance).delete()
                    session.commit()
                st.success("초기화 완료. 봇이 가동되면 다시 시작됩니다.")
                st.rerun()

    # ══════════════════════════════════════════════════════
    #  탭 2: 원샷 시뮬레이션 (기존)
    # ══════════════════════════════════════════════════════
    with sim_tab2:
        st.subheader("원샷 시뮬레이션")
        st.caption("실제 시장 데이터로 전략을 한번에 테스트합니다. 실제 주문 없음.")

        # ── 설정 ──
        sim_col1, sim_col2, sim_col3 = st.columns(3)
    sim_symbols = sim_col1.multiselect(
        "심볼", SYMBOLS, default=["BTCUSDT", "ETHUSDT"])
    strategies_list = list_strategies()
    strategy_names = [s["name"] for s in strategies_list]
    sim_strategy_name = sim_col2.selectbox(
        "전략",
        options=strategy_names,
        format_func=lambda x: next(
            (si["label"] for si in strategies_list if si["name"] == x), x),
        index=strategy_names.index(current_settings.get("strategy", strategy_names[0]))
        if current_settings.get("strategy") in strategy_names else 0,
        key="sim_strategy",
    )
    sim_capital = sim_col3.number_input(
        "가상 자본 (USDT)", value=200.0, step=50.0, min_value=50.0)

    sim_col4, sim_col5 = st.columns(2)
    sim_leverage = sim_col4.select_slider(
        "레버리지", options=[1, 2, 3, 5, 7, 10, 15, 20], value=7, key="sim_lev")
    sim_candle_count = sim_col5.slider(
        "분석 캔들 수 (1분봉)", min_value=100, max_value=1000, value=500, step=100)

    if st.button("시뮬레이션 실행", type="primary"):
        strategy = get_strategy(sim_strategy_name)
        sl_mult = getattr(strategy, "SL_ATR_MULT", 8.0)
        tp_mult = getattr(strategy, "TP_ATR_MULT", 12.0)

        all_results = []

        for symbol in sim_symbols:
            with st.spinner(f"{symbol} 데이터 로딩..."):
                async def _fetch_sim(sym=symbol):
                    client = FuturesClient()
                    await client.connect()
                    try:
                        candles = await client.get_candles(
                            sym, interval="1m", limit=sim_candle_count)
                        htf = await client.get_candles(
                            sym, interval="15m", limit=100)
                        return candles, htf
                    finally:
                        await client.disconnect()
                candles, htf = run_async(_fetch_sim())

            if candles.empty or len(candles) < 50:
                st.warning(f"{symbol}: 데이터 부족")
                continue

            # ATR 계산
            atr_series = _ta.volatility.AverageTrueRange(
                candles["high"], candles["low"], candles["close"], window=14
            ).average_true_range()

            # ── 시뮬레이션 루프 ──
            sim_balance = sim_capital
            sim_trades = []
            sim_position = None  # {side, entry, qty, entry_atr, entry_idx}
            sim_equity = [sim_capital]
            sim_strategy = get_strategy(sim_strategy_name)  # 심볼별 독립 인스턴스

            warmup = 50  # 지표 안정화 구간

            for i in range(warmup, len(candles)):
                window = candles.iloc[:i+1]
                price = float(window.iloc[-1]["close"])
                atr = float(atr_series.iloc[i]) if i < len(atr_series) and not pd.isna(atr_series.iloc[i]) else 0

                if atr <= 0:
                    sim_equity.append(sim_balance)
                    continue

                # 포지션 있으면 SL/TP 체크
                if sim_position:
                    p = sim_position
                    entry_atr = p["entry_atr"]
                    sl_dist = entry_atr * sl_mult
                    tp_dist = entry_atr * tp_mult

                    if p["side"] == "LONG":
                        sl_price = p["entry"] - sl_dist
                        tp_price = p["entry"] + tp_dist
                    else:
                        sl_price = p["entry"] + sl_dist
                        tp_price = p["entry"] - tp_dist

                    hit_sl = (p["side"] == "LONG" and price <= sl_price) or \
                             (p["side"] == "SHORT" and price >= sl_price)
                    hit_tp = (p["side"] == "LONG" and price >= tp_price) or \
                             (p["side"] == "SHORT" and price <= tp_price)

                    if hit_sl or hit_tp:
                        exit_price = sl_price if hit_sl else tp_price
                        if p["side"] == "LONG":
                            pnl = (exit_price - p["entry"]) * p["qty"]
                        else:
                            pnl = (p["entry"] - exit_price) * p["qty"]
                        fee = p["entry"] * p["qty"] * 0.0008
                        sim_balance += pnl - fee
                        sim_trades.append({
                            "symbol": symbol, "side": p["side"],
                            "entry": p["entry"], "exit": exit_price,
                            "sl": sl_price, "tp": tp_price,
                            "pnl": round(pnl - fee, 4),
                            "reason": "SL" if hit_sl else "TP",
                            "entry_idx": p["entry_idx"], "exit_idx": i,
                        })
                        sim_strategy.record_result(pnl - fee)
                        sim_position = None

                # 포지션 없으면 전략 평가
                if not sim_position:
                    htf_window = htf if htf is not None and not htf.empty else None
                    signal = sim_strategy.evaluate(symbol, window, htf_window)

                    if signal.type in (SignalType.BUY, SignalType.SELL):
                        invest = sim_balance * 0.3
                        qty = (invest * sim_leverage) / price
                        if invest >= 5:
                            side = "LONG" if signal.type == SignalType.BUY else "SHORT"
                            sl_d = atr * sl_mult
                            tp_d = atr * tp_mult
                            if side == "LONG":
                                pos_sl = price - sl_d
                                pos_tp = price + tp_d
                            else:
                                pos_sl = price + sl_d
                                pos_tp = price - tp_d
                            sim_position = {
                                "side": side, "entry": price,
                                "qty": qty, "entry_atr": atr,
                                "entry_idx": i,
                                "sl": pos_sl, "tp": pos_tp,
                            }

                # 미실현 손익 반영한 equity
                if sim_position:
                    p = sim_position
                    if p["side"] == "LONG":
                        unrealized = (price - p["entry"]) * p["qty"]
                    else:
                        unrealized = (p["entry"] - price) * p["qty"]
                    sim_equity.append(sim_balance + unrealized)
                else:
                    sim_equity.append(sim_balance)

            # 미청산 포지션 강제 종료
            if sim_position:
                p = sim_position
                final_price = float(candles.iloc[-1]["close"])
                if p["side"] == "LONG":
                    pnl = (final_price - p["entry"]) * p["qty"]
                else:
                    pnl = (p["entry"] - final_price) * p["qty"]
                fee = p["entry"] * p["qty"] * 0.0008
                sim_balance += pnl - fee
                sim_trades.append({
                    "symbol": symbol, "side": p["side"],
                    "entry": p["entry"], "exit": final_price,
                    "sl": p.get("sl", 0), "tp": p.get("tp", 0),
                    "pnl": round(pnl - fee, 4), "reason": "종료",
                    "entry_idx": p["entry_idx"], "exit_idx": len(candles) - 1,
                })

            all_results.append({
                "symbol": symbol, "trades": sim_trades,
                "equity": sim_equity, "final_balance": sim_balance,
                "candles": candles,
            })

        # ── 결과 표시 ──
        if all_results:
            st.divider()
            # 전체 요약
            total_pnl = sum(r["final_balance"] - sim_capital for r in all_results)
            total_trades = sum(len(r["trades"]) for r in all_results)
            total_wins = sum(1 for r in all_results for t in r["trades"] if t["pnl"] > 0)
            total_losses = sum(1 for r in all_results for t in r["trades"] if t["pnl"] <= 0)
            win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
            tp_count = sum(1 for r in all_results for t in r["trades"] if t["reason"] == "TP")
            sl_count = sum(1 for r in all_results for t in r["trades"] if t["reason"] == "SL")

            st.subheader("전체 요약")
            s1, s2, s3, s4, s5 = st.columns(5)
            pnl_color = "normal" if total_pnl >= 0 else "inverse"
            s1.metric("총 손익", f"${total_pnl:+,.2f}",
                      delta=f"{total_pnl/sim_capital*100:+.1f}%")
            s2.metric("승률", f"{win_rate:.1f}%")
            s3.metric("거래 수", f"{total_trades}건")
            s4.metric("TP / SL", f"{tp_count} / {sl_count}")
            s5.metric("수익/손실", f"{total_wins}W / {total_losses}L")

            # 심볼별 상세
            for res in all_results:
                symbol = res["symbol"]
                trades = res["trades"]
                equity = res["equity"]
                cdf = res["candles"]

                sym_pnl = res["final_balance"] - sim_capital
                sym_wins = sum(1 for t in trades if t["pnl"] > 0)
                sym_losses = sum(1 for t in trades if t["pnl"] <= 0)
                sym_tp = sum(1 for t in trades if t["reason"] == "TP")
                sym_sl = sum(1 for t in trades if t["reason"] == "SL")

                st.divider()
                st.subheader(f"📈 {symbol}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("손익", f"${sym_pnl:+,.2f}")
                m2.metric("거래", f"{len(trades)}건")
                m3.metric("승률", f"{sym_wins/(sym_wins+sym_losses)*100:.1f}%"
                          if (sym_wins + sym_losses) > 0 else "—")
                m4.metric("TP/SL", f"{sym_tp}/{sym_sl}")

                # 차트: 가격 + 진입/청산 마커
                tab_chart, tab_equity, tab_trades = st.tabs(["가격 차트", "자산 곡선", "거래 목록"])

                with tab_chart:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        vertical_spacing=0.03, row_heights=[0.8, 0.2])
                    fig.add_trace(go.Candlestick(
                        x=cdf.index, open=cdf["open"], high=cdf["high"],
                        low=cdf["low"], close=cdf["close"], name="Price",
                    ), row=1, col=1)

                    # 진입/청산 마커 + SL/TP 구간
                    for t in trades:
                        color = "#26a69a" if t["pnl"] > 0 else "#ef5350"
                        marker_sym = "triangle-up" if t["side"] == "LONG" else "triangle-down"
                        ei = t["entry_idx"]
                        xi = t["exit_idx"]
                        x_range = list(cdf.index[ei:xi+1])

                        # SL/TP 수평 구간 (진입~청산 사이)
                        if t.get("tp") and t.get("sl") and len(x_range) > 1:
                            fig.add_trace(go.Scatter(
                                x=x_range, y=[t["tp"]] * len(x_range),
                                mode="lines", line=dict(color="#26a69a", width=1, dash="dash"),
                                showlegend=False, hoverinfo="skip",
                            ), row=1, col=1)
                            fig.add_trace(go.Scatter(
                                x=x_range, y=[t["sl"]] * len(x_range),
                                mode="lines", line=dict(color="#ef5350", width=1, dash="dash"),
                                showlegend=False, hoverinfo="skip",
                            ), row=1, col=1)

                        # 진입 마커
                        fig.add_trace(go.Scatter(
                            x=[cdf.index[ei]],
                            y=[t["entry"]],
                            mode="markers",
                            marker=dict(symbol=marker_sym, size=12, color="white",
                                        line=dict(width=2, color=color)),
                            name=f'{t["side"]} 진입',
                            showlegend=False,
                        ), row=1, col=1)
                        # 청산 마커
                        fig.add_trace(go.Scatter(
                            x=[cdf.index[xi]],
                            y=[t["exit"]],
                            mode="markers",
                            marker=dict(symbol="x", size=10, color=color),
                            name=f'{t["reason"]} ${t["pnl"]:+.2f}',
                            showlegend=False,
                        ), row=1, col=1)

                    vol_colors = ["#ef5350" if c < o else "#26a69a"
                                  for c, o in zip(cdf["close"], cdf["open"])]
                    fig.add_trace(go.Bar(x=cdf.index, y=cdf["volume"],
                        marker_color=vol_colors, opacity=0.6, name="Volume"), row=2, col=1)

                    fig.update_layout(template="plotly_dark", height=450,
                                      xaxis_rangeslider_visible=False,
                                      margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                with tab_equity:
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        y=equity, mode="lines",
                        line=dict(color="#00d4aa", width=2), fill="tozeroy"))
                    fig_eq.add_hline(y=sim_capital, line_dash="dot", line_color="gray",
                                     annotation_text=f"시작 ${sim_capital:,.0f}")
                    fig_eq.update_layout(template="plotly_dark", height=300,
                                         yaxis_title="Balance (USDT)")
                    st.plotly_chart(fig_eq, use_container_width=True)

                with tab_trades:
                    if trades:
                        tdf = pd.DataFrame(trades)
                        # 가격 포맷팅
                        price_cols = ["entry", "exit", "sl", "tp"]
                        for pc in price_cols:
                            if pc in tdf.columns:
                                tdf[pc] = tdf[pc].apply(lambda x: f"${x:,.2f}" if x else "—")
                        tdf["pnl"] = tdf["pnl"].apply(lambda x: f"${x:+,.4f}")
                        display_cols = ["side", "entry", "sl", "tp", "exit", "pnl", "reason"]
                        display_cols = [c for c in display_cols if c in tdf.columns]
                        tdf = tdf.rename(columns={
                            "side": "방향", "entry": "진입가",
                            "sl": "손절가", "tp": "익절가",
                            "exit": "청산가", "pnl": "손익", "reason": "사유",
                        })
                        st.dataframe(tdf[["방향", "진입가", "손절가", "익절가", "청산가", "손익", "사유"]],
                                     use_container_width=True)
                    else:
                        st.info("진입 조건을 충족하지 못해 거래가 없습니다.")


# ═══════════════════════════════════════════════════════════
#  백테스팅
# ═══════════════════════════════════════════════════════════
elif page == "백테스팅":
    st.title("백테스팅")
    col1, col2, col3 = st.columns(3)
    symbol = col1.selectbox("심볼", SYMBOLS)
    interval = col2.selectbox("간격", ["1m", "5m", "15m", "1h"], index=0)
    capital = col3.number_input("초기 자본 (USDT)", value=10000.0, step=1000.0)

    if st.button("백테스트 실행", type="primary"):
        with st.spinner("데이터 로딩..."):
            async def fetch():
                client = FuturesClient()
                await client.connect()
                try:
                    return await client.get_candles(symbol, interval, limit=1000)
                finally:
                    await client.disconnect()
            candles = run_async(fetch())

        if candles.empty:
            st.error("데이터를 가져올 수 없습니다.")
        else:
            st.success(f"{len(candles)}개 캔들")
            strategy = TechnicalStrategy()
            backtester = Backtester(strategy=strategy, initial_capital=capital)
            result = backtester.run(symbol, candles)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("수익률", f"{result.total_return_pct:+.2f}%")
            c2.metric("승률", f"{result.win_rate:.1f}%")
            c3.metric("최대낙폭", f"{result.max_drawdown_pct:.2f}%")
            c4.metric("샤프", f"{result.sharpe_ratio:.2f}")

            tab1, tab2 = st.tabs(["자산 곡선", "거래 목록"])
            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=result.equity_curve, mode="lines",
                    line=dict(color="#00d4aa", width=2), fill="tozeroy"))
                fig.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)
            with tab2:
                if result.trades:
                    st.dataframe(pd.DataFrame([{
                        "진입": t.entry_time, "청산": t.exit_time,
                        "진입가": f"${t.entry_price:,.2f}", "청산가": f"${t.exit_price:,.2f}",
                        "손익": f"${t.pnl:+,.2f}", "%": f"{t.pnl_pct:+.2f}%",
                    } for t in result.trades]), use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  거래 내역
# ═══════════════════════════════════════════════════════════
elif page == "거래 내역":
    st.title("거래 내역")
    filter_symbol = st.selectbox("심볼", ["전체"] + SYMBOLS)
    sym = None if filter_symbol == "전체" else filter_symbol
    trades_df = load_trades_df(symbol=sym)

    if trades_df.empty:
        st.warning("거래 기록이 없습니다.")
    else:
        # 총 수수료 / 순손익 요약
        total_fee = trades_df["수수료"].sum()
        total_net = trades_df["순손익"].sum()
        total_gross = trades_df["손익(USDT)"].sum()
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("총 손익 (수수료 전)", f"${total_gross:+,.2f}")
        tc2.metric("총 수수료", f"${total_fee:,.4f}")
        tc3.metric("순손익 (수수료 후)", f"${total_net:+,.2f}")
        tc4.metric("거래 수", f"{len(trades_df)}회")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(trades_df))),
            y=trades_df["순손익"].values,
            marker_color=["#26a69a" if v >= 0 else "#ef5350"
                          for v in trades_df["순손익"].values],
        ))
        fig.update_layout(title="거래별 순손익 (수수료 차감)", template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

        if sym is None:
            summary = trades_df.groupby("심볼").agg(
                거래수=("ID", "count"), 총손익=("손익(USDT)", "sum"),
                총수수료=("수수료", "sum"), 순손익=("순손익", "sum"),
                평균손익=("순손익", "mean"),
            ).round(2)
            st.subheader("심볼별 요약")
            st.dataframe(summary, use_container_width=True)
        st.dataframe(trades_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  설정
# ═══════════════════════════════════════════════════════════
elif page == "설정":
    st.title("⚙️ 거래 설정")

    s = current_settings

    # ── 전략 선택 ──
    st.subheader("전략 선택")
    from src.strategies.registry import list_strategies
    strategies_list = list_strategies()
    strategy_names = [st_item["name"] for st_item in strategies_list]
    current_strategy = s.get("strategy", "momentum_flip_scalper")
    current_idx = strategy_names.index(current_strategy) if current_strategy in strategy_names else 0

    selected_strategy = st.selectbox(
        "매매 전략",
        options=strategy_names,
        format_func=lambda x: next(
            (si["label"] for si in strategies_list if si["name"] == x), x
        ),
        index=current_idx,
    )

    selected_info = next(si for si in strategies_list if si["name"] == selected_strategy)
    st.caption(selected_info["description"])

    mode_labels = {
        "always_flip": "항상 포지션 보유 (방향 전환 시 플립)",
        "signal_only": "신호 기반 진입 (대기 가능, 부분익절/트레일링)",
    }
    mode_text = mode_labels.get(selected_info["mode"], selected_info["mode"])
    mode_color = "#ffa726" if selected_info["mode"] == "always_flip" else "#26a69a"
    st.markdown(f'실행 모드: <span style="color:{mode_color}">**{mode_text}**</span>',
                unsafe_allow_html=True)

    if selected_strategy != current_strategy:
        st.warning("전략 변경 시 실행 중인 봇을 중지 → 재시작해야 적용됩니다.")

    st.divider()

    st.subheader("투자 규모")
    col1, col2 = st.columns(2)

    with col1:
        size_pct = st.slider(
            "잔고 대비 투자 비율 (%)",
            min_value=5, max_value=100, step=5,
            value=int(float(s["position_size_pct"]) * 100),
            help="잔고의 몇 %를 한 포지션에 투자할지 설정합니다.",
        )
        # 실제 예상 투자금 표시
        try:
            account = fetch_all([]). get("account", {})
            balance = account.get("balance", 0)
        except Exception:
            balance = 0
        if balance > 0:
            invest_amt = balance * (size_pct / 100)
            leverage_val = int(float(s["leverage"]))
            st.info(
                f"잔고 **${balance:,.2f}** × {size_pct}% = "
                f"증거금 **${invest_amt:,.2f}** → "
                f"포지션 규모 **${invest_amt * leverage_val:,.2f}** (x{leverage_val})"
            )

    with col2:
        leverage = st.select_slider(
            "레버리지",
            options=[1, 2, 3, 5, 7, 10, 15, 20, 25],
            value=int(float(s["leverage"])),
            help="높을수록 수익/손실이 증폭됩니다.",
        )
        risk_map = {1: "매우 안전", 2: "안전", 3: "보통", 5: "공격적", 7: "공격적",
                    10: "매우 공격적", 15: "위험", 20: "고위험", 25: "극고위험"}
        risk_label = risk_map.get(leverage, "")
        risk_color = "#26a69a" if leverage <= 3 else "#ffa726" if leverage <= 10 else "#ef5350"
        st.markdown(f'위험도: <span style="color:{risk_color}">**{risk_label}**</span>',
                    unsafe_allow_html=True)

    st.subheader("익절 / 손절")

    # v3 사용 시 자동 최적화 표시
    auto_sl = float(s.get("auto_sl_mult", "1.0"))
    auto_tp = float(s.get("auto_tp_mult", "2.0"))
    auto_trail_act = float(s.get("auto_trail_act_mult", "1.5"))
    auto_trail_dist = float(s.get("auto_trail_dist_mult", "0.5"))
    auto_score = float(s.get("auto_opt_score", "0"))
    auto_trades = s.get("auto_opt_trades", "0")
    auto_winrate = s.get("auto_opt_winrate", "0")

    if selected_strategy == "smart_momentum_scalper":
        st.info(
            "v3 전략은 **10분마다 자동 최적화**됩니다. "
            "최근 500개 1분봉으로 최적의 ATR 배수를 찾아 자동 적용합니다."
        )

        oc1, oc2, oc3, oc4 = st.columns(4)
        oc1.metric("손절 (ATR×)", f"{auto_sl:.1f}")
        oc2.metric("익절 (ATR×)", f"{auto_tp:.1f}")
        oc3.metric("트레일링 활성", f"{auto_trail_act:.1f}")
        oc4.metric("트레일링 거리", f"{auto_trail_dist:.1f}")

        oc5, oc6, oc7 = st.columns(3)
        oc5.metric("최적화 점수", f"{auto_score:.3f}")
        oc6.metric("시뮬 거래수", auto_trades)
        oc7.metric("시뮬 승률", f"{auto_winrate}%")

        st.caption("위 값은 최근 시장 데이터 기반으로 10분마다 자동 갱신됩니다.")

        # 수동 오버라이드 옵션
        tp = float(s["tp_pct"]) * 100
        sl = float(s["sl_pct"]) * 100
    else:
        # v1/v2: 기존 수동 설정
        col3, col4 = st.columns(2)
        with col3:
            tp = st.number_input(
                "익절 (%)", min_value=0.1, max_value=10.0, step=0.1,
                value=float(s["tp_pct"]) * 100,
                help="이 수익률에 도달하면 자동 청산합니다.",
            )
            st.caption(f"레버리지 적용: 실효 익절 = {tp * leverage:.1f}%")

        with col4:
            sl = st.number_input(
                "손절 (%)", min_value=0.1, max_value=10.0, step=0.1,
                value=float(s["sl_pct"]) * 100,
                help="이 손실률에 도달하면 자동 청산합니다.",
            )
            st.caption(f"레버리지 적용: 실효 손절 = {sl * leverage:.1f}%")

    st.subheader("분석 주기")
    tick = st.slider(
        "분석 주기 (초)", min_value=5, max_value=120, step=5,
        value=int(float(s["tick_interval"])),
        help="몇 초마다 시장을 분석하고 매매 판단을 내릴지 설정합니다.",
    )
    st.caption(f"1분봉 기준, {60 // tick}회/분 분석")

    # ── 웹훅 ──
    st.subheader("웹훅 알림")
    st.caption("매수/매도/청산 이벤트를 외부 URL로 POST 전송합니다. (Slack, Discord, Telegram Bot, n8n 등)")

    webhook_url = st.text_input(
        "웹훅 URL",
        value=s.get("webhook_url", ""),
        placeholder="https://hooks.slack.com/services/... 또는 비워두면 비활성화",
    )

    wh_col1, wh_col2, wh_col3 = st.columns(3)
    wh_on_open = wh_col1.checkbox("포지션 진입", value=s.get("webhook_on_open", "true") == "true")
    wh_on_close = wh_col2.checkbox("포지션 청산", value=s.get("webhook_on_close", "true") == "true")
    wh_on_tp_sl = wh_col3.checkbox("익절/손절/부분익절", value=s.get("webhook_on_tp_sl", "true") == "true")

    if webhook_url:
        st.caption("전송 형식: `POST { event, timestamp, symbol, direction, price, pnl_usdt, reason, ... }`")

    st.divider()

    # 저장 버튼
    if st.button("설정 저장", type="primary"):
        db.set_setting("strategy", selected_strategy)
        db.set_setting("position_size_pct", str(size_pct / 100))
        db.set_setting("leverage", str(leverage))
        db.set_setting("tp_pct", str(tp / 100))
        db.set_setting("sl_pct", str(sl / 100))
        db.set_setting("webhook_url", webhook_url)
        db.set_setting("webhook_on_open", "true" if wh_on_open else "false")
        db.set_setting("webhook_on_close", "true" if wh_on_close else "false")
        db.set_setting("webhook_on_tp_sl", "true" if wh_on_tp_sl else "false")
        db.set_setting("tick_interval", str(tick))
        st.success("설정이 저장되었습니다. 전략 변경 시 봇을 재시작하세요.")
        st.rerun()

    # 현재 설정 vs 변경 비교
    st.subheader("설정 변경 미리보기")
    current_label = next(
        (si["label"] for si in strategies_list if si["name"] == current_strategy),
        current_strategy,
    )
    selected_label = next(
        (si["label"] for si in strategies_list if si["name"] == selected_strategy),
        selected_strategy,
    )
    preview = pd.DataFrame({
        "항목": ["전략", "투자 비율", "레버리지", "익절", "손절", "분석 주기"],
        "현재": [
            current_label,
            f"{float(s['position_size_pct'])*100:.0f}%",
            f"x{s['leverage']}",
            f"{float(s['tp_pct'])*100:.1f}%",
            f"{float(s['sl_pct'])*100:.1f}%",
            f"{s['tick_interval']}초",
        ],
        "변경 후": [
            selected_label,
            f"{size_pct}%",
            f"x{leverage}",
            f"{tp:.1f}%",
            f"{sl:.1f}%",
            f"{tick}초",
        ],
    })
    st.dataframe(preview, use_container_width=True, hide_index=True)
