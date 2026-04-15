"""Database setup and trade/position persistence using SQLAlchemy."""

import json
from datetime import datetime
from pathlib import Path

from src.utils.timezone import now_kst

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.core.config import settings

# testnet / real 환경별 DB 분리 — 모드 전환 시 데이터 유실 방지
_db_name = "trades_testnet.db" if settings.exchange.testnet else "trades_real.db"

# DATABASE_URL 환경변수에서 디렉토리 추출, 없으면 프로젝트 내 data/
# 예: sqlite:////data/trades.db → /data/ 디렉토리 사용
_default_url = "sqlite+aiosqlite:///data/trades.db"
_env_url = settings.database_url

if _env_url and _env_url != _default_url:
    # sqlite:////data/trades.db → /data 추출
    _db_dir = Path(_env_url.split("///")[-1]).parent if "///" in _env_url else Path("data")
else:
    _db_dir = Path(__file__).parent.parent.parent / "data"

_db_dir.mkdir(parents=True, exist_ok=True)
DB_PATH = _db_dir / _db_name
DB_URL = f"sqlite:///{DB_PATH}"


class Base(DeclarativeBase):
    pass


class TradeRecord(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    fee = Column(Float, nullable=True)          # 왕복 수수료 (USDT)
    net_pnl = Column(Float, nullable=True)      # 순수익 (pnl - fee)
    strategy = Column(String, nullable=True)
    reason = Column(String, nullable=True)          # 청산 사유 (take_profit, stop_loss, flip, signal 등)
    sl_price = Column(Float, nullable=True)         # 손절 가격
    tp_price = Column(Float, nullable=True)         # 익절 가격
    opened_at = Column(DateTime, default=now_kst)
    closed_at = Column(DateTime, nullable=True)


class PositionRecord(Base):
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, unique=True)
    side = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    strategy = Column(String, nullable=True)
    sl_price = Column(Float, nullable=True)         # 손절 가격
    tp_price = Column(Float, nullable=True)         # 익절 가격
    opened_at = Column(DateTime, default=now_kst)


# ─── Paper Trading (가상매매) ─────────────────────────────

class PaperBalance(Base):
    """전략별 가상 잔고."""
    __tablename__ = "paper_balances"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String, nullable=False, unique=True)
    balance = Column(Float, default=200.0)
    initial_balance = Column(Float, default=200.0)
    total_trades = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)


class PaperPosition(Base):
    """전략별 가상 포지션 (심볼당 1개)."""
    __tablename__ = "paper_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    entry_atr = Column(Float, nullable=True)
    sl_price = Column(Float, nullable=True)
    tp_price = Column(Float, nullable=True)
    opened_at = Column(DateTime, default=now_kst)


class PaperTrade(Base):
    """전략별 가상 거래 기록."""
    __tablename__ = "paper_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)
    fee = Column(Float, nullable=True)
    net_pnl = Column(Float, nullable=True)
    sl_price = Column(Float, nullable=True)
    tp_price = Column(Float, nullable=True)
    reason = Column(String, nullable=True)
    opened_at = Column(DateTime, default=now_kst)
    closed_at = Column(DateTime, nullable=True)


class PositionTrail(Base):
    """포지션 생존 중 매 틱 가격 위치 궤적."""
    __tablename__ = "position_trails"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_type = Column(String, nullable=False)       # "real" or "paper"
    trade_id = Column(Integer, nullable=True)          # 청산 후 TradeRecord/PaperTrade id 연결
    symbol = Column(String, nullable=False)
    strategy = Column(String, nullable=True)           # paper용
    entry_price = Column(Float, nullable=False)
    sl_price = Column(Float, nullable=False)
    tp_price = Column(Float, nullable=False)
    price = Column(Float, nullable=False)              # 해당 틱의 현재가
    progress_pct = Column(Float, nullable=False)       # 0=SL, 100=TP
    recorded_at = Column(DateTime, default=now_kst)


class SignalLog(Base):
    """매 틱 전략 평가 결과 기록."""
    __tablename__ = "signal_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    strategy = Column(String, nullable=False)
    signal_type = Column(String, nullable=False)       # BUY, SELL, HOLD, CLOSE
    confidence = Column(Float, nullable=False)
    metadata_json = Column(String, nullable=True)      # JSON string
    source = Column(String, default="real")             # "real" or "paper"
    recorded_at = Column(DateTime, default=now_kst)


class BotState(Base):
    """Tracks which symbols have an active bot."""
    __tablename__ = "bot_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False, unique=True)
    running = Column(Boolean, default=False)
    started_at = Column(DateTime, nullable=True)
    stopped_at = Column(DateTime, nullable=True)


class BotSettings(Base):
    """봇 거래 설정 (대시보드에서 변경 가능)."""
    __tablename__ = "bot_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String, nullable=False, unique=True)
    value = Column(String, nullable=False)


class AgentSwapLog(Base):
    """AI 에이전트 전략 생성/검증/배포/거부 감사 로그."""
    __tablename__ = "agent_swap_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=now_kst)
    action = Column(String, nullable=False)      # generated, validated, deployed, rejected
    strategy_name = Column(String, nullable=False)
    reason = Column(String, nullable=True)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)


class StrategyState(Base):
    """전략 내부 상태 영속화 — 재시작 시 복원에 사용."""
    __tablename__ = "strategy_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String, nullable=False, unique=True)
    state_json = Column(String, nullable=False)   # JSON 직렬화
    updated_at = Column(DateTime, default=now_kst)


def _set_sqlite_pragmas(dbapi_conn, _connection_record):
    """SQLite WAL 모드 + busy_timeout 설정 — 다중 프로세스 접근 시 잠금 방지."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


from sqlalchemy import event as _sa_event  # noqa: E402

engine = create_engine(DB_URL, echo=False)
_sa_event.listen(engine, "connect", _set_sqlite_pragmas)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    Base.metadata.create_all(engine)
    _migrate_add_columns()


def _migrate_add_columns() -> None:
    """기존 DB에 새 컬럼이 없으면 ALTER TABLE로 추가 (SQLite 호환)."""
    migrations = [
        ("trades", "sl_price", "REAL"),
        ("trades", "tp_price", "REAL"),
        ("trades", "reason", "TEXT"),
        ("positions", "sl_price", "REAL"),
        ("positions", "tp_price", "REAL"),
    ]
    with engine.connect() as conn:
        for table, col, col_type in migrations:
            try:
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"
                    )
                )
            except Exception:
                pass  # 이미 존재하면 무시
        conn.commit()


def get_session() -> Session:
    return SessionLocal()


# ─── Position helpers ──────────────────────────────────────

def open_position(
    symbol: str, side: str, entry_price: float, quantity: float,
    strategy: str = "", sl_price: float = 0, tp_price: float = 0,
) -> PositionRecord:
    with get_session() as session:
        pos = PositionRecord(
            symbol=symbol, side=side, entry_price=entry_price,
            quantity=quantity, strategy=strategy,
            sl_price=sl_price or None, tp_price=tp_price or None,
        )
        session.merge(pos)  # upsert by symbol
        session.commit()
    return pos


def close_position(
    symbol: str, exit_price: float, fee: float = 0, reason: str = "",
) -> TradeRecord | None:
    with get_session() as session:
        pos = session.query(PositionRecord).filter_by(symbol=symbol).first()
        if not pos:
            return None

        pnl = (exit_price - pos.entry_price) * pos.quantity
        if pos.side == "SHORT":
            pnl = -pnl
        pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100 if pos.entry_price > 0 else 0
        net_pnl = pnl - fee

        trade = TradeRecord(
            symbol=symbol, side=pos.side, entry_price=pos.entry_price,
            exit_price=exit_price, quantity=pos.quantity,
            pnl=pnl, pnl_pct=pnl_pct, fee=round(fee, 4), net_pnl=round(net_pnl, 4),
            strategy=pos.strategy, reason=reason or None,
            sl_price=pos.sl_price, tp_price=pos.tp_price,
            closed_at=now_kst(),
        )
        session.add(trade)
        session.delete(pos)
        session.commit()
        session.refresh(trade)
        return trade


def update_position_quantity(symbol: str, new_quantity: float) -> None:
    """부분 익절 후 DB 포지션 수량 갱신."""
    with get_session() as session:
        pos = session.query(PositionRecord).filter_by(symbol=symbol).first()
        if pos:
            pos.quantity = new_quantity
            session.commit()


def delete_position(symbol: str) -> None:
    """거래소에서 이미 청산된 포지션의 DB 레코드 정리."""
    with get_session() as session:
        pos = session.query(PositionRecord).filter_by(symbol=symbol).first()
        if pos:
            session.delete(pos)
            session.commit()


def get_open_positions() -> list[PositionRecord]:
    with get_session() as session:
        return session.query(PositionRecord).all()


def get_position(symbol: str) -> PositionRecord | None:
    with get_session() as session:
        return session.query(PositionRecord).filter_by(symbol=symbol).first()


# ─── Trade helpers ─────────────────────────────────────────

def get_today_pnl() -> tuple[float, int]:
    """오늘(KST 기준) 청산된 거래의 PnL 합계와 거래 건수를 반환.

    Returns:
        (total_pnl, trade_count) — 거래가 없으면 (0.0, 0).
    """
    today_start = now_kst().replace(hour=0, minute=0, second=0, microsecond=0)
    with get_session() as session:
        rows = (
            session.query(TradeRecord)
            .filter(TradeRecord.closed_at >= today_start)
            .all()
        )
    total = sum((r.pnl or 0.0) for r in rows)
    return total, len(rows)


def get_trades(symbol: str | None = None, limit: int = 100) -> list[TradeRecord]:
    with get_session() as session:
        q = session.query(TradeRecord).order_by(TradeRecord.closed_at.desc())
        if symbol:
            q = q.filter_by(symbol=symbol)
        return q.limit(limit).all()


# ─── Agent swap log helper ─────────────────────────────────


def log_agent_swap(
    action: str,
    strategy_name: str,
    reason: str | None = None,
    win_rate: float | None = None,
    profit_factor: float | None = None,
) -> None:
    """AI 에이전트 전략 이벤트를 감사 로그에 기록."""
    with get_session() as session:
        session.add(AgentSwapLog(
            action=action,
            strategy_name=strategy_name,
            reason=reason,
            win_rate=win_rate,
            profit_factor=profit_factor,
        ))
        session.commit()


# ─── Strategy State helpers ───────────────────────────────

def save_strategy_state(strategy_name: str, state_dict: dict) -> None:
    """전략 상태를 DB에 저장 (upsert). 직렬화 불가 값은 건너뜀."""
    state_json = json.dumps(state_dict, default=str)
    with get_session() as session:
        row = session.query(StrategyState).filter_by(strategy_name=strategy_name).first()
        if row:
            row.state_json = state_json
            row.updated_at = now_kst()
        else:
            row = StrategyState(
                strategy_name=strategy_name,
                state_json=state_json,
                updated_at=now_kst(),
            )
            session.add(row)
        session.commit()


def load_strategy_state(strategy_name: str) -> dict | None:
    """DB에서 전략 상태를 로드. 없으면 None."""
    with get_session() as session:
        row = session.query(StrategyState).filter_by(strategy_name=strategy_name).first()
        if not row:
            return None
        try:
            return json.loads(row.state_json)
        except (json.JSONDecodeError, TypeError):
            return None


# ─── Risk status helper ────────────────────────────────────

def get_risk_status(
    balance: float = 0.0,
    max_open_positions: int = 3,
    max_daily_loss_pct: float = 0.05,
    kelly_lookback: int = 50,
) -> dict:
    """당일 리스크 상태를 집계하여 반환.

    외부 서비스 없이 DB 데이터만으로 계산하므로 대시보드/테스트에서
    바로 호출할 수 있다.

    Args:
        balance: 현재 계좌 잔고 (USDT). 0이면 daily_pnl_pct는 0 처리.
        max_open_positions: 최대 동시 포지션 수 (설정값).
        max_daily_loss_pct: 일일 최대 DD 비율 (설정값).
        kelly_lookback: Kelly 계산에 사용할 최근 거래 수.

    Returns:
        dict with keys:
            daily_pnl (float)       — 당일 PnL (USDT)
            daily_pnl_pct (float)   — 당일 PnL (잔고 대비 비율, 0~1 범위)
            daily_dd_ok (bool)      — DD 한도 내 여부
            open_positions (int)    — 현재 열린 포지션 수
            max_positions (int)     — 최대 허용 포지션 수
            can_open (bool)         — 진입 가능 여부
            kelly_sizes (dict)      — 전략별 Kelly 추천 사이즈 {strategy: float}
    """
    from src.core.risk_manager import RiskManager

    # ── 당일 PnL ──
    daily_pnl, _ = get_today_pnl()
    daily_pnl_pct = (daily_pnl / balance) if balance > 0 else 0.0

    # ── 현재 포지션 수 ──
    open_pos_list = get_open_positions()
    open_positions = len(open_pos_list)

    # ── RiskManager 판단 ──
    rm = RiskManager(
        max_open_positions=max_open_positions,
        max_daily_loss_pct=max_daily_loss_pct,
    )
    can_open, _ = rm.can_open(open_positions, daily_pnl_pct)
    dd_ok = rm.daily_dd_ok(daily_pnl_pct)

    # ── 전략별 Kelly 계산 ──
    trades = get_trades(limit=kelly_lookback)
    kelly_sizes: dict[str, float] = {}
    if trades:
        # 전략별로 그룹핑
        strategy_trades: dict[str, list[TradeRecord]] = {}
        for t in trades:
            strat = t.strategy or "unknown"
            strategy_trades.setdefault(strat, []).append(t)

        for strat, strat_trades in strategy_trades.items():
            wins = [t for t in strat_trades if (t.pnl_pct or 0.0) > 0]
            losses = [t for t in strat_trades if (t.pnl_pct or 0.0) <= 0]
            if not strat_trades:
                continue
            win_rate = len(wins) / len(strat_trades)
            avg_win = (
                sum(abs(t.pnl_pct or 0.0) for t in wins) / len(wins) / 100.0
                if wins else 0.0
            )
            avg_loss = (
                sum(abs(t.pnl_pct or 0.0) for t in losses) / len(losses) / 100.0
                if losses else 0.0
            )
            # Skip Kelly when there are no wins (avg_win=0 causes division by zero)
            if avg_win > 0:
                kelly_sizes[strat] = rm.kelly_size(win_rate, avg_win, avg_loss)
            else:
                kelly_sizes[strat] = 0.0

    return {
        "daily_pnl": daily_pnl,
        "daily_pnl_pct": daily_pnl_pct,
        "daily_dd_ok": dd_ok,
        "open_positions": open_positions,
        "max_positions": max_open_positions,
        "can_open": can_open,
        "kelly_sizes": kelly_sizes,
    }


# ─── Bot state helpers ─────────────────────────────────────

def set_bot_running(symbol: str, running: bool) -> None:
    with get_session() as session:
        state = session.query(BotState).filter_by(symbol=symbol).first()
        if not state:
            state = BotState(symbol=symbol)
            session.add(state)
        state.running = running
        if running:
            state.started_at = now_kst()
            state.stopped_at = None
        else:
            state.stopped_at = now_kst()
        session.commit()


def is_bot_running(symbol: str) -> bool:
    with get_session() as session:
        state = session.query(BotState).filter_by(symbol=symbol).first()
        return state.running if state else False


def get_all_bot_states() -> dict[str, bool]:
    with get_session() as session:
        states = session.query(BotState).all()
        return {s.symbol: s.running for s in states}


# ─── Settings helpers ─────────────────────────────────────

# 기본값 정의 — SL/TP/레버리지/투자비율은 각 전략 클래스에서 정의
_DEFAULT_SETTINGS = {
    "strategy": "pattern_scalper",   # 활성 전략
    "tick_interval": "15",           # 분석 주기 (초)
    # 웹훅
    "webhook_url": "",
    "webhook_on_open": "true",
    "webhook_on_close": "true",
    "webhook_on_tp_sl": "true",
    # AI Agent
    "ai_agent_enabled": "true",
    "ai_agent_interval": "120",
    "ai_agent_last_run": "",
    "ai_agent_last_strategy": "",
    # ── 페이퍼 승률 기반 자동 스위칭 (Phase 18) ──
    "paper_selector_enabled": "true",     # 자동 선택 활성화
    "paper_selector_min_trades": "10",    # 최소 거래 수 (샘플 신뢰도)
    "paper_selector_min_winrate": "0.50", # 최소 승률 임계값
    "paper_selector_min_net_pnl": "0",    # 최소 순손익 (USDT) — 음수면 전략 실격
    "trading_paused": "false",            # 모든 전략 실격 시 true로 설정
    "trading_paused_reason": "",          # 정지 사유 기록
    "paper_selector_last_run": "",        # 최근 실행 시각
    "paper_selector_last_pick": "",       # 최근 선정 전략명
}


def get_setting(key: str) -> str:
    with get_session() as session:
        row = session.query(BotSettings).filter_by(key=key).first()
        if row:
            return row.value
        return _DEFAULT_SETTINGS.get(key, "")


def get_setting_float(key: str) -> float:
    try:
        return float(get_setting(key))
    except (ValueError, TypeError):
        default = _DEFAULT_SETTINGS.get(key, "0")
        return float(default)


def get_setting_int(key: str) -> int:
    try:
        return int(float(get_setting(key)))
    except (ValueError, TypeError):
        default = _DEFAULT_SETTINGS.get(key, "0")
        return int(float(default))


def set_setting(key: str, value: str) -> None:
    with get_session() as session:
        row = session.query(BotSettings).filter_by(key=key).first()
        if not row:
            row = BotSettings(key=key, value=value)
            session.add(row)
        else:
            row.value = value
        session.commit()


def get_all_settings() -> dict[str, str]:
    result = dict(_DEFAULT_SETTINGS)
    with get_session() as session:
        rows = session.query(BotSettings).all()
        for r in rows:
            result[r.key] = r.value
    return result


# ─── Position Trail helpers ───────────────────────────────

def _calc_progress_pct(side: str, price: float, sl: float, tp: float) -> float:
    """SL=0%, TP=100% 기준 현재가 위치. LONG/SHORT 방향 무관하게 동일."""
    if side == "LONG":
        rng = tp - sl
        if rng <= 0:
            return 50.0
        return max(0.0, min(100.0, (price - sl) / rng * 100))
    else:
        rng = sl - tp
        if rng <= 0:
            return 50.0
        return max(0.0, min(100.0, (sl - price) / rng * 100))


def record_trail(
    trade_type: str, symbol: str, side: str,
    entry_price: float, sl_price: float, tp_price: float,
    price: float, strategy: str = "",
) -> None:
    """매 틱 포지션 궤적 기록."""
    pct = _calc_progress_pct(side, price, sl_price, tp_price)
    with get_session() as session:
        session.add(PositionTrail(
            trade_type=trade_type, symbol=symbol, strategy=strategy or None,
            entry_price=entry_price, sl_price=sl_price, tp_price=tp_price,
            price=price, progress_pct=round(pct, 2),
        ))
        session.commit()


def link_trails_to_trade(
    trade_type: str, trade_id: int, symbol: str,
    entry_price: float, strategy: str = "",
    session: "Session | None" = None,
) -> None:
    """포지션 청산 시, 해당 포지션의 trail 레코드에 trade_id 연결.

    session 인자를 전달하면 그 세션 안에서 실행하고 commit은 호출자가 담당.
    전달하지 않으면 자체 세션을 열어 commit까지 처리.
    """
    def _run(s):
        q = s.query(PositionTrail).filter_by(
            trade_type=trade_type, symbol=symbol,
            entry_price=entry_price, trade_id=None,
        )
        if strategy:
            q = q.filter_by(strategy=strategy)
        q.update({"trade_id": trade_id})

    if session is not None:
        _run(session)
    else:
        with get_session() as s:
            _run(s)
            s.commit()


def get_trail(trade_type: str, trade_id: int) -> list[dict]:
    """거래 ID로 궤적 조회."""
    with get_session() as session:
        rows = session.query(PositionTrail).filter_by(
            trade_type=trade_type, trade_id=trade_id,
        ).order_by(PositionTrail.recorded_at.asc()).all()
        return [{
            "time": r.recorded_at.isoformat() if r.recorded_at else "",
            "price": r.price,
            "pct": r.progress_pct,
        } for r in rows]


# ─── Paper strategy stats (자동 스위칭용) ────────────────

def get_paper_strategy_stats(min_trades: int = 10) -> list[dict]:
    """모든 페이퍼 전략의 성과 지표 반환.

    - balance, initial_balance, total_trades, wins, losses
    - winrate = wins / total_trades (0~1)
    - net_pnl = balance - initial_balance
    - net_pnl_pct = net_pnl / initial_balance
    - eligible = total_trades >= min_trades
    """
    with get_session() as session:
        rows = session.query(PaperBalance).all()
        result = []
        for r in rows:
            total = r.total_trades or 0
            wins = r.wins or 0
            losses = r.losses or 0
            winrate = (wins / total) if total > 0 else 0.0
            net_pnl = (r.balance or 0.0) - (r.initial_balance or 0.0)
            net_pnl_pct = (net_pnl / r.initial_balance) if r.initial_balance else 0.0
            result.append({
                "strategy": r.strategy,
                "balance": r.balance,
                "initial_balance": r.initial_balance,
                "total_trades": total,
                "wins": wins,
                "losses": losses,
                "winrate": winrate,
                "net_pnl": net_pnl,
                "net_pnl_pct": net_pnl_pct,
                "eligible": total >= min_trades,
            })
        return result


# ─── Signal Log helpers ──────────────────────────────────

_SIGNAL_LOG_MAX = 500


def log_signal(
    symbol: str, strategy: str, signal_type: str,
    confidence: float, metadata: dict | None = None,
    source: str = "real",
) -> None:
    """전략 평가 결과를 DB에 기록. 500건 초과 시 오래된 것부터 삭제."""
    meta_str = json.dumps(metadata, default=str) if metadata else None
    with get_session() as session:
        session.add(SignalLog(
            symbol=symbol, strategy=strategy, signal_type=signal_type,
            confidence=round(confidence, 4), metadata_json=meta_str,
            source=source,
        ))
        # 링버퍼: 500건 초과 시 오래된 것 삭제
        count = session.query(SignalLog).count()
        if count > _SIGNAL_LOG_MAX:
            excess = count - _SIGNAL_LOG_MAX
            old_ids = [r.id for r in session.query(SignalLog.id)
                       .order_by(SignalLog.recorded_at.asc())
                       .limit(excess).all()]
            if old_ids:
                session.query(SignalLog).filter(
                    SignalLog.id.in_(old_ids)).delete(synchronize_session=False)
        session.commit()


def get_recent_signals(
    symbol: str | None = None, limit: int = 20, source: str | None = None,
) -> list[dict]:
    """최근 전략 판단 로그 조회."""
    with get_session() as session:
        q = session.query(SignalLog).order_by(SignalLog.recorded_at.desc())
        if symbol:
            q = q.filter_by(symbol=symbol)
        if source:
            q = q.filter_by(source=source)
        rows = q.limit(limit).all()
        return [{
            "id": r.id,
            "symbol": r.symbol,
            "strategy": r.strategy,
            "signal_type": r.signal_type,
            "confidence": r.confidence,
            "metadata": json.loads(r.metadata_json) if r.metadata_json else {},
            "source": r.source,
            "recorded_at": r.recorded_at.isoformat() if r.recorded_at else "",
        } for r in rows]


# ─── Settings hash (핫리로드 감지용) ─────────────────────

def get_settings_hash() -> str:
    """현재 설정의 해시값 반환 — 변경 감지용."""
    import hashlib
    settings = get_all_settings()
    raw = json.dumps(settings, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()
