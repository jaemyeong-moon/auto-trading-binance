"""Database setup and trade/position persistence using SQLAlchemy."""

from datetime import datetime
from pathlib import Path

from src.utils.timezone import now_kst

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.core.config import settings

# testnet / real 환경별 DB 분리 — 모드 전환 시 데이터 유실 방지
_db_name = "trades_testnet.db" if settings.exchange.testnet else "trades_real.db"
DB_PATH = Path(__file__).parent.parent.parent / "data" / _db_name
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
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


engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()


# ─── Position helpers ──────────────────────────────────────

def open_position(
    symbol: str, side: str, entry_price: float, quantity: float, strategy: str = ""
) -> PositionRecord:
    with get_session() as session:
        pos = PositionRecord(
            symbol=symbol, side=side, entry_price=entry_price,
            quantity=quantity, strategy=strategy,
        )
        session.merge(pos)  # upsert by symbol
        session.commit()
    return pos


def close_position(symbol: str, exit_price: float, fee: float = 0) -> TradeRecord | None:
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
            strategy=pos.strategy, closed_at=now_kst(),
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

def get_trades(symbol: str | None = None, limit: int = 100) -> list[TradeRecord]:
    with get_session() as session:
        q = session.query(TradeRecord).order_by(TradeRecord.closed_at.desc())
        if symbol:
            q = q.filter_by(symbol=symbol)
        return q.limit(limit).all()


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

# 기본값 정의
_DEFAULT_SETTINGS = {
    "strategy": "momentum_flip_scalper",  # 기본 전략
    "position_size_pct": "0.1",     # 잔고 대비 투자 비율 (10%)
    "leverage": "5",                 # 레버리지 배수
    "tp_pct": "0.01",               # 익절 % (1%)
    "sl_pct": "0.005",              # 손절 % (0.5%)
    "tick_interval": "15",           # 분석 주기 (초)
    # 자동 최적화 결과 (10분마다 갱신)
    "auto_sl_mult": "1.0",
    "auto_tp_mult": "2.0",
    "auto_trail_act_mult": "1.5",
    "auto_trail_dist_mult": "0.5",
    "auto_opt_score": "0",
    "auto_opt_trades": "0",
    "auto_opt_winrate": "0",
    # 웹훅
    "webhook_url": "",               # 빈 문자열이면 비활성화
    "webhook_on_open": "true",       # 포지션 진입 시
    "webhook_on_close": "true",      # 포지션 청산 시
    "webhook_on_tp_sl": "true",      # 익절/손절 시
    # AI Agent
    "ai_agent_enabled": "true",      # AI 전략 에이전트 활성화
    "ai_agent_interval": "120",      # 실행 주기 (틱 수, 기본 ~1시간)
    "ai_agent_last_run": "",         # 마지막 실행 시각
    "ai_agent_last_strategy": "",    # 마지막 생성 전략
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
