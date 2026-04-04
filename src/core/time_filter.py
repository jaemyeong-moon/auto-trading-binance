"""DB 거래 기록 기반 시간대 필터 — 승률 낮은 시간대 자동 차단.

거래 기록(실거래 + 가상매매)에서 시간대별 승률을 분석하고,
승률이 임계값 미만인 시간대의 거래를 차단한다.
캐시 TTL 1시간으로 DB 부하 최소화.
"""

import time as _time

from src.core.database import TradeRecord, PaperTrade, get_session
from src.utils.timezone import KST, now_kst

# ── 설정 ──
MIN_TRADES_PER_HOUR = 3   # 최소 거래 수 (미달 시 판단 보류)
WIN_RATE_THRESHOLD = 0.35  # 35% 미만이면 차단
_CACHE_TTL = 3600          # 1시간

# ── 캐시 (단일 프로세스 용) ──
_cache: dict = {"hours": set(), "updated": 0.0}

# 테스트/디버그용 — True이면 시간대 필터 무시
_FORCE_TRADEABLE: bool = False

# DB 데이터 없을 때 폴백 (Phase 6 분석 결과 기반)
_FALLBACK_BLOCKED = set(range(0, 10))  # 0~9시 KST


def get_blocked_hours_kst() -> set[int]:
    """DB에서 시간대별 승률을 계산하고, 승률 낮은 시간대를 반환.

    Returns:
        차단할 KST 시간(0-23) 집합
    """
    now = _time.time()
    if now - _cache["updated"] < _CACHE_TTL and _cache["updated"] > 0:
        return _cache["hours"]

    try:
        all_trades = []
        with get_session() as session:
            for Model in [TradeRecord, PaperTrade]:
                trades = session.query(Model).filter(
                    Model.closed_at.isnot(None),
                    Model.exit_price.isnot(None),
                ).all()
                all_trades.extend(trades)
    except Exception:
        return _FALLBACK_BLOCKED

    if not all_trades:
        _cache["hours"] = _FALLBACK_BLOCKED
        _cache["updated"] = now
        return _FALLBACK_BLOCKED

    # 시간대별 집계 — DB의 opened_at은 now_kst()로 저장되므로 이미 KST
    hourly: dict[int, dict] = {h: {"wins": 0, "total": 0} for h in range(24)}
    for t in all_trades:
        if not t.opened_at:
            continue
        kst_hour = t.opened_at.hour  # 이미 KST 기준 저장
        hourly[kst_hour]["total"] += 1
        pnl = getattr(t, "net_pnl", None) or getattr(t, "pnl", 0) or 0
        if pnl > 0:
            hourly[kst_hour]["wins"] += 1

    blocked = set()
    for h, data in hourly.items():
        if data["total"] >= MIN_TRADES_PER_HOUR:
            wr = data["wins"] / data["total"]
            if wr < WIN_RATE_THRESHOLD:
                blocked.add(h)

    _cache["hours"] = blocked
    _cache["updated"] = now
    return blocked


def is_tradeable_hour() -> bool:
    """현재 KST 시간이 거래 가능 시간인지 확인."""
    if _FORCE_TRADEABLE:
        return True
    now_hour = now_kst().hour
    return now_hour not in get_blocked_hours_kst()
