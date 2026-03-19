"""시간대 유틸 — 한국 시간(KST) 기준."""

from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))


def now_kst() -> datetime:
    """현재 한국 시간."""
    return datetime.now(KST)
