"""페이퍼 매매 승률 기반 자동 전략 선택기.

Phase 18 — 사용자 요구: "가상매매에서 승률이 좋으면 자동으로 봇전환을해"

주기적으로 모든 등록된 전략의 페이퍼 성과를 평가하여
- 최소 거래 수 (min_trades) 이상
- 승률 (winrate) ≥ 임계값
- 순손익 (net_pnl) > 임계값
조건을 만족하는 후보들 중 가장 높은 net_pnl_pct를 가진 전략을 선택.

모든 전략이 실격이면 `trading_paused=true`로 설정해서 엔진이
신규 진입을 전면 차단하도록 한다 (기존 포지션은 그대로 관리).

레지스트리에 등록된 전략만 후보가 된다 (paper DB에는 과거 전략이
남아있을 수 있으므로 필터링).
"""

from __future__ import annotations

import structlog

from src.core import database as db
from src.strategies.registry import list_strategies
from src.utils.timezone import now_kst

logger = structlog.get_logger()


def _registered_strategy_names() -> set[str]:
    return {s["name"] for s in list_strategies()}


def evaluate_candidates() -> tuple[list[dict], dict]:
    """페이퍼 성과를 로드 후 실격/통과 분류.

    Returns:
        (ranked_eligible, thresholds) — ranked_eligible는 net_pnl_pct 내림차순.
    """
    min_trades = db.get_setting_int("paper_selector_min_trades") or 10
    min_winrate = db.get_setting_float("paper_selector_min_winrate") or 0.5
    min_net_pnl = db.get_setting_float("paper_selector_min_net_pnl")  # 0 기본

    thresholds = {
        "min_trades": min_trades,
        "min_winrate": min_winrate,
        "min_net_pnl": min_net_pnl,
    }

    registered = _registered_strategy_names()
    stats = db.get_paper_strategy_stats(min_trades=min_trades)

    eligible: list[dict] = []
    for s in stats:
        if s["strategy"] not in registered:
            continue  # 미등록 전략(삭제된 AI 전략 등) 제외
        if not s["eligible"]:
            continue
        if s["winrate"] < min_winrate:
            continue
        if s["net_pnl"] <= min_net_pnl:
            continue
        eligible.append(s)

    eligible.sort(key=lambda x: x["net_pnl_pct"], reverse=True)
    return eligible, thresholds


def select_and_apply() -> dict:
    """후보 평가 후 DB 설정에 반영. 엔진이 _check_hot_reload로 감지해서 스왑.

    Returns:
        {"action": "switched|kept|paused", "strategy": str, "reason": str}
    """
    if db.get_setting("paper_selector_enabled") != "true":
        return {"action": "disabled", "strategy": "", "reason": "selector_disabled"}

    eligible, thresholds = evaluate_candidates()
    current = db.get_setting("strategy") or ""

    db.set_setting("paper_selector_last_run", now_kst().isoformat())

    if not eligible:
        # 적격 전략 없음 → 모든 거래 정지
        if db.get_setting("trading_paused") != "true":
            db.set_setting("trading_paused", "true")
            db.set_setting(
                "trading_paused_reason",
                f"no_eligible_strategy (min_trades={thresholds['min_trades']}, "
                f"min_winrate={thresholds['min_winrate']}, "
                f"min_net_pnl={thresholds['min_net_pnl']})",
            )
            logger.warning(
                "selector.all_disqualified",
                thresholds=thresholds,
                action="trading_paused",
            )
        return {
            "action": "paused",
            "strategy": current,
            "reason": "no_eligible_strategy",
        }

    best = eligible[0]
    best_name = best["strategy"]

    # 거래 재개
    if db.get_setting("trading_paused") == "true":
        db.set_setting("trading_paused", "false")
        db.set_setting("trading_paused_reason", "")
        logger.info("selector.trading_resumed", strategy=best_name)

    db.set_setting("paper_selector_last_pick", best_name)

    if best_name == current:
        logger.info(
            "selector.kept",
            strategy=best_name,
            winrate=round(best["winrate"], 3),
            net_pnl=round(best["net_pnl"], 2),
            trades=best["total_trades"],
        )
        return {"action": "kept", "strategy": best_name, "reason": "already_active"}

    db.set_setting("strategy", best_name)
    logger.info(
        "selector.switched",
        old=current,
        new=best_name,
        winrate=round(best["winrate"], 3),
        net_pnl=round(best["net_pnl"], 2),
        net_pnl_pct=round(best["net_pnl_pct"] * 100, 2),
        trades=best["total_trades"],
    )
    db.log_agent_swap(
        action="selector_switched",
        strategy_name=best_name,
        reason=f"paper winrate {best['winrate']:.1%} over {best['total_trades']} trades",
        win_rate=best["winrate"],
    )
    return {"action": "switched", "strategy": best_name, "reason": "higher_winrate"}
