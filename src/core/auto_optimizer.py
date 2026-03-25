"""Auto Optimizer — 최근 데이터 기반 TP/SL 배수 자동 최적화.

주기적으로 최근 캔들 데이터에 대해 다양한 ATR 배수 조합을
시뮬레이션하고, 최적의 배수를 DB에 저장한다.
엔진과 전략은 이 값을 실시간으로 읽어 적용한다.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import ta
import structlog

from src.core import database as db

logger = structlog.get_logger()

# 탐색 범위 (넓은 SL + 넓은 TP = 승률 우선)
SL_RANGE = [5.0, 6.0, 8.0, 10.0]                      # ATR 배수
TP_RANGE = [8.0, 10.0, 12.0, 15.0]                    # ATR 배수
TRAIL_ACT_RANGE = [8.0, 10.0, 12.0]                   # ATR 배수
TRAIL_DIST_RANGE = [2.0, 3.0, 4.0]                    # ATR 배수

COMMISSION = 0.0004  # 왕복 수수료


@dataclass
class SimResult:
    sl_mult: float
    tp_mult: float
    trail_act_mult: float
    trail_dist_mult: float
    total_trades: int
    wins: int
    losses: int
    total_pnl: float
    max_drawdown: float
    profit_factor: float   # 총수익 / 총손실
    win_rate: float
    avg_rr: float          # 평균 리스크-리워드
    score: float           # 종합 점수


def _simulate_params(
    candles: pd.DataFrame,
    atr_series: pd.Series,
    sl_mult: float,
    tp_mult: float,
    trail_act_mult: float,
    trail_dist_mult: float,
    direction_series: pd.Series,
) -> SimResult:
    """한 세트의 파라미터로 빠른 시뮬레이션."""
    close = candles["close"].values
    atr = atr_series.values
    directions = direction_series.values

    trades_pnl = []
    position = None  # (side, entry_price, entry_atr, highest, lowest)
    peak_equity = 0.0
    equity = 0.0
    max_dd = 0.0

    for i in range(50, len(close)):
        price = close[i]
        cur_atr = atr[i]

        if cur_atr <= 0 or np.isnan(cur_atr):
            continue

        # 포지션 있으면 체크
        if position is not None:
            side, entry, entry_atr, highest, lowest = position
            highest = max(highest, price)
            lowest = min(lowest, price)
            position = (side, entry, entry_atr, highest, lowest)

            sl_dist = entry_atr * sl_mult
            tp_dist = entry_atr * tp_mult
            trail_act = entry_atr * trail_act_mult
            trail_dist = entry_atr * trail_dist_mult

            if side == "LONG":
                pnl_raw = price - entry
                # 손절
                if pnl_raw <= -sl_dist:
                    pnl = -sl_dist - (entry * COMMISSION)
                    trades_pnl.append(pnl)
                    position = None
                    continue
                # 익절
                if pnl_raw >= tp_dist:
                    pnl = tp_dist - (entry * COMMISSION)
                    trades_pnl.append(pnl)
                    position = None
                    continue
                # 트레일링
                if pnl_raw >= trail_act:
                    trail_stop = highest - trail_dist
                    if price <= trail_stop:
                        pnl = (trail_stop - entry) - (entry * COMMISSION)
                        trades_pnl.append(pnl)
                        position = None
                        continue
            else:  # SHORT
                pnl_raw = entry - price
                if pnl_raw <= -sl_dist:
                    pnl = -sl_dist - (entry * COMMISSION)
                    trades_pnl.append(pnl)
                    position = None
                    continue
                if pnl_raw >= tp_dist:
                    pnl = tp_dist - (entry * COMMISSION)
                    trades_pnl.append(pnl)
                    position = None
                    continue
                if pnl_raw >= trail_act:
                    trail_stop = lowest + trail_dist
                    if price >= trail_stop:
                        pnl = (entry - trail_stop) - (entry * COMMISSION)
                        trades_pnl.append(pnl)
                        position = None
                        continue
        else:
            # 진입 (direction_series 기반)
            d = directions[i]
            if d != 0 and cur_atr > 0:
                side = "LONG" if d > 0 else "SHORT"
                position = (side, price, cur_atr, price, price)

        # Equity tracking
        equity = sum(trades_pnl)
        peak_equity = max(peak_equity, equity)
        dd = peak_equity - equity
        max_dd = max(max_dd, dd)

    # 결과 계산
    if not trades_pnl:
        return SimResult(
            sl_mult=sl_mult, tp_mult=tp_mult,
            trail_act_mult=trail_act_mult, trail_dist_mult=trail_dist_mult,
            total_trades=0, wins=0, losses=0, total_pnl=0,
            max_drawdown=0, profit_factor=0, win_rate=0, avg_rr=0, score=-999,
        )

    wins = [p for p in trades_pnl if p > 0]
    losses = [p for p in trades_pnl if p <= 0]
    total_pnl = sum(trades_pnl)
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.0001
    profit_factor = gross_profit / gross_loss
    win_rate = len(wins) / len(trades_pnl) if trades_pnl else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0.0001
    avg_rr = avg_win / avg_loss if avg_loss > 0 else 0

    # 종합 점수: profit_factor × win_rate × (1 - drawdown비율) × 거래수 보정
    trade_count_bonus = min(len(trades_pnl) / 10, 1.0)  # 최소 10건은 있어야 신뢰
    dd_penalty = 1.0 - min(max_dd / (abs(total_pnl) + 0.01), 1.0)
    score = profit_factor * win_rate * dd_penalty * trade_count_bonus

    return SimResult(
        sl_mult=sl_mult, tp_mult=tp_mult,
        trail_act_mult=trail_act_mult, trail_dist_mult=trail_dist_mult,
        total_trades=len(trades_pnl), wins=len(wins), losses=len(losses),
        total_pnl=total_pnl, max_drawdown=max_dd,
        profit_factor=round(profit_factor, 2),
        win_rate=round(win_rate * 100, 1),
        avg_rr=round(avg_rr, 2), score=round(score, 4),
    )


def optimize(candles_1m: pd.DataFrame, htf_candles: pd.DataFrame | None = None) -> dict:
    """최근 캔들 데이터로 최적 ATR 배수를 찾는다.

    Returns:
        {"sl_mult": float, "tp_mult": float, "trail_act_mult": float,
         "trail_dist_mult": float, "score": float, "details": dict}
    """
    if len(candles_1m) < 100:
        return {}

    close = candles_1m["close"]
    high = candles_1m["high"]
    low = candles_1m["low"]

    # ATR 계산
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # 방향 신호 생성 (EMA cross 기반 — 단순하지만 일관성 있음)
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    direction = pd.Series(0, index=candles_1m.index)
    direction[ema8 > ema21] = 1   # LONG
    direction[ema8 < ema21] = -1  # SHORT
    # 크로스 시점만 진입
    direction_change = direction.diff().abs()
    entry_signal = direction.where(direction_change > 0, 0)

    # 그리드 탐색 (가장 빠른 조합만 — 5×6×3×3 = 270개)
    best = None
    all_results = []

    for sl in SL_RANGE:
        for tp in TP_RANGE:
            if tp < sl * 1.5:
                continue  # RR < 1.5인 조합은 건너뜀

            for ta_mult in TRAIL_ACT_RANGE:
                for td_mult in TRAIL_DIST_RANGE:
                    result = _simulate_params(
                        candles_1m, atr, sl, tp, ta_mult, td_mult, entry_signal,
                    )
                    all_results.append(result)
                    if best is None or result.score > best.score:
                        best = result

    if best is None or best.total_trades < 5:
        return {}

    # 상위 3개 결과
    top3 = sorted(all_results, key=lambda r: r.score, reverse=True)[:3]

    return {
        "sl_mult": best.sl_mult,
        "tp_mult": best.tp_mult,
        "trail_act_mult": best.trail_act_mult,
        "trail_dist_mult": best.trail_dist_mult,
        "score": best.score,
        "details": {
            "trades": best.total_trades,
            "win_rate": best.win_rate,
            "profit_factor": best.profit_factor,
            "avg_rr": best.avg_rr,
            "total_pnl": round(best.total_pnl, 2),
            "max_drawdown": round(best.max_drawdown, 2),
        },
        "top3": [
            {
                "sl": r.sl_mult, "tp": r.tp_mult,
                "trail_act": r.trail_act_mult, "trail_dist": r.trail_dist_mult,
                "score": r.score, "trades": r.total_trades,
                "win_rate": r.win_rate, "pf": r.profit_factor,
            }
            for r in top3
        ],
    }


def run_and_save(candles_1m: pd.DataFrame, htf_candles: pd.DataFrame | None = None) -> dict:
    """최적화 실행 후 결과를 DB에 저장."""
    result = optimize(candles_1m, htf_candles)
    if not result:
        logger.warning("optimizer.no_result", msg="Not enough data or trades")
        return {}

    db.set_setting("auto_sl_mult", str(result["sl_mult"]))
    db.set_setting("auto_tp_mult", str(result["tp_mult"]))
    db.set_setting("auto_trail_act_mult", str(result["trail_act_mult"]))
    db.set_setting("auto_trail_dist_mult", str(result["trail_dist_mult"]))
    db.set_setting("auto_opt_score", str(result["score"]))
    db.set_setting("auto_opt_trades", str(result["details"]["trades"]))
    db.set_setting("auto_opt_winrate", str(result["details"]["win_rate"]))

    logger.info(
        "optimizer.updated",
        sl=result["sl_mult"], tp=result["tp_mult"],
        trail_act=result["trail_act_mult"], trail_dist=result["trail_dist_mult"],
        score=result["score"],
        trades=result["details"]["trades"],
        win_rate=result["details"]["win_rate"],
        pf=result["details"]["profit_factor"],
    )
    return result
