"""차트 패턴 감지 라이브러리.

지원 패턴:
  반전 패턴 (Reversal):
    - 쌍바닥 (Double Bottom) → LONG
    - 쌍봉 (Double Top) → SHORT
    - 역머리어깨 (Inverse Head & Shoulders) → LONG
    - 머리어깨 (Head & Shoulders) → SHORT

  지속 패턴 (Continuation):
    - 상승 깃발 (Bull Flag) → LONG
    - 하락 깃발 (Bear Flag) → SHORT
    - 삼각수렴 돌파 (Triangle Breakout) → 방향

  거래량 분석:
    - 거래량 다이버전스 (가격↑ + 거래량↓ = 약세)
    - 거래량 급등 (Volume Spike) → 방향 전환 신호
    - OBV 추세 (On Balance Volume)
"""

from dataclasses import dataclass
import numpy as np


# ─── 공통 상수 ────────────────────────────────────────────

LOOKBACK = 120         # 패턴 탐색 캔들 수
EXTREMA_WINDOW = 3     # 극점 탐색 윈도우
MIN_DISTANCE = 5       # 극점 간 최소 간격
PROXIMITY_PCT = 0.005  # 가격 근접도 0.5%


# ─── 공통 유틸 ────────────────────────────────────────────

@dataclass
class PatternResult:
    """감지된 패턴."""
    name: str               # 패턴 이름
    direction: str          # "LONG" | "SHORT"
    strength: float         # 0~1 패턴 강도
    entry_price: float      # 현재가
    sl_price: float         # 손절가
    tp_price: float         # 목표가
    neckline: float = 0.0
    pattern_height: float = 0.0
    detail: dict | None = None


def _local_minima(values: np.ndarray, w: int = EXTREMA_WINDOW) -> list[int]:
    minima = []
    for i in range(w, len(values) - w):
        if values[i] == values[max(0, i - w):i + w + 1].min():
            minima.append(i)
    return minima


def _local_maxima(values: np.ndarray, w: int = EXTREMA_WINDOW) -> list[int]:
    maxima = []
    for i in range(w, len(values) - w):
        if values[i] == values[max(0, i - w):i + w + 1].max():
            maxima.append(i)
    return maxima


# ─── 반전 패턴 ────────────────────────────────────────────

def detect_double_bottom(
    low: np.ndarray, high: np.ndarray, close: np.ndarray, atr: float,
) -> PatternResult | None:
    """쌍바닥 → LONG. 두 저점이 근접하고 넥라인 돌파."""
    n = min(LOOKBACK, len(low))
    lo, hi, cl = low[-n:], high[-n:], close[-n:]
    mins = _local_minima(lo)
    maxs = _local_maxima(hi)
    if len(mins) < 2:
        return None

    for i in range(len(mins) - 1, 0, -1):
        i2, i1 = mins[i], mins[i - 1]
        if i2 - i1 < MIN_DISTANCE:
            continue
        l1, l2 = lo[i1], lo[i2]
        avg = (l1 + l2) / 2
        if abs(l1 - l2) / avg > PROXIMITY_PCT:
            continue
        peaks = [m for m in maxs if i1 < m < i2]
        if not peaks:
            continue
        neck_idx = max(peaks, key=lambda m: hi[m])
        neck = hi[neck_idx]
        height = neck - avg
        if height / avg < 0.001:
            continue
        price = cl[-1]
        dist = (price - neck) / neck
        if dist < -0.02 or dist > 0.01:
            continue

        strength = min(1.0, (1 - abs(l1 - l2) / avg / PROXIMITY_PCT +
                             min(1.0, height / avg / 0.01)) / 2)
        sl = min(l1, l2) - atr * 0.3
        tp = neck + height
        return PatternResult(
            name="double_bottom", direction="LONG", strength=strength,
            entry_price=price, sl_price=sl, tp_price=tp,
            neckline=neck, pattern_height=height,
            detail={"first": round(l1, 2), "second": round(l2, 2)},
        )
    return None


def detect_double_top(
    low: np.ndarray, high: np.ndarray, close: np.ndarray, atr: float,
) -> PatternResult | None:
    """쌍봉 → SHORT."""
    n = min(LOOKBACK, len(high))
    lo, hi, cl = low[-n:], high[-n:], close[-n:]
    maxs = _local_maxima(hi)
    mins = _local_minima(lo)
    if len(maxs) < 2:
        return None

    for i in range(len(maxs) - 1, 0, -1):
        i2, i1 = maxs[i], maxs[i - 1]
        if i2 - i1 < MIN_DISTANCE:
            continue
        h1, h2 = hi[i1], hi[i2]
        avg = (h1 + h2) / 2
        if abs(h1 - h2) / avg > PROXIMITY_PCT:
            continue
        troughs = [m for m in mins if i1 < m < i2]
        if not troughs:
            continue
        neck_idx = min(troughs, key=lambda m: lo[m])
        neck = lo[neck_idx]
        height = avg - neck
        if height / avg < 0.001:
            continue
        price = cl[-1]
        dist = (neck - price) / neck
        if dist < -0.02 or dist > 0.01:
            continue

        strength = min(1.0, (1 - abs(h1 - h2) / avg / PROXIMITY_PCT +
                             min(1.0, height / avg / 0.01)) / 2)
        sl = max(h1, h2) + atr * 0.3
        tp = neck - height
        return PatternResult(
            name="double_top", direction="SHORT", strength=strength,
            entry_price=price, sl_price=sl, tp_price=tp,
            neckline=neck, pattern_height=height,
            detail={"first": round(h1, 2), "second": round(h2, 2)},
        )
    return None


def detect_inv_head_shoulders(
    low: np.ndarray, high: np.ndarray, close: np.ndarray, atr: float,
) -> PatternResult | None:
    """역머리어깨 → LONG. 세 저점 중 가운데가 가장 낮고, 좌우 어깨가 근접."""
    n = min(LOOKBACK, len(low))
    lo, hi, cl = low[-n:], high[-n:], close[-n:]
    mins = _local_minima(lo)
    maxs = _local_maxima(hi)
    if len(mins) < 3:
        return None

    for i in range(len(mins) - 1, 1, -1):
        r_idx, h_idx, l_idx = mins[i], mins[i - 1], mins[i - 2]
        if r_idx - l_idx < MIN_DISTANCE * 2:
            continue

        head = lo[h_idx]
        l_shoulder = lo[l_idx]
        r_shoulder = lo[r_idx]

        # 머리가 가장 낮아야 함
        if head >= l_shoulder or head >= r_shoulder:
            continue
        # 양 어깨 근접
        avg_shoulder = (l_shoulder + r_shoulder) / 2
        if abs(l_shoulder - r_shoulder) / avg_shoulder > PROXIMITY_PCT * 1.5:
            continue

        # 넥라인: 어깨 사이 고점 2개의 평균
        peaks_l = [m for m in maxs if l_idx < m < h_idx]
        peaks_r = [m for m in maxs if h_idx < m < r_idx]
        if not peaks_l or not peaks_r:
            continue
        neck_l = hi[max(peaks_l, key=lambda m: hi[m])]
        neck_r = hi[max(peaks_r, key=lambda m: hi[m])]
        neck = (neck_l + neck_r) / 2

        height = neck - head
        if height / neck < 0.002:
            continue

        price = cl[-1]
        dist = (price - neck) / neck
        if dist < -0.02 or dist > 0.015:
            continue

        strength = min(1.0, height / neck / 0.01) * 0.8
        sl = head - atr * 0.3
        tp = neck + height
        return PatternResult(
            name="inv_head_shoulders", direction="LONG", strength=strength,
            entry_price=price, sl_price=sl, tp_price=tp,
            neckline=neck, pattern_height=height,
            detail={"head": round(head, 2),
                    "l_shoulder": round(l_shoulder, 2),
                    "r_shoulder": round(r_shoulder, 2)},
        )
    return None


def detect_head_shoulders(
    low: np.ndarray, high: np.ndarray, close: np.ndarray, atr: float,
) -> PatternResult | None:
    """머리어깨 → SHORT. 세 고점 중 가운데가 가장 높고, 좌우 근접."""
    n = min(LOOKBACK, len(high))
    lo, hi, cl = low[-n:], high[-n:], close[-n:]
    maxs = _local_maxima(hi)
    mins = _local_minima(lo)
    if len(maxs) < 3:
        return None

    for i in range(len(maxs) - 1, 1, -1):
        r_idx, h_idx, l_idx = maxs[i], maxs[i - 1], maxs[i - 2]
        if r_idx - l_idx < MIN_DISTANCE * 2:
            continue

        head = hi[h_idx]
        l_shoulder = hi[l_idx]
        r_shoulder = hi[r_idx]

        if head <= l_shoulder or head <= r_shoulder:
            continue
        avg_shoulder = (l_shoulder + r_shoulder) / 2
        if abs(l_shoulder - r_shoulder) / avg_shoulder > PROXIMITY_PCT * 1.5:
            continue

        troughs_l = [m for m in mins if l_idx < m < h_idx]
        troughs_r = [m for m in mins if h_idx < m < r_idx]
        if not troughs_l or not troughs_r:
            continue
        neck_l = lo[min(troughs_l, key=lambda m: lo[m])]
        neck_r = lo[min(troughs_r, key=lambda m: lo[m])]
        neck = (neck_l + neck_r) / 2

        height = head - neck
        if height / head < 0.002:
            continue

        price = cl[-1]
        dist = (neck - price) / neck
        if dist < -0.02 or dist > 0.015:
            continue

        strength = min(1.0, height / head / 0.01) * 0.8
        sl = head + atr * 0.3
        tp = neck - height
        return PatternResult(
            name="head_shoulders", direction="SHORT", strength=strength,
            entry_price=price, sl_price=sl, tp_price=tp,
            neckline=neck, pattern_height=height,
            detail={"head": round(head, 2),
                    "l_shoulder": round(l_shoulder, 2),
                    "r_shoulder": round(r_shoulder, 2)},
        )
    return None


# ─── 지속 패턴 ────────────────────────────────────────────

def detect_bull_flag(
    low: np.ndarray, high: np.ndarray, close: np.ndarray,
    volume: np.ndarray, atr: float,
) -> PatternResult | None:
    """상승 깃발 → LONG. 급등(깃대) 후 하향 조정(깃발) → 돌파."""
    if len(close) < 40:
        return None
    n = min(60, len(close))
    cl, hi, lo, vol = close[-n:], high[-n:], low[-n:], volume[-n:]

    # 깃대: 최근 30~60봉 내 가장 강한 상승 구간 찾기
    # 10봉 연속 상승률
    for pole_end in range(n - 15, n - 30, -1):
        pole_start = max(0, pole_end - 15)
        pole_rise = (cl[pole_end] - cl[pole_start]) / cl[pole_start]
        if pole_rise < 0.01:  # 최소 1% 상승
            continue

        # 깃발: 깃대 끝 이후 하향 조정 (고점 하락 + 저점 하락)
        flag = cl[pole_end:]
        if len(flag) < 5:
            continue
        flag_hi = hi[pole_end:]
        flag_lo = lo[pole_end:]

        # 조정 폭 확인: 깃대 상승분의 30~70% 조정
        flag_drop = (cl[pole_end] - flag_lo.min()) / cl[pole_end]
        if flag_drop < 0.003 or flag_drop > pole_rise * 0.7:
            continue

        # 현재가가 깃발 상단 근처에서 돌파 시도
        flag_top = flag_hi.max()
        price = cl[-1]
        if price < flag_top * 0.998:
            continue

        # 거래량: 깃대 구간 > 깃발 구간
        pole_vol = vol[pole_start:pole_end].mean()
        flag_vol = vol[pole_end:].mean()
        vol_shrink = flag_vol < pole_vol * 0.8

        strength = min(1.0, pole_rise / 0.02) * (0.9 if vol_shrink else 0.6)
        sl = flag_lo.min() - atr * 0.3
        tp = price + (cl[pole_end] - cl[pole_start])  # 깃대 높이만큼

        return PatternResult(
            name="bull_flag", direction="LONG", strength=strength,
            entry_price=price, sl_price=sl, tp_price=tp,
            pattern_height=cl[pole_end] - cl[pole_start],
            detail={"pole_rise_pct": round(pole_rise * 100, 2),
                    "flag_drop_pct": round(flag_drop * 100, 2),
                    "vol_shrink": vol_shrink},
        )
    return None


def detect_bear_flag(
    low: np.ndarray, high: np.ndarray, close: np.ndarray,
    volume: np.ndarray, atr: float,
) -> PatternResult | None:
    """하락 깃발 → SHORT. 급락 후 상향 조정 → 하향 돌파."""
    if len(close) < 40:
        return None
    n = min(60, len(close))
    cl, hi, lo, vol = close[-n:], high[-n:], low[-n:], volume[-n:]

    for pole_end in range(n - 15, n - 30, -1):
        pole_start = max(0, pole_end - 15)
        pole_drop = (cl[pole_start] - cl[pole_end]) / cl[pole_start]
        if pole_drop < 0.01:
            continue

        flag = cl[pole_end:]
        if len(flag) < 5:
            continue
        flag_hi = hi[pole_end:]
        flag_lo = lo[pole_end:]

        flag_rise = (flag_hi.max() - cl[pole_end]) / cl[pole_end]
        if flag_rise < 0.003 or flag_rise > pole_drop * 0.7:
            continue

        flag_bottom = flag_lo.min()
        price = cl[-1]
        if price > flag_bottom * 1.002:
            continue

        pole_vol = vol[pole_start:pole_end].mean()
        flag_vol = vol[pole_end:].mean()
        vol_shrink = flag_vol < pole_vol * 0.8

        strength = min(1.0, pole_drop / 0.02) * (0.9 if vol_shrink else 0.6)
        sl = flag_hi.max() + atr * 0.3
        tp = price - (cl[pole_start] - cl[pole_end])

        return PatternResult(
            name="bear_flag", direction="SHORT", strength=strength,
            entry_price=price, sl_price=sl, tp_price=tp,
            pattern_height=cl[pole_start] - cl[pole_end],
            detail={"pole_drop_pct": round(pole_drop * 100, 2),
                    "flag_rise_pct": round(flag_rise * 100, 2),
                    "vol_shrink": vol_shrink},
        )
    return None


def detect_triangle_breakout(
    low: np.ndarray, high: np.ndarray, close: np.ndarray,
    volume: np.ndarray, atr: float,
) -> PatternResult | None:
    """삼각수렴 돌파. 고점 하락 + 저점 상승 → 수렴 후 돌파 방향."""
    if len(close) < 30:
        return None
    n = min(60, len(close))
    hi, lo, cl, vol = high[-n:], low[-n:], close[-n:], volume[-n:]

    maxs = _local_maxima(hi, w=2)
    mins = _local_minima(lo, w=2)
    if len(maxs) < 3 or len(mins) < 3:
        return None

    # 고점 추세선 (하락?)
    recent_maxs = maxs[-4:]
    highs_vals = [hi[m] for m in recent_maxs]
    highs_declining = all(highs_vals[i] >= highs_vals[i + 1] for i in range(len(highs_vals) - 1))

    # 저점 추세선 (상승?)
    recent_mins = mins[-4:]
    lows_vals = [lo[m] for m in recent_mins]
    lows_rising = all(lows_vals[i] <= lows_vals[i + 1] for i in range(len(lows_vals) - 1))

    # 수렴 확인: 고점 하락 + 저점 상승
    if not (highs_declining and lows_rising):
        return None

    # 수렴 폭 확인
    apex_high = highs_vals[-1]
    apex_low = lows_vals[-1]
    width = (apex_high - apex_low) / apex_high
    if width > 0.02:  # 아직 충분히 수렴 안 됨
        return None

    price = cl[-1]
    height = highs_vals[0] - lows_vals[0]  # 삼각형 입구 높이

    # 돌파 방향 판단
    if price > apex_high:
        # 상향 돌파
        vol_avg = vol[-20:].mean()
        vol_now = vol[-1]
        vol_confirm = vol_now > vol_avg * 1.1

        strength = min(1.0, height / price / 0.01) * (0.8 if vol_confirm else 0.5)
        sl = apex_low - atr * 0.3
        tp = price + height * 0.6

        return PatternResult(
            name="triangle_up", direction="LONG", strength=strength,
            entry_price=price, sl_price=sl, tp_price=tp,
            pattern_height=height,
            detail={"width_pct": round(width * 100, 2),
                    "vol_confirm": vol_confirm},
        )
    elif price < apex_low:
        # 하향 돌파
        vol_avg = vol[-20:].mean()
        vol_now = vol[-1]
        vol_confirm = vol_now > vol_avg * 1.1

        strength = min(1.0, height / price / 0.01) * (0.8 if vol_confirm else 0.5)
        sl = apex_high + atr * 0.3
        tp = price - height * 0.6

        return PatternResult(
            name="triangle_down", direction="SHORT", strength=strength,
            entry_price=price, sl_price=sl, tp_price=tp,
            pattern_height=height,
            detail={"width_pct": round(width * 100, 2),
                    "vol_confirm": vol_confirm},
        )
    return None


# ─── 거래량 분석 ──────────────────────────────────────────

@dataclass
class VolumeSignal:
    """거래량 분석 결과."""
    bias: str        # "BULLISH" | "BEARISH" | "NEUTRAL"
    strength: float  # 0~1
    signals: list    # 감지된 신호 목록


def analyze_volume(
    close: np.ndarray, volume: np.ndarray, lookback: int = 20,
) -> VolumeSignal:
    """거래량 종합 분석: 다이버전스 + 스파이크 + OBV."""
    if len(close) < lookback + 5:
        return VolumeSignal("NEUTRAL", 0.0, [])

    signals = []
    score = 0  # 양수=BULLISH, 음수=BEARISH

    cl = close[-lookback:]
    vol = volume[-lookback:]

    # 1. 거래량 다이버전스
    price_trend = cl[-1] - cl[0]  # 가격 변화
    vol_trend = vol[-5:].mean() - vol[:5].mean()  # 거래량 변화

    if price_trend > 0 and vol_trend < 0:
        # 가격↑ + 거래량↓ = 약세 다이버전스 (상승 힘 약화)
        signals.append("bearish_divergence")
        score -= 2
    elif price_trend < 0 and vol_trend < 0:
        # 가격↓ + 거래량↓ = 하락 약화 (반등 가능)
        signals.append("bullish_exhaustion")
        score += 1
    elif price_trend > 0 and vol_trend > 0:
        # 가격↑ + 거래량↑ = 강한 상승
        signals.append("bullish_confirm")
        score += 2
    elif price_trend < 0 and vol_trend > 0:
        # 가격↓ + 거래량↑ = 강한 하락 (또는 셀링 클라이맥스)
        signals.append("bearish_confirm")
        score -= 2

    # 2. 거래량 스파이크 (최근 3봉 vs 평균)
    vol_avg = vol.mean()
    recent_vol = vol[-3:].mean()
    spike_ratio = recent_vol / vol_avg if vol_avg > 0 else 1.0

    if spike_ratio > 2.0:
        # 거래량 급등 → 방향 전환 가능
        if price_trend < 0:
            signals.append("selling_climax")
            score += 2  # 셀링 클라이맥스 → 반등 예측
        else:
            signals.append("buying_climax")
            score -= 1  # 바잉 클라이맥스 → 과열
    elif spike_ratio > 1.5:
        signals.append("volume_surge")

    # 3. OBV (On Balance Volume) 추세
    obv = np.zeros(len(cl))
    for i in range(1, len(cl)):
        if cl[i] > cl[i - 1]:
            obv[i] = obv[i - 1] + vol[i]
        elif cl[i] < cl[i - 1]:
            obv[i] = obv[i - 1] - vol[i]
        else:
            obv[i] = obv[i - 1]

    obv_trend = obv[-1] - obv[-5]
    if obv_trend > 0 and price_trend <= 0:
        signals.append("obv_bullish_div")  # OBV↑ + 가격↓ = 매집
        score += 2
    elif obv_trend < 0 and price_trend >= 0:
        signals.append("obv_bearish_div")  # OBV↓ + 가격↑ = 분산
        score -= 2

    # 종합 판단
    max_score = 6
    norm = min(1.0, abs(score) / max_score)
    if score > 0:
        bias = "BULLISH"
    elif score < 0:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return VolumeSignal(bias=bias, strength=norm, signals=signals)


# ─── 통합 스캐너 ─────────────────────────────────────────

def scan_all_patterns(
    low: np.ndarray, high: np.ndarray, close: np.ndarray,
    volume: np.ndarray, atr: float,
) -> list[PatternResult]:
    """모든 패턴을 스캔하고 감지된 것들을 반환."""
    results = []

    detectors = [
        lambda: detect_double_bottom(low, high, close, atr),
        lambda: detect_double_top(low, high, close, atr),
        lambda: detect_inv_head_shoulders(low, high, close, atr),
        lambda: detect_head_shoulders(low, high, close, atr),
        lambda: detect_bull_flag(low, high, close, volume, atr),
        lambda: detect_bear_flag(low, high, close, volume, atr),
        lambda: detect_triangle_breakout(low, high, close, volume, atr),
    ]

    for detect in detectors:
        try:
            p = detect()
            if p is not None:
                results.append(p)
        except Exception:
            continue

    # strength 기준 정렬
    results.sort(key=lambda p: p.strength, reverse=True)
    return results
