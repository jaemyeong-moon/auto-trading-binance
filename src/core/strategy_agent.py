"""AI Strategy Agent — 멀티 LLM 기반 전략 자동 생성/평가/교체 시스템.

지원 LLM: Anthropic Claude, OpenAI GPT, Google Gemini
(.env에 API 키만 넣으면 자동 감지)

주기적으로 트레이딩 성과를 분석하고, 현재 전략이 부진하면
LLM에게 신규 전략 코드를 작성하게 한 뒤, 검증(구문/백테스트)을 거쳐
자동으로 등록하고 교체한다.
"""

import ast
import importlib
import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import structlog

from src.core import database as db
from src.core.config import settings
from src.strategies.base import Strategy
from src.strategies.registry import _REGISTRY, get_strategy
from src.utils.timezone import now_kst

logger = structlog.get_logger()

# ─── 경로 ────────────────────────────────────────────────
AI_STRATEGY_DIR = Path(__file__).parent.parent / "strategies" / "ai_generated"
AI_STRATEGY_DIR.mkdir(parents=True, exist_ok=True)

# ─── 성과 임계값 ─────────────────────────────────────────
MIN_TRADES_FOR_EVAL = 10          # 최소 거래 수
POOR_WIN_RATE = 40.0              # 이 이하면 성과 부진
POOR_PNL_THRESHOLD = -50.0        # 총 PnL이 이 이하면 부진 (USDT)
EVAL_LOOKBACK_HOURS = 24          # 최근 24시간 거래 평가
MAX_AI_STRATEGIES = 5             # 최대 AI 생성 전략 수 (디스크 절약)
MAX_FIX_ATTEMPTS = 3              # 코드 검증 실패 시 최대 재시도 횟수

# ─── 백테스트 게이트 기준 ─────────────────────────────────
MIN_WIN_RATE = 0.45               # 백테스트 최소 승률 (45%)
MIN_PROFIT_FACTOR = 1.1           # 백테스트 최소 손익비


@dataclass
class PerformanceReport:
    """현재 전략의 성과 요약."""
    strategy_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_win: float
    max_loss: float
    consecutive_losses: int
    period_hours: int
    is_poor: bool
    reason: str = ""


@dataclass
class AgentReport:
    """AI Agent 실행 결과 보고서."""
    timestamp: datetime
    current_strategy: str
    performance: PerformanceReport
    action_taken: str  # "none" | "new_strategy_created" | "strategy_switched"
    new_strategy_name: str = ""
    new_strategy_file: str = ""
    analysis: str = ""
    backtest_result: dict = field(default_factory=dict)
    attempts: int = 0
    error: str = ""


# ─── 성과 분석 ────────────────────────────────────────────

def analyze_performance(strategy_name: str | None = None,
                        lookback_hours: int = EVAL_LOOKBACK_HOURS) -> PerformanceReport:
    """최근 거래 기록을 분석하여 성과 보고서 생성."""
    if strategy_name is None:
        strategy_name = db.get_setting("strategy")

    trades = db.get_trades(limit=500)
    cutoff = now_kst() - timedelta(hours=lookback_hours)

    # 해당 전략의 최근 거래만 필터
    recent = [
        t for t in trades
        if t.strategy == strategy_name
        and t.closed_at is not None
        and t.closed_at >= cutoff
    ]

    if not recent:
        return PerformanceReport(
            strategy_name=strategy_name,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, total_pnl=0.0, avg_pnl=0.0,
            max_win=0.0, max_loss=0.0, consecutive_losses=0,
            period_hours=lookback_hours, is_poor=False,
            reason="insufficient_data",
        )

    pnls = [t.pnl for t in recent if t.pnl is not None]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)
    win_rate = (len(wins) / len(pnls) * 100) if pnls else 0.0

    # 연속 손실 계산 (최신부터)
    consec_losses = 0
    for p in pnls:
        if p <= 0:
            consec_losses += 1
        else:
            break

    is_poor = False
    reason = ""
    if len(pnls) >= MIN_TRADES_FOR_EVAL:
        if win_rate < POOR_WIN_RATE:
            is_poor = True
            reason = f"low_win_rate({win_rate:.1f}%)"
        elif total_pnl < POOR_PNL_THRESHOLD:
            is_poor = True
            reason = f"negative_pnl({total_pnl:.2f})"
        elif consec_losses >= 5:
            is_poor = True
            reason = f"consecutive_losses({consec_losses})"

    return PerformanceReport(
        strategy_name=strategy_name,
        total_trades=len(pnls),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_pnl=total_pnl / len(pnls) if pnls else 0.0,
        max_win=max(wins) if wins else 0.0,
        max_loss=min(losses) if losses else 0.0,
        consecutive_losses=consec_losses,
        period_hours=lookback_hours,
        is_poor=is_poor,
        reason=reason,
    )


# ─── 프롬프트 빌드 ───────────────────────────────────────

def _build_strategy_base_code() -> str:
    """Strategy 베이스 클래스와 임포트 예시를 문자열로 반환."""
    return '''
# 필수 임포트 (이것만 사용 가능)
import numpy as np
import pandas as pd
import ta  # ta 라이브러리 (RSI, MACD, BB, ATR, ADX, EMA 등)

from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.registry import register

# ── Strategy ABC 인터페이스 ──
# class Strategy(ABC):
#     @property
#     @abstractmethod
#     def name(self) -> str: ...           # 고유 이름 (snake_case)
#
#     @property
#     def label(self) -> str: ...          # 표시명
#
#     @property
#     def description(self) -> str: ...    # 설명
#
#     @property
#     def mode(self) -> ExecutionMode: ... # ALWAYS_FLIP or SIGNAL_ONLY
#
#     @abstractmethod
#     def evaluate(self, symbol: str, candles: pd.DataFrame,
#                  htf_candles: pd.DataFrame | None = None) -> Signal:
#         """1분봉 candles (columns: open, high, low, close, volume)
#            htf_candles: 15분봉 (선택)
#            Returns: Signal(symbol, type, confidence, source, metadata)
#            type: SignalType.BUY | SELL | HOLD | CLOSE
#            confidence: 0.0 ~ 1.0"""
#
#     def record_result(self, pnl: float) -> None: ...  # 선택: 결과 추적

# ── Signal 생성 예시 ──
# Signal(symbol=symbol, type=SignalType.BUY, confidence=0.8,
#        source=self.name, metadata={"reason": "..."})

# ── 반드시 @register 데코레이터 사용 ──
# @register
# class MyStrategy(Strategy):
#     ...
'''


def _build_existing_strategies_summary() -> str:
    """기존 전략들의 핵심 요약."""
    summaries = []
    for name, cls in _REGISTRY.items():
        inst = cls()
        summaries.append(f"- {name}: {inst.label} — {inst.description}")
    return "\n".join(summaries)


def _build_analysis_prompt(perf: PerformanceReport,
                           market_data_summary: str = "") -> str:
    """LLM에게 보낼 분석 + 전략 생성 프롬프트."""
    return f"""당신은 암호화폐 선물 트레이딩 전략을 설계하는 전문가입니다.

## 현재 상황

현재 전략: {perf.strategy_name}
평가 기간: 최근 {perf.period_hours}시간
총 거래: {perf.total_trades}건
승률: {perf.win_rate:.1f}%
총 손익: {perf.total_pnl:+.2f} USDT
평균 손익: {perf.avg_pnl:+.2f} USDT
최대 수익: {perf.max_win:+.2f} USDT
최대 손실: {perf.max_loss:+.2f} USDT
연속 손실: {perf.consecutive_losses}회
성과 부진 판정: {perf.is_poor} ({perf.reason})

{f"## 최근 시장 데이터 요약" + chr(10) + market_data_summary if market_data_summary else ""}

## 기존 전략 목록
{_build_existing_strategies_summary()}

## 작업 지침

1. 먼저 현재 전략의 문제점을 분석하세요.
2. 기존 전략들과 차별화된 새로운 전략을 설계하세요.
3. 아래 규칙을 반드시 준수하는 Python 코드를 작성하세요.

## 코드 규칙 (엄격 준수)

{_build_strategy_base_code()}

### 반드시 지켜야 할 규칙:
1. `@register` 데코레이터를 클래스 위에 반드시 붙일 것
2. `name` 프로퍼티는 **고유한 snake_case 문자열** (기존 이름과 겹치면 안됨)
3. `mode`는 `ExecutionMode.SIGNAL_ONLY` 사용 권장
4. `evaluate()` 메서드는 반드시 `Signal` 객체를 반환
5. candles DataFrame은 columns: open, high, low, close, volume
6. 최소 50개 이상의 캔들이 필요하면 len(candles) < N 체크 후 HOLD 반환
7. 외부 API 호출이나 파일 I/O 금지 — ta 라이브러리와 pandas/numpy만 사용
8. 클래스 하나만 정의할 것
9. 전략 이름은 "ai_" 접두사로 시작할 것 (예: "ai_mean_reversion_v1")

## 출력 형식

반드시 아래 형식으로 출력하세요:

### 분석
(현재 전략 문제점 분석)

### 전략 설계
(새 전략의 핵심 아이디어)

### 코드
```python
(완전한 Python 코드 — 임포트부터 클래스 정의까지)
```
"""


def _build_evaluation_prompt(perf: PerformanceReport) -> str:
    """전략 교체 없이 분석만 하는 프롬프트."""
    return f"""당신은 암호화폐 선물 트레이딩 전략을 평가하는 전문가입니다.

## 현재 전략 성과

전략: {perf.strategy_name}
평가 기간: 최근 {perf.period_hours}시간
총 거래: {perf.total_trades}건
승률: {perf.win_rate:.1f}%
총 손익: {perf.total_pnl:+.2f} USDT
평균 손익: {perf.avg_pnl:+.2f} USDT
최대 수익: {perf.max_win:+.2f} USDT
최대 손실: {perf.max_loss:+.2f} USDT
연속 손실: {perf.consecutive_losses}회

## 기존 전략 목록
{_build_existing_strategies_summary()}

## 작업

1. 현재 전략의 성과를 평가하세요.
2. 개선 제안이 있으면 구체적으로 알려주세요.
3. 전략 교체가 필요한지 판단하세요.

답변은 한국어로, 핵심만 간결하게 작성하세요 (200자 이내).
"""


def _build_fix_prompt(error: str, code_or_hint: str = "") -> str:
    """검증 실패 시 LLM에게 보낼 수정 요청 프롬프트."""
    parts = [
        "## 코드 검증 실패 — 수정 필요\n",
        f"**에러:**\n```\n{error}\n```\n",
    ]
    if code_or_hint and not code_or_hint.startswith("응답에"):
        parts.append(
            "위 에러를 수정한 **완전한 코드**를 다시 작성해주세요.\n"
            "부분 수정이 아니라, 처음부터 끝까지 전체 코드를 ```python 블록에 넣어주세요.\n"
        )
    else:
        parts.append(f"{code_or_hint}\n")

    parts.append(
        "### 체크리스트\n"
        "- [ ] `@register` 데코레이터 있는가?\n"
        "- [ ] `class XxxStrategy(Strategy):` 형태인가?\n"
        "- [ ] `name` 프로퍼티가 `\"ai_\"` 로 시작하는 문자열을 반환하는가?\n"
        "- [ ] `evaluate()` 메서드가 `Signal` 객체를 반환하는가?\n"
        "- [ ] `SignalType.BUY/SELL/HOLD` 를 사용하는가?\n"
        "- [ ] 금지된 임포트(os, subprocess, exec 등)가 없는가?\n"
        "- [ ] 코드가 ```python ... ``` 블록 안에 있는가?\n"
    )
    return "\n".join(parts)


# ─── 코드 생성 & 검증 ────────────────────────────────────

def _extract_code_block(response_text: str) -> str:
    """LLM 응답에서 ```python 코드 블록을 추출."""
    lines = response_text.split("\n")
    in_code = False
    code_lines = []

    for line in lines:
        if line.strip().startswith("```python"):
            in_code = True
            continue
        if line.strip() == "```" and in_code:
            in_code = False
            continue
        if in_code:
            code_lines.append(line)

    return "\n".join(code_lines).strip()


def _extract_analysis(response_text: str) -> str:
    """LLM 응답에서 분석 텍스트를 추출."""
    # 코드 블록 이전의 텍스트
    parts = response_text.split("```python")
    if parts:
        return parts[0].strip()
    return response_text[:500]


def _validate_syntax(code: str) -> tuple[bool, str]:
    """Python 구문 검증."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def _validate_strategy_structure(code: str) -> tuple[bool, str]:
    """전략 코드의 필수 구조 검증."""
    checks = [
        ("@register", "Missing @register decorator"),
        ("class ", "Missing class definition"),
        ("Strategy", "Must inherit from Strategy"),
        ("def evaluate(self", "Missing evaluate() method"),
        ("def name(self", "Missing name property"),
        ("SignalType.", "Must use SignalType for signals"),
        ('"ai_', 'Strategy name must start with "ai_"'),
    ]
    for pattern, msg in checks:
        if pattern not in code:
            return False, msg
    return True, ""


def _validate_no_dangerous_code(code: str) -> tuple[bool, str]:
    """AST 기반 위험 코드 검출.

    허용 최상위 모듈 화이트리스트와 금지 함수 호출 블랙리스트를 AST 순회로 검사한다.
    문자열 패턴 매칭 대신 AST를 사용하므로 우회(obfuscation)를 방지한다.
    """
    ALLOWED_TOP_MODULES = {
        "numpy", "pandas", "ta", "src", "np", "pd", "math", "dataclasses",
    }
    FORBIDDEN_CALLS = {
        "exec", "eval", "compile", "__import__", "open",
    }

    class _DangerVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.error: str = ""

        def _top_module(self, name: str) -> str:
            """'a.b.c' → 'a'"""
            return name.split(".")[0]

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                top = self._top_module(alias.name)
                if top not in ALLOWED_TOP_MODULES:
                    self.error = f"import '{alias.name}' not allowed (module: {top})"
                    return
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            module = node.module or ""
            top = self._top_module(module)
            if top and top not in ALLOWED_TOP_MODULES:
                self.error = f"from '{module}' import not allowed (module: {top})"
                return
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            # Direct function call: exec(...), eval(...), open(...)
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    self.error = f"call to '{node.func.id}' not allowed"
                    return
            # Attribute call: builtins.eval(...) etc.
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_CALLS:
                    self.error = f"call to '{node.func.attr}' not allowed"
                    return
            self.generic_visit(node)

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        # Syntax errors are handled upstream; treat as safe here
        return False, f"SyntaxError during security check: {exc}"

    visitor = _DangerVisitor()
    visitor.visit(tree)
    if visitor.error:
        return False, visitor.error
    return True, ""


def _save_strategy_file(code: str, strategy_name: str) -> Path:
    """전략 코드를 파일로 저장."""
    filename = f"{strategy_name}.py"
    filepath = AI_STRATEGY_DIR / filename
    filepath.write_text(code, encoding="utf-8")
    return filepath


def _load_strategy_module(filepath: Path, strategy_name: str) -> tuple[bool, str]:
    """저장된 전략 모듈을 동적으로 로드하고 레지스트리에 등록."""
    module_name = f"src.strategies.ai_generated.{strategy_name}"

    # 이미 로드된 모듈이 있으면 제거
    if module_name in sys.modules:
        del sys.modules[module_name]

    try:
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            return False, "Failed to create module spec"

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return True, ""
    except Exception:
        tb = traceback.format_exc()
        return False, f"Module load error:\n{tb}"


def _run_backtest_validation(strategy_name: str,
                             candles: pd.DataFrame) -> tuple[bool, dict]:
    """백테스트로 전략 기본 동작 검증.

    통과 기준:
    - 최소 3건 이상 거래
    - 승률 MIN_WIN_RATE(45%) 이상
    - 손익비(Profit Factor) MIN_PROFIT_FACTOR(1.1) 이상
    """
    from src.backtesting.backtest import Backtester

    try:
        strategy = get_strategy(strategy_name)
        bt = Backtester(strategy, initial_capital=10000.0)
        result = bt.run("BTCUSDT", candles)

        # Profit Factor = 총 이익 / 총 손실 (절대값)
        gross_profit = sum(t.pnl for t in result.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in result.trades if t.pnl <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        )

        win_rate_ratio = result.win_rate / 100.0  # BacktestResult.win_rate는 % 단위

        summary = {
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_factor": round(profit_factor, 3),
            "total_return_pct": round(result.total_return_pct, 2),
            "max_drawdown_pct": round(result.max_drawdown_pct, 2),
            "sharpe_ratio": round(result.sharpe_ratio, 2),
        }

        # 최소한 거래가 발생해야 유효
        if result.total_trades < 3:
            logger.info("agent.backtest_gate_rejected",
                        strategy=strategy_name, reason="too_few_trades",
                        total_trades=result.total_trades)
            return False, {**summary, "error": "Too few trades in backtest"}

        # 승률 기준 미달
        if win_rate_ratio < MIN_WIN_RATE:
            logger.info("agent.backtest_gate_rejected",
                        strategy=strategy_name, reason="low_win_rate",
                        win_rate=result.win_rate, min_win_rate=MIN_WIN_RATE * 100)
            return False, {**summary,
                           "error": f"Win rate {result.win_rate:.1f}% < {MIN_WIN_RATE * 100:.0f}% minimum"}

        # 손익비 기준 미달
        if profit_factor < MIN_PROFIT_FACTOR:
            logger.info("agent.backtest_gate_rejected",
                        strategy=strategy_name, reason="low_profit_factor",
                        profit_factor=profit_factor, min_profit_factor=MIN_PROFIT_FACTOR)
            return False, {**summary,
                           "error": f"Profit factor {profit_factor:.3f} < {MIN_PROFIT_FACTOR} minimum"}

        return True, summary
    except Exception:
        tb = traceback.format_exc()
        logger.info("agent.backtest_gate_rejected",
                    strategy=strategy_name, reason="exception", error=tb[:200])
        return False, {"error": f"Backtest failed:\n{tb}"}


def _cleanup_old_strategies() -> None:
    """오래된 AI 전략 파일 정리 (MAX_AI_STRATEGIES 초과 시)."""
    files = sorted(AI_STRATEGY_DIR.glob("ai_*.py"), key=lambda f: f.stat().st_mtime)
    while len(files) > MAX_AI_STRATEGIES:
        old_file = files.pop(0)
        old_name = old_file.stem
        # 현재 활성 전략이면 건너뜀
        if db.get_setting("strategy") == old_name:
            continue
        # 레지스트리에서 제거
        if old_name in _REGISTRY:
            del _REGISTRY[old_name]
        old_file.unlink(missing_ok=True)
        logger.info("agent.cleanup_old_strategy", removed=old_name)


# ─── 메인 에이전트 ────────────────────────────────────────

class AIStrategyAgent:
    """LLM을 사용해 전략을 자동 분석/생성/교체하는 에이전트.

    지원 프로바이더: anthropic (Claude), openai (GPT), gemini (Gemini).
    .env에 API 키를 넣으면 자동 감지되고, provider 인자로 직접 지정도 가능.
    """

    def __init__(self, provider: str | None = None,
                 api_key: str | None = None) -> None:
        from src.core.llm_provider import create_provider
        self._llm = create_provider(provider_name=provider, api_key=api_key)
        self._last_report: AgentReport | None = None

    async def run(self, candles_for_backtest: pd.DataFrame | None = None,
                  market_data_summary: str = "",
                  force: bool = False) -> AgentReport:
        """에이전트 실행 메인 루프.

        Args:
            candles_for_backtest: 백테스트용 캔들 데이터 (없으면 검증 스킵)
            market_data_summary: 최근 시장 요약 (있으면 프롬프트에 포함)
            force: True면 성과와 무관하게 신규 전략 생성

        Returns:
            AgentReport — 실행 결과 보고서
        """
        timestamp = now_kst()
        current_strategy = db.get_setting("strategy")

        # 1. 성과 분석
        perf = analyze_performance(current_strategy)
        logger.info("agent.performance_analyzed",
                     strategy=current_strategy,
                     trades=perf.total_trades,
                     win_rate=f"{perf.win_rate:.1f}%",
                     pnl=f"{perf.total_pnl:+.2f}",
                     is_poor=perf.is_poor)

        # 2. 데이터 부족 or 양호 → 분석만
        if not force and (perf.total_trades < MIN_TRADES_FOR_EVAL or not perf.is_poor):
            analysis = await self._evaluate_only(perf)
            report = AgentReport(
                timestamp=timestamp,
                current_strategy=current_strategy,
                performance=perf,
                action_taken="none",
                analysis=analysis,
            )
            self._last_report = report
            return report

        # 3. 성과 부진 → 신규 전략 생성
        logger.info("agent.generating_new_strategy",
                     reason=perf.reason, strategy=current_strategy)

        report = await self._generate_new_strategy(
            perf, timestamp, current_strategy,
            candles_for_backtest, market_data_summary,
        )
        self._last_report = report
        return report

    async def _evaluate_only(self, perf: PerformanceReport) -> str:
        """전략 교체 없이 분석만 수행."""
        try:
            prompt = _build_evaluation_prompt(perf)
            return self._llm.chat(prompt, max_tokens=500)
        except Exception as e:
            logger.exception("agent.evaluation_failed")
            return f"Evaluation failed: {e}"

    async def _generate_new_strategy(
        self,
        perf: PerformanceReport,
        timestamp: datetime,
        current_strategy: str,
        candles: pd.DataFrame | None,
        market_data_summary: str,
    ) -> AgentReport:
        """LLM에게 신규 전략을 생성하게 하고, 실패 시 에러를 피드백하여 재시도.

        흐름:
        1. 첫 번째 프롬프트로 전략 생성 요청
        2. 검증 (구문 → 구조 → 보안 → 로드 → 백테스트)
        3. 실패 시 → 에러 메시지를 LLM에게 돌려보내 수정 요청
        4. 최대 MAX_FIX_ATTEMPTS(3)회까지 반복
        5. 모두 실패하면 현재 전략 유지
        """
        from src.core.llm_provider import Message

        # 대화 히스토리 유지 (LLM이 이전 시도의 맥락을 기억)
        initial_prompt = _build_analysis_prompt(perf, market_data_summary)
        conversation: list[Message] = [Message("user", initial_prompt)]

        analysis = ""
        all_errors: list[str] = []

        for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
            # LLM 호출
            try:
                if attempt == 1:
                    response_text = self._llm.chat(initial_prompt, max_tokens=4096)
                else:
                    response_text = self._llm.chat_messages(
                        conversation, max_tokens=4096,
                    )
            except Exception as e:
                logger.exception("agent.llm_api_failed",
                                 provider=self._llm.name, attempt=attempt)
                return AgentReport(
                    timestamp=timestamp,
                    current_strategy=current_strategy,
                    performance=perf,
                    action_taken="none",
                    error=f"LLM API error ({self._llm.name}): {e}",
                )

            # 대화 히스토리에 추가
            conversation.append(Message("assistant", response_text))

            if attempt == 1:
                analysis = _extract_analysis(response_text)

            code = _extract_code_block(response_text)
            if not code:
                err_msg = "No ```python code block found in response."
                all_errors.append(f"[attempt {attempt}] {err_msg}")
                logger.warning("agent.no_code_block", attempt=attempt)
                conversation.append(Message("user", _build_fix_prompt(
                    err_msg,
                    "응답에 ```python 코드 블록이 없습니다. "
                    "반드시 ```python ... ``` 형식으로 완전한 코드를 포함해주세요."
                )))
                continue

            # 검증 파이프라인
            validation_error = self._run_validation_pipeline(code)
            if validation_error:
                all_errors.append(f"[attempt {attempt}] {validation_error}")
                logger.warning("agent.validation_failed",
                               attempt=attempt, error=validation_error)
                conversation.append(Message("user", _build_fix_prompt(
                    validation_error, code,
                )))
                continue

            # 전략 이름 추출
            strategy_name = self._extract_strategy_name(code)
            if not strategy_name:
                err_msg = (
                    "Could not extract strategy name. "
                    'The name property must return a string starting with "ai_".'
                )
                all_errors.append(f"[attempt {attempt}] {err_msg}")
                conversation.append(Message("user", _build_fix_prompt(
                    err_msg, code,
                )))
                continue

            # 파일 저장 + 동적 로드
            filepath = _save_strategy_file(code, strategy_name)
            ok, load_err = _load_strategy_module(filepath, strategy_name)
            if not ok:
                filepath.unlink(missing_ok=True)
                all_errors.append(f"[attempt {attempt}] Module load: {load_err}")
                logger.warning("agent.load_failed",
                               attempt=attempt, error=load_err[:200])
                conversation.append(Message("user", _build_fix_prompt(
                    f"Module load failed:\n{load_err}", code,
                )))
                continue

            # 백테스트 검증
            backtest_result = {}
            if candles is not None and len(candles) > 100:
                bt_ok, backtest_result = _run_backtest_validation(
                    strategy_name, candles,
                )
                if not bt_ok:
                    bt_err = backtest_result.get("error", "Too few trades")
                    all_errors.append(
                        f"[attempt {attempt}] Backtest: {bt_err}"
                    )
                    logger.warning("agent.backtest_failed",
                                   attempt=attempt, result=backtest_result)

                    # 레지스트리에서 제거 후 재시도
                    if strategy_name in _REGISTRY:
                        del _REGISTRY[strategy_name]
                    filepath.unlink(missing_ok=True)

                    conversation.append(Message("user", _build_fix_prompt(
                        f"Backtest failed: {bt_err}\n"
                        "전략이 실제로 거래를 발생시켜야 합니다. "
                        "evaluate()가 BUY/SELL 신호를 충분히 생성하는지 확인하고, "
                        "조건을 너무 엄격하게 만들지 마세요.",
                        code,
                    )))
                    continue

            # 모든 검증 통과 — 전략 교체
            db.set_setting("strategy", strategy_name)
            _cleanup_old_strategies()

            logger.info("agent.strategy_switched",
                         old=current_strategy, new=strategy_name,
                         attempts=attempt, backtest=backtest_result)

            return AgentReport(
                timestamp=timestamp,
                current_strategy=current_strategy,
                performance=perf,
                action_taken="strategy_switched",
                new_strategy_name=strategy_name,
                new_strategy_file=str(filepath),
                analysis=analysis,
                backtest_result=backtest_result,
                attempts=attempt,
            )

        # 모든 재시도 실패
        error_summary = "\n".join(all_errors)
        logger.error("agent.all_attempts_failed",
                      attempts=MAX_FIX_ATTEMPTS, errors=error_summary[:500])
        return AgentReport(
            timestamp=timestamp,
            current_strategy=current_strategy,
            performance=perf,
            action_taken="none",
            analysis=analysis,
            error=f"All {MAX_FIX_ATTEMPTS} attempts failed:\n{error_summary}",
        )

    def _run_validation_pipeline(self, code: str) -> str:
        """코드 검증 파이프라인. 통과하면 빈 문자열, 실패하면 에러 메시지."""
        for validate_fn in [_validate_syntax, _validate_strategy_structure,
                            _validate_no_dangerous_code]:
            ok, err = validate_fn(code)
            if not ok:
                return f"{validate_fn.__name__}: {err}"
        return ""

    def _extract_strategy_name(self, code: str) -> str:
        """코드에서 전략 name 프로퍼티 값을 추출."""
        # return "ai_xxx" 패턴 찾기
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("return") and '"ai_' in stripped:
                # return "ai_something"
                start = stripped.index('"ai_')
                end = stripped.index('"', start + 1)
                return stripped[start + 1:end]
        return ""

    def get_last_report(self) -> AgentReport | None:
        return self._last_report

    def format_report(self, report: AgentReport) -> str:
        """사람이 읽기 좋은 보고서 형식으로 변환."""
        perf = report.performance
        lines = [
            "=" * 50,
            f"  AI Strategy Agent Report",
            f"  {report.timestamp.strftime('%Y-%m-%d %H:%M:%S KST')}",
            f"  LLM: {self._llm.name}",
            "=" * 50,
            "",
            f"  현재 전략: {report.current_strategy}",
            f"  평가 기간: 최근 {perf.period_hours}시간",
            f"  총 거래: {perf.total_trades}건",
            f"  승률: {perf.win_rate:.1f}%",
            f"  총 손익: {perf.total_pnl:+.2f} USDT",
            f"  연속 손실: {perf.consecutive_losses}회",
            f"  성과 부진: {'Yes' if perf.is_poor else 'No'}"
            + (f" ({perf.reason})" if perf.reason else ""),
            "",
        ]

        if report.analysis:
            lines.append("  [AI 분석]")
            # 줄바꿈 처리
            for al in report.analysis.split("\n")[:10]:
                lines.append(f"  {al}")
            lines.append("")

        if report.action_taken == "strategy_switched":
            lines.append(f"  [조치] 전략 교체: {report.current_strategy} -> {report.new_strategy_name}")
            if report.attempts > 1:
                lines.append(f"  시도: {report.attempts}회 (자동 수정 {report.attempts - 1}회)")
            lines.append(f"  파일: {report.new_strategy_file}")
            if report.backtest_result:
                bt = report.backtest_result
                lines.append(f"  백테스트: {bt.get('total_trades', 0)}건, "
                             f"승률 {bt.get('win_rate', 0):.1f}%, "
                             f"수익률 {bt.get('total_return_pct', 0):.2f}%")
        elif report.action_taken == "none":
            lines.append("  [조치] 없음 — 현재 전략 유지")

        if report.error:
            lines.append(f"  [오류] {report.error}")

        lines.append("")
        lines.append("=" * 50)
        return "\n".join(lines)


# ─── 편의 함수 ────────────────────────────────────────────

def load_all_ai_strategies() -> None:
    """서버 시작 시 기존 AI 생성 전략들을 모두 로드."""
    for filepath in AI_STRATEGY_DIR.glob("ai_*.py"):
        strategy_name = filepath.stem
        ok, err = _load_strategy_module(filepath, strategy_name)
        if ok:
            logger.info("agent.loaded_ai_strategy", name=strategy_name)
        else:
            logger.warning("agent.failed_to_load_ai_strategy",
                           name=strategy_name, error=err)
