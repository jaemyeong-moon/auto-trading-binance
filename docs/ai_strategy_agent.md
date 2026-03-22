# AI Strategy Agent — 자동 전략 생성 시스템

> 최초 적용일: 2026-03-21
> 상태: **운용 가능**
> 코드: `src/core/strategy_agent.py`, `src/core/llm_provider.py`

---

## 한줄 요약

트레이딩 성과를 주기적으로 분석하고, 성과가 부진하면 AI(Claude/GPT/Gemini 중 택1)가 **신규 전략 코드를 자동으로 작성 → 검증 → 교체**한다. 컴퓨터만 켜 두면 봇이 스스로 전략을 진화시킨다.

---

## 지원 LLM

| 프로바이더 | 모델 | 환경변수 | 설치 패키지 |
|-----------|------|---------|------------|
| **Anthropic** | Claude Sonnet 4 | `ANTHROPIC_API_KEY` | `anthropic` |
| **OpenAI** | GPT-4o | `OPENAI_API_KEY` | `openai` |
| **Google** | Gemini 2.5 Flash | `GEMINI_API_KEY` | `google-genai` |

**자동 감지**: `.env`에 API 키를 넣으면 자동으로 해당 프로바이더를 사용한다. 여러 키가 있으면 우선순위: Anthropic > OpenAI > Gemini.

---

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install -e ".[agent]"       # 3개 LLM SDK 전부 설치
# 또는 전체 설치
pip install -e ".[dev,ml,dashboard,agent]"
```

필요한 프로바이더 하나만 설치해도 된다:
```bash
pip install anthropic          # Claude만
pip install openai             # GPT만
pip install google-genai       # Gemini만
```

### 2. API 키 설정 (3개 중 하나만 있으면 됨)

`.env` 파일:

```env
# 기존 설정
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret

# ── AI Agent용 (아래 3개 중 하나만 설정하면 됨) ──

# 방법 1: Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-api03-...

# 방법 2: OpenAI GPT
OPENAI_API_KEY=sk-proj-...

# 방법 3: Google Gemini
GEMINI_API_KEY=AIzaSy...
```

### 3. 실행

```bash
python -m src.main
```

시작 시 다음과 같이 표시되면 정상:

```
⚡ Futures Scalper started
  Symbols: ['BTCUSDT', 'ETHUSDT']
  Leverage: 5x
  Strategy: momentum_flip_scalper
  Testnet: True
  AI Agent: enabled (provider: openai)     ← 감지된 프로바이더 표시
```

`AI Agent: disabled` 라고 뜨면 `.env`의 API 키를 확인한다.

### 4. 그 다음?

**아무것도 안 해도 된다.** 컴퓨터를 켜 두면 봇이 알아서:
- 매시간 트레이딩 성과를 점검하고
- 문제가 있으면 새 전략을 만들어 교체하고
- 웹훅(Discord/Slack)으로 보고한다

---

## 동작 흐름

### 전체 타임라인

```
봇 시작
  │
  ├─ DB 초기화
  ├─ 기존 AI 전략 파일 로드 (src/strategies/ai_generated/*.py)
  ├─ LLM 프로바이더 자동 감지 (환경변수 기반)
  ├─ 거래소 연결
  ├─ 심볼별 트레이딩 루프 시작
  │
  │   ┌─────────── 매 30초 (tick) ───────────┐
  │   │                                       │
  │   │  [매 틱] 전략 evaluate → 매매 실행     │
  │   │  [20틱마다] TP/SL 자동 최적화          │
  │   │  [120틱마다] ★ AI Agent 실행           │
  │   │                                       │
  │   └───────────────────────────────────────┘
  │
  ▼ (Ctrl+C로 종료)
```

### AI Agent 상세 흐름

```
★ AI Agent 실행 (약 1시간 주기)
  │
  ├─ 1단계: 성과 분석
  │   └─ DB에서 현재 전략의 최근 24시간 거래 기록 조회
  │       ├─ 총 거래수, 승률, 총 PnL, 연속 손실 등 계산
  │       └─ PerformanceReport 생성
  │
  ├─ 2단계: 부진 여부 판정
  │   │
  │   ├─ [거래 < 10건] → "데이터 부족" → 분석만 하고 종료
  │   │
  │   ├─ [성과 양호] → LLM에게 짧은 평가만 요청 → 종료
  │   │   (현재 전략 유지, 개선 제안만 로그에 기록)
  │   │
  │   └─ [성과 부진] → 3단계로 진행
  │       판정 기준:
  │       • 승률 < 40%
  │       • 총 PnL < -50 USDT
  │       • 연속 5회 이상 손실
  │
  ├─ 3단계: 신규 전략 생성 + 자동 수정 루프 (최대 3회)
  │   │
  │   │  LLM에게 보내는 프롬프트:
  │   │  ┌──────────────────────────────────┐
  │   │  │ • 현재 전략의 상세 성과 데이터     │
  │   │  │ • 최근 시장 상황 (가격, ATR, 변동) │
  │   │  │ • 기존 전략 목록 및 설명           │
  │   │  │ • Strategy ABC 인터페이스 명세     │
  │   │  │ • 코드 규칙 (임포트 제한 등)       │
  │   │  └──────────────────────────────────┘
  │   │
  │   │  ┌─────────────── 재시도 루프 ───────────────┐
  │   │  │                                            │
  │   │  │  [1회차] LLM에게 전략 생성 요청              │
  │   │  │     ↓                                      │
  │   │  │  5중 검증 실행                              │
  │   │  │     ↓                                      │
  │   │  │  [실패 시] 에러 메시지를 LLM에게 피드백       │
  │   │  │     "SyntaxError at line 42: ..."          │
  │   │  │     "체크리스트 확인 후 전체 코드 재작성"      │
  │   │  │     ↓                                      │
  │   │  │  [2회차] LLM이 에러를 보고 수정된 코드 생성   │
  │   │  │     ↓                                      │
  │   │  │  5중 검증 재실행                            │
  │   │  │     ↓                                      │
  │   │  │  [실패 시] 다시 피드백 → [3회차] ...         │
  │   │  │     ↓                                      │
  │   │  │  [3회 모두 실패] → 현재 전략 유지, 오류 로그  │
  │   │  │                                            │
  │   │  └────────────────────────────────────────────┘
  │   │
  │   │  ※ 대화 히스토리가 유지되므로 LLM이 이전 시도와
  │   │    에러를 기억하고 점진적으로 수정함
  │
  ├─ 4단계: 5중 검증 파이프라인 (매 시도마다 실행)
  │   │
  │   ├─ ① 구문 검증 (ast.parse)
  │   │     → SyntaxError 있으면 에러 피드백
  │   │
  │   ├─ ② 구조 검증
  │   │     → @register, Strategy 상속, evaluate() 등 누락 시 피드백
  │   │
  │   ├─ ③ 보안 검증
  │   │     → os, subprocess, exec 등 감지 시 피드백
  │   │
  │   ├─ ④ 동적 로드
  │   │     → import 실패 시 traceback을 LLM에게 전달
  │   │
  │   └─ ⑤ 백테스트
  │         → 거래 미발생 시 "조건이 너무 엄격" 피드백
  │
  ├─ 5단계: 전략 교체 (핫 스왑)
  │   │
  │   ├─ 전략 파일 저장: src/strategies/ai_generated/ai_xxx.py
  │   ├─ DB 설정 업데이트: strategy = "ai_xxx"
  │   ├─ 엔진 내 전략 인스턴스 교체 (다음 틱부터 적용)
  │   └─ 오래된 AI 전략 정리 (최대 5개 유지)
  │
  └─ 6단계: 보고
      │
      ├─ 구조화된 로그 출력 (structlog)
      └─ 웹훅 알림 (Discord/Slack, 설정된 경우)
          "🤖 AI Agent: 전략 교체
           momentum_flip_scalper → ai_mean_reversion_v1
           사유: low_win_rate(35.2%)
           LLM: openai/gpt-4o"
```

### 코드가 만들어지는 과정 (상세)

```
 "전략이 승률 33%야, 새 전략 짜줘"    ← 1회차 프롬프트
              │
              ▼
      ┌──────────────┐
      │  LLM (API)   │  Claude / GPT / Gemini
      └──────┬───────┘
              │
              ▼
  "### 분석                            ← LLM 응답 (텍스트)
   현재 전략은 횡보장에서 ...
   ### 코드
   ```python
   @register
   class AiMeanReversionV1(Strategy):
       ...
   ```"
              │
              ▼
      ┌──────────────┐
      │ 코드 추출     │  ```python ... ``` 블록만 파싱
      └──────┬───────┘
              │
              ▼
      ┌──────────────┐
      │ 5중 검증      │  구문 → 구조 → 보안 → 로드 → 백테스트
      └──────┬───────┘
              │
         통과? ─── Yes ──→ 파일 저장 → 핫 스왑 → 완료!
              │
             No
              │
              ▼
      ┌──────────────────────────────────┐
      │  에러 피드백 프롬프트 생성         │
      │                                  │
      │  "SyntaxError at line 42:        │
      │   unexpected indent              │
      │                                  │
      │   수정한 전체 코드를 다시 작성하세요. │
      │   체크리스트:                      │
      │   - [ ] @register 있는가?         │
      │   - [ ] name이 ai_로 시작하는가?   │
      │   - [ ] ..."                     │
      └──────────┬───────────────────────┘
                 │
                 ▼
         ┌──────────────┐
         │  LLM (2회차)  │  이전 대화 맥락 + 에러 메시지를 보고 수정
         └──────┬───────┘
                │
                ▼
         5중 검증 재실행
                │
           통과? ─── Yes ──→ 완료!
                │
               No → 3회차까지 반복
                │
         3회 모두 실패 → 현재 전략 유지, 오류 로그
```

---

## 설정값

대시보드 또는 DB에서 조정 가능한 값들:

| 설정 키 | 기본값 | 설명 |
|---------|--------|------|
| `ai_agent_enabled` | `true` | AI Agent 활성화 여부 |
| `ai_agent_interval` | `120` | 실행 주기 (틱 수, 120 × 30초 = ~1시간) |

성과 판정 기준 (`strategy_agent.py` 상수):

| 상수 | 값 | 설명 |
|------|-----|------|
| `MIN_TRADES_FOR_EVAL` | 10 | 최소 거래 수 (미만이면 판정 보류) |
| `POOR_WIN_RATE` | 40% | 이하면 성과 부진 |
| `POOR_PNL_THRESHOLD` | -50 USDT | 이하면 성과 부진 |
| `EVAL_LOOKBACK_HOURS` | 24 | 평가 기간 (시간) |
| `MAX_AI_STRATEGIES` | 5 | 디스크에 보관할 최대 AI 전략 수 |
| `MAX_FIX_ATTEMPTS` | 3 | 코드 검증 실패 시 LLM 재시도 횟수 |

---

## 파일 구조

```
src/
├── core/
│   ├── strategy_agent.py          ← AI Agent 메인 (성과 분석, 검증, 교체)
│   └── llm_provider.py            ← LLM 추상화 (Claude/GPT/Gemini 통합)
├── strategies/
│   ├── ai_generated/              ← AI가 만든 전략들이 여기 저장됨
│   │   ├── __init__.py
│   │   ├── ai_mean_reversion_v1.py    (예시)
│   │   └── ai_volatility_breakout.py  (예시)
│   ├── base.py                    ← Strategy ABC (AI도 이걸 상속)
│   └── registry.py                ← 전략 등록/조회
```

---

## 안전장치

### 코드 보안

AI가 생성한 코드에서 다음 패턴이 감지되면 **즉시 거부**된다:

| 차단 패턴 | 이유 |
|-----------|------|
| `import os` | 시스템 접근 차단 |
| `import subprocess` | 프로세스 실행 차단 |
| `import socket` | 네트워크 접근 차단 |
| `open(` | 파일 I/O 차단 |
| `exec(`, `eval(` | 임의 코드 실행 차단 |
| `__import__` | 동적 임포트 차단 |
| `import requests`, `import httpx` | 외부 HTTP 차단 |

### 허용되는 임포트

```python
import numpy as np
import pandas as pd
import ta                          # 기술적 지표 라이브러리
from src.core.models import Signal, SignalType
from src.strategies.base import ExecutionMode, Strategy
from src.strategies.registry import register
```

### 기타 안전장치

- **이름 규칙**: 전략 이름은 반드시 `ai_` 접두사 (기존 전략과 충돌 방지)
- **백테스트 필수**: 최소 3건 거래가 발생해야 유효
- **자동 정리**: AI 전략 파일은 최대 5개만 유지, 오래된 것부터 삭제
- **API 키 없으면 비활성화**: LLM 키가 하나도 없으면 자동으로 꺼짐
- **테스트넷 기본**: 거래소는 항상 testnet 모드가 기본

---

## 로그 예시

### 성과 양호 시

```
llm.provider_created  provider=openai/gpt-4o
agent.performance_analyzed  strategy=momentum_flip_scalper trades=25 win_rate=52.0% pnl=+12.50 is_poor=False
agent.report  report="... 현재 전략 유지 ..."
```

### 성과 부진 → 1회차에 성공

```
agent.generating_new_strategy  reason=low_win_rate(33.3%)
agent.strategy_switched  old=momentum_flip_scalper new=ai_adaptive_rsi_v1 attempts=1
agent.hot_swap  symbol=BTCUSDT old=momentum_flip_scalper new=ai_adaptive_rsi_v1
```

### 1회차 실패 → 2회차에 자동 수정 성공

```
agent.generating_new_strategy  reason=low_win_rate(33.3%)
agent.validation_failed  attempt=1 error="_validate_syntax: SyntaxError at line 42"
agent.strategy_switched  old=momentum_flip_scalper new=ai_rsi_bounce_v1 attempts=2
agent.hot_swap  symbol=BTCUSDT old=momentum_flip_scalper new=ai_rsi_bounce_v1
```

### 3회 모두 실패

```
agent.validation_failed  attempt=1 error="_validate_syntax: SyntaxError ..."
agent.load_failed  attempt=2 error="ImportError: cannot import ..."
agent.backtest_failed  attempt=3 result={error: "Too few trades"}
agent.all_attempts_failed  attempts=3
```

---

## FAQ

### Q: AI Agent를 끄고 싶으면?

`.env`에서 LLM API 키를 모두 제거하거나, 대시보드에서 `ai_agent_enabled`를 `false`로 설정한다.

### Q: 프로바이더를 바꾸고 싶으면?

`.env`에서 원하는 프로바이더의 API 키만 남기면 된다. 여러 키가 있으면 우선순위: Anthropic > OpenAI > Gemini.

### Q: AI가 만든 전략이 안 좋으면?

AI Agent가 다음 주기에 다시 성과를 분석하고, 부진하면 또 새 전략을 만든다. 자동으로 계속 개선을 시도한다.

### Q: 이전 전략으로 돌아가고 싶으면?

대시보드에서 `strategy` 설정을 원하는 전략 이름(예: `smart_momentum_scalper`)으로 변경하면 된다.

### Q: API 비용은?

약 1시간에 1회 호출. 성과 양호 시 ~500토큰(평가만), 전략 생성 시 ~4000토큰.

| 프로바이더 | 월 예상 비용 |
|-----------|-------------|
| Anthropic Claude | ~$2~5 |
| OpenAI GPT-4o | ~$1~3 |
| Google Gemini Flash | ~$0.5~1 (가장 저렴) |

### Q: AI가 만든 전략 코드를 보고 싶으면?

`src/strategies/ai_generated/` 폴더에 `.py` 파일로 저장되어 있다. 일반 Python 파일이니 에디터로 열어볼 수 있다.

### Q: 동시에 여러 심볼을 돌리면?

모든 심볼이 같은 전략을 공유한다. AI Agent가 전략을 교체하면 모든 심볼에 동시 적용된다.

### Q: "코드를 짜고 빌딩한다"는 게 정확히 뭔가?

1. LLM API에 텍스트 프롬프트를 보냄 ("이런 성과가 나오는데 새 전략 짜줘")
2. LLM이 Python 코드를 텍스트로 응답
3. 그 텍스트에서 ```python 블록을 추출
4. 구문/보안/구조 검증 후 `.py` 파일로 저장
5. `importlib`로 런타임에 동적 로드 (프로세스 재시작 불필요)
6. `@register` 데코레이터가 자동으로 전략 레지스트리에 등록

별도의 컴파일이나 빌드 과정은 없다. Python이라 파일 저장 → import하면 바로 실행 가능하다.
