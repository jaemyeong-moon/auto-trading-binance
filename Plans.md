# Auto-Trader 매매전략 고도화 계획

작성일: 2026-03-31
갱신일: 2026-04-16 (Phase 19 추가 — 실거래 투입 준비/봇 복구/경량화/신규 전략)

**목표**: 실거래 + 가상매매 데이터를 분석하여 전략별 약점을 정량화하고, 데이터 기반으로 진입/청산 로직을 개선하여 승률·수익성을 높인다.

> Phase 1~12: 전부 완了 → Archive 섹션 참조

## Phase 13: 리스크 관리 고도화 — 손실 방어 시스템

**목적**: 포지션 사이징이 고정 20%뿐, max_open_positions 미적용, 상관도 필터 없음 → 실거래 리스크 축소.

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 13.1 | `src/core/risk_manager.py` 신설 + 단위 테스트 먼저 (`tests/test_risk_manager.py`) — 포지션 사이징/동시포지션/DD 차단 인터페이스 | 테스트 pass, RiskManager 클래스 API 확정 (can_open, position_size, daily_dd_ok) | 12.1 | cc:완了 |
| 13.2 | **max_open_positions 실제 적용** — 엔진이 현재 포지션 수 확인 후 진입 차단 | TradingConfig.max_open_positions=3 초과 시 로그 + 거부, 테스트로 검증 | 13.1 | cc:완了 |
| 13.3 | **일일 최대 DD 자동 차단** — 일일 손실이 `max_daily_loss_pct`(신규 설정) 초과 시 당일 신규 진입 정지 | DB에서 당일 손익 집계 → 한도 초과 시 RiskManager.can_open=False, 테스트 pass | 13.1 | cc:완了 |
| 13.4 | **변동성(ATR) 기반 동적 포지션 사이징** — 고ATR 구간에서 사이즈 축소 | 기존 POSITION_SIZE_PCT에 ATR 계수 곱, 백테스트에서 최대 드로다운 감소 검증 | 13.1 | cc:완了 |
| 13.5 | **심볼 간 상관도 필터** — 최근 30일 수익률 상관계수 > 0.8 쌍은 동시 포지션 차단 | `scripts/correlation.py` 산출 + RiskManager에 반영, 테스트 pass | 13.1 | cc:완了 |
| 13.6 | **Kelly Criterion 기반 사이즈 제안** (선택 적용) — 전략별 승률/손익비로 최적 비율 계산, 상한 POSITION_SIZE_PCT | 전략별 Kelly 계산 로직 + 안전계수 0.25 적용, 단위 테스트 | 13.1, 7.2 | cc:완了 |
| 13.7 | 모바일 대시보드에 리스크 상태 카드 — 당일 DD / 동시 포지션 / Kelly 추천 사이즈 표시 | 홈탭에 카드 추가, API 엔드포인트 `/api/risk/status` 추가 | 13.2, 13.3, 13.6 | cc:완了 |
| 13.8 | 1주일 실거래에서 리스크 방어 작동 로그 수집 및 분석 | 차단 이벤트 집계 + 차단이 없었다면 발생했을 손실 추정 리포트 | 13.2, 13.3, 13.5 | cc:TODO |

## Phase 14: v12 실거래 2차 최적화 — 데이터 기반 재튜닝

**목적**: Phase 8 이후 누적 실거래 데이터로 v12 패턴/SL-TP/시간 필터를 재조정.

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 14.1 | 실거래 2주+ 데이터 스냅샷 수집 — trades_real.db 백업 + 가상매매 병렬 데이터 확보 | `data/snapshots/YYYYMMDD_trades.db` 저장, 거래 수 100건+ 확인 | 13.8 | cc:TODO |
| 14.2 | **v12 패턴별 실거래 승률 재산출** — 7개 패턴 각각의 거래수/승률/PF 계산 (패턴 로그 매칭) | 패턴별 리포트 출력 (scripts/v12_pattern_report.py), 최약 패턴 2개 식별 | 14.1 | cc:TODO |
| 14.3 | **최약 패턴 비활성화 or 진입 조건 강화** — 승률 40% 미만 패턴 제거/보강 | pattern_scalper.py 수정, 단위 테스트 업데이트, 백테스트 개선 확인 | 14.2, 12.6 | cc:TODO |
| 14.4 | **SL/TP 배수 재튜닝** — 실거래 체결 데이터로 auto_optimizer 재실행 + 수동 검증 | SL_ATR_MULT/TP_ATR_MULT 업데이트 근거 리포트, 백테스트 PF 개선 | 14.1 | cc:TODO |
| 14.5 | **트레일링 발동 조건 최적화** — highest/lowest 갱신 후 트레일링 거리 재조정 | 백테스트에서 트레일링 청산 비율 상승 + 수익 중앙값 개선 | 14.4, 12.6 | cc:TODO |
| 14.6 | v12.2 서버 배포 + 1주 실거래 비교 추적 | 배포 커밋 기록 + 개선 전/후 승률·PF·MDD 비교표 | 14.3, 14.4, 14.5, 13.2 | cc:TODO |

## Phase 15: 신규 전략 v13 — 멀티 TF × 오더북 × 펀딩비

**목적**: 가격 캔들만 쓰는 기존 전략과 다른 시그널 소스로 포트폴리오 다변화.

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 15.1 | **BinanceClient 확장** — `get_order_book(symbol, depth)`, `get_funding_rate(symbol)`, `get_open_interest(symbol)` 메서드 추가 + 테스트 먼저 | `tests/test_binance_client.py` 신규 메서드 테스트 pass, 실거래 호출 스모크 OK | 12.7 | cc:완了 |
| 15.2 | **멀티 TF 프레임워크 정비** — 현재 15m+1h 고정을 Strategy 속성으로 선언적 지정 (`TIMEFRAMES = ["5m","15m","1h"]`) | base.py Strategy에 TIMEFRAMES 속성, 엔진이 자동 조회/주입, 기존 전략 마이그레이션 | 12.2 | cc:완了 |
| 15.3 | **오더북 불균형 피처** 계산 유틸 (`src/strategies/features/orderbook.py`) + 단위 테스트 | bid/ask 깊이 비율, 스프레드, 대규모 주문벽 탐지 함수 + 테스트 pass | 15.1 | cc:완了 |
| 15.4 | **펀딩비 & OI 피처** — funding rate 추세 + OI 변화율 유틸 + 단위 테스트 | `src/strategies/features/derivatives.py` + 테스트 pass | 15.1 | cc:완了 |
| 15.5 | **v13 전략 초안 구현** (`src/strategies/orderflow_v13.py`) — 멀티 TF 추세 일치 + 오더북 불균형 + 펀딩비 극단치 진입 | 전략 등록, 단위 테스트 pass, 백테스트 30일 실행 성공 | 15.2, 15.3, 15.4, 12.1 | cc:완了 |
| 15.6 | v13 백테스트 + 파라미터 스윕 | Sharpe/PF/승률 리포트, 기존 전략 대비 상관도 < 0.5 검증 | 15.5 | cc:TODO |
| 15.7 | v13 가상매매 배포 + 1주 모니터링 | 서버 가동 + 거래 10건+ 확인 후 실거래 전환 판단 | 15.6, 13.2 | cc:TODO |

## Phase 16: AI Strategy Agent 본격 가동

**목적**: `src/core/strategy_agent.py` 뼈대는 이미 존재 → 안전성 보강 후 실전 투입.

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 16.1 | GeminiProvider 완성 + LLM provider 간 인터페이스 통합 테스트 | 3개 provider 모두 동일 chat_messages() 계약 pass, `tests/test_llm_provider.py` | 12.8 | cc:완了 |
| 16.2 | **AI 생성 전략 보안 샌드박스 강화** — AST 화이트리스트 (import 제한, 파일 IO/네트워크 금지), 실행 시간 제한 | 악의적 코드 10종 샘플에 대해 검증 실패 + 정상 전략 통과 테스트 pass | 12.8 | cc:완了 |
| 16.3 | **AI 생성 전략 백테스트 게이트** — 자동 배포 전 최소 성과 기준(승률 45%+, PF 1.1+) 충족 필수 | 기준 미달 시 자동 폐기 + 로그, 기준 통과 케이스/실패 케이스 테스트 | 16.2 | cc:완了 |
| 16.4 | **성과 부진 판단 로직 검증** — `analyze_performance()` 의 트리거 조건(승률/PF/거래수) 실데이터로 회귀 | 실제 실거래 2주 데이터 넣었을 때 합리적 트리거 여부 리포트 | 14.1, 16.1 | cc:TODO |
| 16.5 | **AI Agent 실거래 가동 모드** — 현재 페이퍼 모드만 → 리스크매니저 승인 후 실거래 반영 | 16.2+16.3+13.1 통과한 전략만 실거래 반영, 핫스왑 로그 DB 기록 | 16.2, 16.3, 13.1 | cc:완了 |
| 16.6 | 모바일 대시보드 — AI 에이전트 활동 로그 카드 (생성 시도/통과/폐기/스왑) | `/api/agent/activity` 엔드포인트 + UI 카드, 최근 20건 표시 | 16.5 | cc:TODO |
| 16.7 | AI Agent 2주 실전 가동 및 효과 측정 | 생성 전략 개수 / 통과율 / 핫스왑 수익 기여 리포트 | 16.5, 16.6 | cc:TODO |

## Phase 17: 안정성 개선 — 재시작 내성

**목적**: 봇 재시작 시 전략 상태가 초기화되어 쿨다운/거래제한이 무시되는 문제 해결.

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 17.1 | **전략 상태 DB 영속화** — 매 틱 V12State/V9State 등을 DB에 저장, 재시작 시 복원 | 재시작 후 cooldown_remaining/consecutive_losses/trades_this_hour 유지됨을 테스트로 검증 | 12.2 | cc:완了 |
| 17.2 | **재시작 시 기존 포지션 동기화** — 거래소 포지션 조회 후 전략 state에 반영 | 재시작 후 기존 포지션의 entry_price/sl/tp/ticks_in_position이 올바르게 복원됨 | 17.1 | cc:완了 |
| 17.3 | **재시작 쿨다운** — 재시작 직후 N틱 동안 신규 진입 억제 (관찰 모드) | 재시작 후 첫 5분간 진입 없음, 이후 정상 거래 시작 | 17.1 | cc:완了 |

## Phase 19: 실거래 투입 준비 — 봇 복구 + 경량화 + 신규 전략

**목적**: 서버 봇 완전 미작동(confidence=0.43 고정) 긴급 수리, PC 웹 제거/모바일 집중, 1코어 1GB 최적화, 신규 전략 2종 투입.

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 19.1 | **엔진 오더북/펀딩비/OI 데이터 주입** — _fetch_candles + _paper_loop에 candles.attrs 주입 | v13 confidence > 0.6 로그 확인, 테스트 전체 통과 | 15.1 | cc:완了 |
| 19.2 | **서버 DB trading_paused 리셋** — false 설정 + paper_selector_min_trades=5 | 서버 DB 설정 확인 | 19.1 | cc:완了 |
| 19.3 | **PC 웹 대시보드 제거** — app.py 삭제 + streamlit/plotly 의존성 제거 | 모바일만 남음, pyproject.toml 정리 | - | cc:완了 |
| 19.4 | **모바일 수동 전략 선택 API** — POST /api/strategy/manual, /api/strategy/auto | 엔드포인트 동작, 전략 전환 확인 | - | cc:완了 |
| 19.5 | **VWAP Reversion v14** — 거래량 가중 평균가 회귀 전략 | 레지스트리 등록, 구문 검증 통과 | - | cc:완了 |
| 19.6 | **Volatility Breakout v15** — 래리 윌리엄스 변동성 돌파 전략 | 레지스트리 등록, 구문 검증 통과 | - | cc:완了 |
| 19.7 | 서버 배포 + 로그 검증 | git push → pull → restart, confidence > 0.6 시그널 확인 | 19.1~19.6 | cc:TODO |
| 19.8 | 1주 가상매매 데이터 수집 후 실거래 전환 판단 | paper_trades 50건+ 수집, 전략별 승률 비교 | 19.7 | cc:TODO |

---

## Completed

_완료된 태스크가 여기에 기록됩니다._

## Archive

_오래된 완료 태스크는 여기로 이동합니다._
