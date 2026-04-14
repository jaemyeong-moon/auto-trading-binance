# Auto-Trader 매매전략 고도화 계획

작성일: 2026-03-31
갱신일: 2026-04-13 (Phase 12-16 추가 — TDD 기반 안정화/리스크/최적화/v13/AI Agent)

**목표**: 실거래 + 가상매매 데이터를 분석하여 전략별 약점을 정량화하고, 데이터 기반으로 진입/청산 로직을 개선하여 승률·수익성을 높인다.

---

## Phase 1: 진단 — 왜 대부분의 전략이 거래를 안 하는가?

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 1.1 | 페이퍼 트레이더 진입 조건 디버깅 — v2/v3/v9/v10이 거래 0건인 원인 분석 | 각 전략의 미거래 원인 문서화 | - | cc:완了 |
| 1.2 | v1(momentum_flip) 손실 패턴 분석 — 28건 거래의 SL/TP 비율, 시간대별 승패 | 분석 리포트 출력 | - | cc:완了 |
| 1.3 | 페이퍼 트레이더 캔들 공급 확인 — evaluate()에 전달되는 캔들 수/타임프레임 점검 | 각 전략에 필요한 캔들이 충분히 전달됨을 확인 | - | cc:완了 |

## Phase 2: 수리 — 모든 전략이 정상 가상매매하도록 수정

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 2.1 | 페이퍼 트레이더 캔들 공급 수정 (1m:300, 15m:500) | v9/v10 insufficient_htf 해소 | 1.1, 1.3 | cc:완了 |
| 2.2 | v1 SL/TP 비율 조정 — SL 체결이 TP의 1.8배 | → Phase 5.7로 이동 | 1.2 | cc:완了 |

## Phase 3: 성과 수집 — 충분한 데이터 확보 (1~3일 운영)

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 3.1 | 수정된 전략 서버 배포 (git push → pull → restart) | 서버에서 모든 전략이 가상매매 실행 확인 | 2.1, 2.2 | cc:완了 |
| 3.2 | 성과 모니터링 스크립트 작성 (scripts/paper_report.py) | 원커맨드로 전략 성과 테이블 출력 | 3.1 | cc:완了 |

## Phase 4: 분석 & 신규 전략 생성 — 데이터 기반 전략 개발

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 4.1 | 전략별 성과 비교 분석 — v1 29건 상세 분석 (시간대/방향/연속손실) | 비교 리포트 생성 | 3.2 | cc:완了 |
| 4.2 | 분석 결과: 새벽 0-9시 승률 10%, LONG 약세, 연속손실 4회 빈번 | 개선 방향 도출 | 4.1 | cc:완了 |
| 4.3 | v11 Data-Driven Scalper 작성 — 데이터 기반 8점 진입 시스템 | 전략 등록 완료 | 4.2 | cc:완了 |
| 4.4 | v11 서버 배포 완료 (604cca9) | 6개 전략 가상매매 중 | 4.3 | cc:완了 |

## Phase 5: 반복 최적화 1차 — v12 멀티 패턴 엔진

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 5.1 | v12 Pattern Scalper 최초 구현 + 15분봉 전환 | 전략 등록 및 서버 가동 | 4.4 | cc:완了 |
| 5.2 | 15분봉 전환 후 서버 안정화 — API 과부하/CPU 100% 해결 | 서버 CPU 정상, 틱 정상 순환 | 5.1 | cc:완了 |
| 5.3 | 대시보드 TradingView 차트 임베드 + 거래 마커 | 바이낸스 차트 + 거래 내역 표시 | - | cc:완了 |
| 5.4 | v12 멀티 패턴 엔진 업그레이드 — 7개 패턴 + 거래량 분석 | patterns.py 분리, 7패턴 스캔 동작 | 5.2 | cc:완了 |
| 5.5 | 바이낸스 실제 가격 데이터로 v12 백테스트 — 미진입 원인 분석 | 패턴별 미진입 사유 통계 + 놓친 기회 리스트 산출 | 5.4 | cc:완了 |
| 5.6 | v12.1 리팩토링 — 모멘텀 확인 + SL/TP 청산 + 트레일링 + 최대보유 | 41일 백테스트 승률 50%, PF 1.06, ROI +0.6% | 5.5 | cc:완了 |
| 5.7 | 개선된 v12 서버 배포 + 가상매매 모니터링 | 서버 가동 + 24h 내 10건+ 거래 확인 | 5.6 | cc:완了 [f716547] |
| 5.8 | v1 SL/TP 비율 조정 — SL 체결이 TP의 1.8배 문제 | 데이터 기반 SL/TP 재조정 | 5.6 | cc:완了 [cd7da33] |

## Phase 5.9: 포지션 가격 경로 차트

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 5.9.1 | TradeRecord에 sl_price, tp_price 컬럼 추가 + 마이그레이션 | 실거래 기록에 SL/TP 가격 저장, 기존 DB 호환 | - | cc:완了 |
| 5.9.2 | 엔진에서 실거래 진입/청산 시 sl_price, tp_price 기록 | _open_position, _close_current에서 TradeRecord에 SL/TP 저장 | 5.9.1 | cc:완了 |
| 5.9.3 | 거래 상세 API — 포지션 구간 캔들 + SL/TP 데이터 반환 | /api/trade/{id}/chart 엔드포인트, opened_at~closed_at 구간 바이낸스 캔들 + SL/TP 반환 | 5.9.2 | cc:완了 |
| 5.9.4 | 모바일 대시보드 포지션 차트 UI — 캔들 + SL/TP 밴드 + 진입/청산 마커 | 거래 내역에서 탭하면 entry→exit 캔들차트 + SL/TP 수평선 표시 | 5.9.3 | cc:완了 [8a61558] |
| 5.9.5 | 포지션 궤적 추적 + SL/TP 프로그레스바 — 매 틱 가격 위치 기록, 모바일 궤적 조회 API | PositionTrail DB 모델 + record/link 헬퍼 + /api/trade/{id}/trail 엔드포인트 + 프로그레스바 UI | 5.9.4 | cc:완了 [92a474f] |

---

## Phase 6: 데이터 수집 & 심층 분석 — 실거래/가상매매 성과 정량화

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 6.1 | 실거래 성과 종합 리포트 스크립트 — trades_real.db 전체 분석 | 전략별 승률/PF/MDD/평균보유시간/시간대별 승패 출력 (scripts/real_report.py) | - | cc:완了 |
| 6.2 | 가상매매 전 전략 성과 비교 분석 — 7개 전략 병렬 성과 랭킹 | 전략별 ROI/승률/거래수/PF 비교 테이블 + 최약 전략 식별 | 6.1 | cc:완了 |
| 6.3 | 손실 패턴 심층 분석 — 연속손실/시간대/방향/청산사유별 분류 | 전략별 손실 클러스터링 리포트 (어떤 조건에서 지는지 정량화) | 6.2 | cc:완了 |
| 6.4 | SL/TP 적중률 분석 — 각 전략의 SL 체결 vs TP 체결 비율, 트레일링 효과 | SL/TP/trailing/signal/flip 별 청산 비율 + 평균 수익/손실 | 6.2 | cc:완了 |

## Phase 7: 전략 개선 — 분석 결과 기반 로직 수정

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 7.1 | 시간대 필터 고도화 — 실거래 데이터 기반 수익/손실 시간대 재계산 | 각 전략에 데이터 기반 시간대 필터 적용 (하드코딩 → DB 기반) | 6.3 | cc:완了 |
| 7.2 | SL/TP 배수 전략별 최적화 — auto-optimizer 결과 vs 실제 성과 비교 | optimizer 추천값 vs 실제 체결 분석 → 배수 조정 | 6.4 | cc:완了 |
| 7.3 | 진입 신호 강화 — 승률 높은 조건 조합 발굴 | 분석에서 도출된 고승률 조건을 기존 전략에 반영 | 6.3, 6.4 | cc:완了 |
| 7.4 | 개선 전략 백테스트 검증 — 변경 전/후 성과 비교 | 개선 전략의 백테스트 승률·PF이 기존 대비 향상 확인 | 7.1, 7.2, 7.3 | cc:완了 |

## Phase 8: 배포 & 모니터링 — 개선 전략 실전 투입

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 8.1 | 개선 전략 서버 배포 + 실거래 전환 | git push → pull → restart, 실거래 정상 가동 확인 | 7.4 | cc:완了 [9089c7f] |
| 8.2 | 1주일 실거래 성과 추적 — Phase 6 리포트 재실행으로 개선 효과 검증 | 개선 전/후 승률·PF·MDD 비교표 | 8.1 | cc:완了 [4f70959] |

## Phase 9: 전략 판단 모니터링 — 봇 진입/관망 사유 실시간 확인

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 9.1 | 매 틱 전략 평가 결과(Signal + metadata)를 DB에 저장 — 최근 N건 링버퍼 | signal_logs 테이블 생성, 심볼/전략/신호/confidence/metadata JSON 기록, 500건 초과 시 자동 정리 | - | cc:완了 [22df449] |
| 9.2 | `/api/signals` 엔드포인트 — 최근 전략 판단 내역 조회 | 심볼별 최근 20건 신호 + metadata JSON 반환 | 9.1 | cc:완了 [22df449] |
| 9.3 | 모바일 홈탭에 "전략 판단 로그" 카드 — 실시간 진입/관망 사유 표시 | 신호타입, confidence, 주요 지표(RSI/ATR/패턴/스코어) 시각화 | 9.2 | cc:완了 [22df449] |

## Phase 10: 모바일 UI 개선

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 10.1 | 홈탭 레이아웃 리팩토링 — 계좌 요약 + 포지션 + 판단로그 배치 최적화 | 정보 밀도 향상, 주요 정보 스크롤 없이 확인 가능 | 9.3 | cc:완了 [22df449] |
| 10.2 | 거래 내역 탭 개선 — 필터(승/패/심볼) + 성과 요약 카드 | 승률/PF/총손익 한눈에 확인, 필터 동작 | - | cc:완了 [22df449] |
| 10.3 | 설정 탭 UX 개선 — ATR 배수/트레일링 파라미터 노출 + 그룹화 | 현재 숨겨진 설정(auto_sl_mult 등)도 UI에서 조정 가능 | - | cc:완了 [22df449] |

## Phase 11: 봇 동적 재실행 — 설정 변경 시 자동 적용

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 11.1 | 엔진 설정 핫 리로드 — 매 틱 DB 설정 변경 감지 시 자동 반영 | 전략 외 설정(레버리지/SL/TP/틱간격) 재시작 없이 즉시 적용 | - | cc:완了 [22df449] |
| 11.2 | 전략 변경 시 봇 자동 재시작 — 설정 저장 시 graceful restart | 전략 변경 후 봇이 자동으로 새 전략으로 전환, 포지션 안전 처리 | 11.1 | cc:완了 [22df449] |
| 11.3 | 재시작 상태 알림 — 모바일에서 봇 재시작 진행 상태 표시 | "재시작 중..." → "정상 가동" 상태 전환 UI 표시 | 11.2 | cc:완了 [22df449] |

---

## Phase 12: 테스트 인프라 & 회귀 테스트 — TDD 기반 마련

**목적**: 최근 3대 실거래 버그(bac0bed: max_hold/레버리지/SHORT 편향) 재발 방지 + 핵심 모듈 커버리지 확보. 모든 후속 Phase의 기반.

| Task | 내용 | DoD | Depends | Status |
|------|------|-----|---------|--------|
| 12.0 | **베이스라인 수리** — 사전 깨진 테스트 3건 + 환경 정비 | (a) `test_aggressive_scalper.py` v9 재작성 or 제거, (b) adaptive/smart scalper cooldown→blocked_hour 반영 업데이트, (c) pytest-asyncio 설치 안내 or requirements 보강, (d) `pytest -q` 녹색 확인 | - | cc:완了 |
| 12.1 | `tests/conftest.py` 작성 — 공통 fixture (mock BinanceClient, 샘플 candles, TradingConfig, 임시 DB) | `pytest --collect-only` 에 fixture 인식, 기존 테스트 전부 통과 유지 | 12.0 | cc:완了 |
| 12.2 | `tests/test_futures_engine.py` — 엔진 tick 루프 단위 테스트 (mock client) | tick 1회당 candles 조회→evaluate→signal→order 경로 커버, 정상/에러 케이스 3개 이상 | 12.0, 12.1 | cc:완了 |
| 12.3 | **회귀 테스트 — MAX_HOLD_HOURS**: 전략별 max_hold 초과 시 강제 청산 호출 확인 | 전략 속성값(v1/v12 등) 기준으로 정확히 청산 트리거됨을 검증하는 테스트 pass | 12.2 | cc:완了 |
| 12.4 | **회귀 테스트 — 레버리지 DB/전략 동기화**: 엔진이 `getattr(strategy, "LEVERAGE", 5)` 올바르게 읽고 바이낸스 호출에 반영 | mock client 에 setLeverage 호출 기록 확인, 전략별 값이 넘어감 | 12.2 | cc:완了 |
| 12.5 | **회귀 테스트 — SHORT 편향 제거**: 동일 시나리오에서 LONG/SHORT 진입 비율 편향 없음 | 합성 캔들에서 대칭적 조건 → LONG/SHORT 신호 수 동등 (±10%) | 12.2 | cc:완了 |
| 12.6 | `tests/test_pattern_scalper.py` — v12 패턴별 진입/청산/트레일링/V12State 동작 | 7패턴 중 핵심 3패턴 + 트레일링 업데이트 + partial_tp 동작 검증 | 12.1 | cc:완了 |
| 12.7 | `tests/test_binance_client.py` — mock aiohttp 으로 client 메서드 계약 테스트 | get_candles/place_order/get_balance 응답 파싱 정상 + 에러 핸들링 | 12.1 | cc:완了 |
| 12.8 | `tests/test_strategy_agent.py` — LLM 호출 mock, 생성→검증→등록 파이프라인 | _generate_new_strategy, _validate_and_register 각각 mock 으로 pass/fail 경로 커버 | 12.1 | cc:완了 |
| 12.9 | 커버리지 리포트 집계 — `pytest --cov=src` 실행, 주요 모듈 50%+ 목표 | engine/strategies/exchange 50%+, 커버리지 HTML 리포트 생성 | 12.2, 12.6, 12.7, 12.8 | cc:완了 |
| 12.10 | CI 훅 또는 로컬 pre-push 스크립트 — 테스트 실패 시 push 차단 | `scripts/test_gate.sh` (또는 pre-commit hook) 동작 확인 | 12.9 | cc:TODO |

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
| 15.5 | **v13 전략 초안 구현** (`src/strategies/orderflow_v13.py`) — 멀티 TF 추세 일치 + 오더북 불균형 + 펀딩비 극단치 진입 | 전략 등록, 단위 테스트 pass, 백테스트 30일 실행 성공 | 15.2, 15.3, 15.4, 12.1 | cc:TODO |
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
| 16.5 | **AI Agent 실거래 가동 모드** — 현재 페이퍼 모드만 → 리스크매니저 승인 후 실거래 반영 | 16.2+16.3+13.1 통과한 전략만 실거래 반영, 핫스왑 로그 DB 기록 | 16.2, 16.3, 13.1 | cc:TODO |
| 16.6 | 모바일 대시보드 — AI 에이전트 활동 로그 카드 (생성 시도/통과/폐기/스왑) | `/api/agent/activity` 엔드포인트 + UI 카드, 최근 20건 표시 | 16.5 | cc:TODO |
| 16.7 | AI Agent 2주 실전 가동 및 효과 측정 | 생성 전략 개수 / 통과율 / 핫스왑 수익 기여 리포트 | 16.5, 16.6 | cc:TODO |

---

## Completed

_완료된 태스크가 여기에 기록됩니다._

## Archive

_오래된 완료 태스크는 여기로 이동합니다._
