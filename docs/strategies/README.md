# 전략 이력

매매 전략의 변경 이력을 관리합니다.
새 전략 적용 시 `vN_전략이름.md` 형식으로 문서를 추가하고, 아래 표를 업데이트합니다.

## 전략 목록

| 버전 | 이름 | 적용일 | 상태 | 요약 |
|------|------|--------|------|------|
| v1 | [Momentum Flip Scalper](v1_momentum_flip_scalper.md) | 2026-03-18 | 운용중 | EMA(3/8) 크로스 + 거래량 필터 + 역추세 자동전환 |
| v2 | [Adaptive Scalper](v2_adaptive_scalper.md) | 2026-03-19 | 운용 가능 | 시장 상태 판단 + 점수 진입 + 횡보 금지 + 부분익절/트레일링 |
| v3 | [Smart Momentum Scalper](v3_smart_momentum_scalper.md) | 2026-03-19 | 운용 가능 | ATR 동적 TP/SL + 실행비용 차감 + RR 2:1 보장 + 과다매매 방지 |
| v4 | [Aggressive Momentum Rider](v4_aggressive_momentum_rider.md) | 2026-03-19 | 운용 가능 | 모멘텀 폭발 즉시 진입 + 타이트 TP/SL + 횡보 매매 + 연패 방향반전 |

## 문서 작성 규칙

- 파일명: `vN_전략이름.md`
- 필수 항목: 한줄요약, 진입/청산 조건, 파라미터, 알려진 한계, 향후 최적화
- 이전 전략은 상태를 `폐기` 또는 `대체됨`으로 변경하고, 폐기 사유를 기록
