"""페이퍼 트레이딩 성과 리포트 — SSH에서 실행용.

Usage: python scripts/paper_report.py
"""
import sys
sys.path.insert(0, ".")

from src.core.database import SessionLocal, PaperBalance, PaperTrade


def report():
    with SessionLocal() as s:
        print("=" * 70)
        print("  전략별 페이퍼 트레이딩 성과 리포트")
        print("=" * 70)
        print()
        print(f"{'전략':<30s} {'잔고':>8s} {'ROI':>7s} {'거래':>4s} "
              f"{'승':>3s} {'패':>3s} {'승률':>6s} {'평균수익':>8s} {'평균손실':>8s}")
        print("-" * 70)

        for b in s.query(PaperBalance).order_by(PaperBalance.balance.desc()).all():
            wr = round(b.wins / b.total_trades * 100, 1) if b.total_trades > 0 else 0
            roi = round((b.balance - b.initial_balance) / b.initial_balance * 100, 2)

            trades = s.query(PaperTrade).filter_by(strategy=b.strategy).all()
            wins = [t for t in trades if t.net_pnl > 0]
            losses = [t for t in trades if t.net_pnl <= 0]
            avg_win = round(sum(t.net_pnl for t in wins) / len(wins), 2) if wins else 0
            avg_loss = round(sum(t.net_pnl for t in losses) / len(losses), 2) if losses else 0

            print(f"{b.strategy:<30s} {b.balance:>8.2f} {roi:>6.1f}% {b.total_trades:>4d} "
                  f"{b.wins:>3d} {b.losses:>3d} {wr:>5.1f}% {avg_win:>+8.2f} {avg_loss:>+8.2f}")

        print()
        print("최근 10건 거래:")
        print("-" * 70)
        for t in s.query(PaperTrade).order_by(PaperTrade.closed_at.desc()).limit(10).all():
            reason = t.reason or "-"
            print(f"{t.strategy:<30s} {t.symbol:<8s} {t.side:<5s} "
                  f"pnl={t.net_pnl:>+8.4f} {reason:<3s} {t.closed_at}")


if __name__ == "__main__":
    report()
