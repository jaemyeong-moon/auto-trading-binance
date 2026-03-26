#!/usr/bin/env python3
"""CLI로 웹훅 설정을 조회/변경하는 스크립트.

Usage:
    python scripts/set-webhook.py                          # 현재 설정 조회
    python scripts/set-webhook.py URL                      # 웹훅 URL 설정
    python scripts/set-webhook.py URL --no-open            # 진입 알림 끄기
    python scripts/set-webhook.py --off                    # 웹훅 비활성화
    python scripts/set-webhook.py --test                   # 테스트 메시지 전송
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core import database as db


def show_current():
    url = db.get_setting("webhook_url")
    on_open = db.get_setting("webhook_on_open")
    on_close = db.get_setting("webhook_on_close")
    on_tp_sl = db.get_setting("webhook_on_tp_sl")

    if not url:
        print("웹훅: 비활성화 (URL 미설정)")
    else:
        print(f"웹훅 URL : {url}")
        print(f"포지션 진입 : {'ON' if on_open == 'true' else 'OFF'}")
        print(f"포지션 청산 : {'ON' if on_close == 'true' else 'OFF'}")
        print(f"익절/손절  : {'ON' if on_tp_sl == 'true' else 'OFF'}")


async def test_webhook():
    from src.notifications.webhook import send_raw
    url = db.get_setting("webhook_url")
    if not url:
        print("웹훅 URL이 설정되어 있지 않습니다.")
        return
    print(f"테스트 전송 중... → {url}")
    await send_raw("🔔 Auto-Trader 웹훅 테스트 메시지입니다.")
    print("전송 완료.")


def main():
    parser = argparse.ArgumentParser(description="웹훅 설정 관리")
    parser.add_argument("url", nargs="?", help="웹훅 URL (Discord, Slack, 일반)")
    parser.add_argument("--off", action="store_true", help="웹훅 비활성화")
    parser.add_argument("--no-open", action="store_true", help="포지션 진입 알림 끄기")
    parser.add_argument("--no-close", action="store_true", help="포지션 청산 알림 끄기")
    parser.add_argument("--no-tp-sl", action="store_true", help="익절/손절 알림 끄기")
    parser.add_argument("--test", action="store_true", help="테스트 메시지 전송")
    args = parser.parse_args()

    db.init_db()

    # 테스트 모드
    if args.test:
        asyncio.run(test_webhook())
        return

    # 비활성화
    if args.off:
        db.set_setting("webhook_url", "")
        print("웹훅이 비활성화되었습니다.")
        return

    # URL 설정
    if args.url:
        db.set_setting("webhook_url", args.url)
        db.set_setting("webhook_on_open", "false" if args.no_open else "true")
        db.set_setting("webhook_on_close", "false" if args.no_close else "true")
        db.set_setting("webhook_on_tp_sl", "false" if args.no_tp_sl else "true")
        print("웹훅 설정 저장 완료:")
        show_current()
        return

    # 인자 없으면 현재 설정 조회
    show_current()


if __name__ == "__main__":
    main()
