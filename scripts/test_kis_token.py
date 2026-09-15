"""KIS 토큰 발급이 동시 요청에서 한 번만 일어나는지 확인한다.

네트워크 없이 발급 함수를 세는 스텁으로 바꿔 검사한다.

    uv run python scripts/test_kis_token.py

배경: 토큰이 없는 상태에서 동시 요청이 들어오면 저마다 발급을 시도했다.
KIS 는 토큰 발급 자체에 빈도 제한이 있어 그 요청들이 403 으로 떨어진다
(2026-09-15 실측: 동시 5건으로 채팅을 돌리자 전부 403 -> 시세 누락).
서버 기동 직후 사용자가 몰리면 그대로 재현되는 경로다.
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.external import kis


async def scenario_cold_start(concurrency: int = 10) -> tuple[int, int]:
    """토큰이 없는 상태에서 동시 호출 -> 발급은 1회여야 한다."""
    kis._token = None
    kis._token_expired_at = None
    calls = 0

    async def fake_issue() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)          # 실제 HTTP 왕복을 흉내
        kis._token = "TOKEN"
        kis._token_expired_at = datetime.now() + timedelta(hours=23)
        return kis._token

    original, kis._issue_access_token = kis._issue_access_token, fake_issue
    try:
        got = await asyncio.gather(*[kis._get_access_token() for _ in range(concurrency)])
    finally:
        kis._issue_access_token = original
    return calls, len(set(got))


async def scenario_expired() -> int:
    """만료된 토큰이면 다시 발급해야 한다."""
    kis._token = "OLD"
    kis._token_expired_at = datetime.now() - timedelta(seconds=1)
    calls = 0

    async def fake_issue() -> str:
        nonlocal calls
        calls += 1
        kis._token = "NEW"
        kis._token_expired_at = datetime.now() + timedelta(hours=23)
        return kis._token

    original, kis._issue_access_token = kis._issue_access_token, fake_issue
    try:
        token = await kis._get_access_token()
    finally:
        kis._issue_access_token = original
    assert token == "NEW", f"만료 토큰을 그대로 반환: {token}"
    return calls


async def scenario_cached() -> int:
    """유효한 토큰이 있으면 발급하지 않아야 한다."""
    kis._token = "CACHED"
    kis._token_expired_at = datetime.now() + timedelta(hours=1)
    calls = 0

    async def fake_issue() -> str:
        nonlocal calls
        calls += 1
        return "NEW"

    original, kis._issue_access_token = kis._issue_access_token, fake_issue
    try:
        token = await kis._get_access_token()
    finally:
        kis._issue_access_token = original
    assert token == "CACHED", f"캐시를 안 씀: {token}"
    return calls


async def main() -> int:
    failed = 0

    calls, distinct = await scenario_cold_start()
    if calls == 1 and distinct == 1:
        print(f"  PASS  동시 10건 -> 발급 {calls}회, 모두 같은 토큰")
    else:
        failed += 1
        print(f"  FAIL  동시 10건 -> 발급 {calls}회(기대 1), 서로 다른 토큰 {distinct}종(기대 1)")

    calls = await scenario_expired()
    if calls == 1:
        print("  PASS  만료된 토큰이면 다시 발급한다")
    else:
        failed += 1
        print(f"  FAIL  만료 재발급 {calls}회(기대 1)")

    calls = await scenario_cached()
    if calls == 0:
        print("  PASS  유효한 토큰이 있으면 발급하지 않는다")
    else:
        failed += 1
        print(f"  FAIL  캐시가 있는데 발급 {calls}회(기대 0)")

    print()
    print("실패 %d건" % failed if failed else "전부 통과")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
