"""/chat/stream SSE 변환 계층 회귀 테스트 (LLM·네트워크 없이).

여기서 보는 것은 `app/routers/chat.py` 의 세 헬퍼다.

    _with_heartbeat   토큰 사이 침묵이 길어지면 None 을 흘린다
    _token_events     토큰 스트림 -> SSE 이벤트 문자열 (빈 토큰 제거 + keepalive)
    _ingest_all_sources  적재를 동시에 돌리고 실패는 건너뛴다

async 제너레이터를 손으로 다루는 코드라 조용히 깨지기 쉽다.
깨지면 채팅이 통째로 멎으므로 별도 테스트를 둔다.
"""
import asyncio
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FAILURES: list[str] = []


def check(name, passed, extra=""):
    print(("PASS  " if passed else "FAIL  ") + name + (f"  {extra}" if extra else ""))
    if not passed:
        FAILURES.append(name)


# 적재 함수는 외부 API를 타므로 import 전에 스텁으로 바꾼다
_ing = types.ModuleType("app.services.rag.ingestion")
CALLS: list[str] = []
FAIL_ON: set[str] = set()


def _stub(kind):
    async def _f(ticker, *a, **kw):
        CALLS.append(f"{kind}:{ticker}")
        await asyncio.sleep(0.05)          # 동시 실행이면 총 0.05초, 순차면 0.15초
        if kind in FAIL_ON:
            raise RuntimeError(f"{kind} 실패")
        return len(kind)
    return _f


_ing.ingest_news = _stub("news")
_ing.ingest_disclosures = _stub("dart")
_ing.ingest_financials = _stub("fin")
sys.modules["app.services.rag.ingestion"] = _ing

from app.routers.chat import (            # noqa: E402
    _ingest_all_sources, _token_events, _with_heartbeat, _SSE_KEEPALIVE,
)


async def _gen(items, delay=0.0):
    for it in items:
        if delay:
            await asyncio.sleep(delay)
        yield it


def run(coro):
    return asyncio.run(coro)


# =========================================================
# 1. 빈 토큰 제거
# =========================================================
async def t_empty():
    print("\n--- 빈 토큰 제거 ---")
    events = [e async for e in _token_events(_gen(["", "", "안녕", "", "하세요"]))]
    check("빈 토큰은 이벤트가 안 나감", len(events) == 2, f"{len(events)}개")
    check("내용은 순서대로 보존",
          all(w in e for e, w in zip(events, ["안녕", "하세요"])), events)
    check("SSE 형식", all(e.startswith("data: ") and e.endswith("\n\n") for e in events))

    empty_only = [e async for e in _token_events(_gen(["", "", ""]))]
    check("전부 비면 이벤트 0개", empty_only == [], empty_only)

    nothing = [e async for e in _token_events(_gen([]))]
    check("빈 스트림도 안전", nothing == [], nothing)


# =========================================================
# 2. heartbeat
# =========================================================
async def t_heartbeat():
    print("\n--- 침묵 시 keepalive ---")
    # 0.3초 간격 토큰 2개 + heartbeat 간격 0.1초 -> 사이사이 None 이 끼어야 한다
    got = [c async for c in _with_heartbeat(_gen(["가", "나"], delay=0.3), interval=0.1)]
    check("토큰은 모두 전달", [c for c in got if c is not None] == ["가", "나"], got)
    check("침묵 구간에 heartbeat 발생", got.count(None) >= 2, f"None {got.count(None)}개")

    # 토큰이 빠르면 heartbeat 없이 끝나야 한다 (불필요한 이벤트 금지)
    fast = [c async for c in _with_heartbeat(_gen(["가", "나"]), interval=5.0)]
    check("빠르면 heartbeat 없음", None not in fast, fast)

    events = [e async for e in _token_events(_gen(["", "가"], delay=0.3))]
    # interval 기본값(10초)보다 짧으므로 keepalive 없이 토큰 1개만
    check("_token_events 기본 동작", events and "가" in events[0], events)


async def t_heartbeat_sse():
    print("\n--- keepalive SSE 형식 ---")
    import app.routers.chat as C
    original = C._HEARTBEAT_INTERVAL_SEC
    C._HEARTBEAT_INTERVAL_SEC = 0.1
    try:
        events = [e async for e in _token_events(_gen(["가"], delay=0.35))]
    finally:
        C._HEARTBEAT_INTERVAL_SEC = original
    keepalives = [e for e in events if e == _SSE_KEEPALIVE]
    check("keepalive 가 섞여 나온다", len(keepalives) >= 2, f"{len(keepalives)}개")
    # EventSource 는 ':' 로 시작하는 줄을 무시한다 - 클라이언트 변경이 필요 없는 이유
    check("keepalive 는 SSE 주석", _SSE_KEEPALIVE.startswith(":"), repr(_SSE_KEEPALIVE))
    check("토큰도 정상 전달", any("가" in e for e in events), events)


# =========================================================
# 3. 적재 동시 실행
# =========================================================
async def t_ingest():
    print("\n--- 적재 동시 실행 ---")
    CALLS.clear()
    started = asyncio.get_running_loop().time()
    result = await _ingest_all_sources(["삼성전자", "SK하이닉스"])
    elapsed = asyncio.get_running_loop().time() - started

    check("티커별 결과", set(result) == {"삼성전자", "SK하이닉스"}, list(result))
    check("소스 3종 모두 호출", len(CALLS) == 6, f"{len(CALLS)}회")
    # 각 호출이 0.05초. 순차면 0.3초, 동시면 0.05초대.
    check("순차가 아니라 동시", elapsed < 0.15, f"{elapsed:.2f}초")
    check("건수 반환", result["삼성전자"]["news"] == 4, result["삼성전자"])

    # 한 소스가 실패해도 나머지로 답변해야 한다
    FAIL_ON.add("dart")
    try:
        CALLS.clear()
        result = await _ingest_all_sources(["삼성전자"])
    finally:
        FAIL_ON.discard("dart")
    check("실패한 소스는 0으로", result["삼성전자"]["dart"] == 0, result["삼성전자"])
    check("나머지 소스는 살아있음",
          result["삼성전자"]["news"] == 4 and result["삼성전자"]["dart_financials"] == 3,
          result["삼성전자"])


if __name__ == "__main__":
    run(t_empty())
    run(t_heartbeat())
    run(t_heartbeat_sse())
    run(t_ingest())

    print("\n" + "=" * 55)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("ALL PASS")
