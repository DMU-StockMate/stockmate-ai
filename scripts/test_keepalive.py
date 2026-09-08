"""하트비트 응답 계층 회귀 테스트 (LLM·네트워크 없이).

`app/core/keepalive.py` 는 응답이 오래 걸릴 때 공백을 흘려 Cloudflare 의
100초 벽을 넘기는 장치다. 여기서 지켜야 할 성질은 네 가지다.

1. 빠른 응답은 **하나도 안 달라진다** (객체 그대로, 예외도 그대로).
2. 느린 응답의 본문은 공백을 걷어내면 **바이트 단위로 원래 JSON 과 같다.**
   백엔드가 두 경로를 구분할 수 없어야 한다.
3. 하트비트가 시작된 뒤 실패하면 **questions 없는 에러 본문**이 나간다.
   상태 코드는 이미 200 으로 굳었으므로 성공과 구분되는 것은 본문뿐이다.
4. 수신 측이 끊으면 생성 작업을 **취소해 GPU 를 놓아준다.**

조용히 깨지면 퀴즈 생성 세 경로가 통째로 영향을 받으므로 별도 테스트를 둔다.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 테스트가 몇 초 만에 끝나도록 config 를 읽기 전에 값을 줄여 둔다.
os.environ["RESPONSE_KEEPALIVE_DELAY_SEC"] = "0.3"
os.environ["RESPONSE_KEEPALIVE_INTERVAL_SEC"] = "0.1"

from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.keepalive import json_with_keepalive, encode_json  # noqa: E402

FAILURES: list[str] = []


def check(name, passed, extra=""):
    print(("PASS  " if passed else "FAIL  ") + name + (f"  {extra}" if extra else ""))
    if not passed:
        FAILURES.append(name)


def run(coro):
    asyncio.run(coro)


class Q(BaseModel):
    question: str
    choices: list[str]


class Payload(BaseModel):
    questions: list[Q]


SAMPLE = Payload(questions=[Q(question="PER 이 낮으면?", choices=["저평가", "고평가"])])


class Boom(ValueError):
    pass


async def fast_ok():
    await asyncio.sleep(0.01)
    return SAMPLE


async def fast_boom():
    await asyncio.sleep(0.01)
    raise Boom("주제를 찾지 못했습니다")


async def slow_ok(seconds=0.55):
    await asyncio.sleep(seconds)
    return SAMPLE


async def slow_boom(seconds=0.55):
    await asyncio.sleep(seconds)
    raise Boom("재시도 소진")


async def drain(resp):
    """StreamingResponse 본문을 전부 모은다."""
    out = b""
    async for chunk in resp.body_iterator:
        out += chunk
    return out


# ─────────────────────────────────────────────────────────────

async def t_fast_path():
    """제한 시간 안에 끝나면 예전 동작 그대로여야 한다."""
    result = await json_with_keepalive(fast_ok(), label="fast")
    check("빠른 응답은 모델 객체 그대로", result is SAMPLE, type(result).__name__)
    check("빠른 응답은 스트리밍이 아님", not isinstance(result, StreamingResponse))

    # 예외가 그대로 올라와야 라우터의 except 가 422/500 으로 바꿀 수 있다
    try:
        await json_with_keepalive(fast_boom(), label="fast-boom")
        check("빠른 예외는 그대로 전파", False, "예외가 안 났다")
    except Boom as e:
        check("빠른 예외는 그대로 전파", str(e) == "주제를 찾지 못했습니다")
    except Exception as e:  # noqa: BLE001
        check("빠른 예외는 그대로 전파", False, f"{type(e).__name__}: {e}")


async def t_slow_path():
    """느리면 공백을 흘리고, 본문은 원래 JSON 과 바이트 단위로 같아야 한다."""
    resp = await json_with_keepalive(slow_ok(), label="slow")
    check("느린 응답은 StreamingResponse", isinstance(resp, StreamingResponse))
    check("Content-Type 은 application/json",
          resp.media_type == "application/json", resp.media_type)

    body = await drain(resp)
    stripped = body.lstrip()
    check("앞에 공백 하트비트가 붙는다", len(body) > len(stripped),
          f"공백 {len(body) - len(stripped)}바이트")

    expected = JSONResponse(content=json.loads(SAMPLE.model_dump_json())).body
    check("공백을 걷어낸 본문이 FastAPI 응답과 바이트 동일", stripped == expected,
          f"{stripped[:60]!r} vs {expected[:60]!r}")

    # 표준 파서가 선행 공백을 그냥 넘기는지 (= 백엔드 코드 변경 불필요의 근거)
    parsed = json.loads(body.decode("utf-8"))
    check("공백이 붙은 채로 그대로 파싱된다",
          parsed["questions"][0]["question"] == "PER 이 낮으면?", parsed)

    check("한글이 이스케이프되지 않는다", "PER".encode() in stripped and b"\\u" not in stripped)


async def t_slow_failure():
    """하트비트 뒤 실패는 성공 응답과 절대 헷갈리면 안 된다.

    상태 코드는 이미 200 으로 굳었다. uvicorn 이 청크 스트림을 정상 종료하므로
    연결을 끊어도 클라이언트는 '200 + 빈 본문'을 볼 뿐 에러를 인지하지 못한다
    (2026-09-08 curl -v 로 확인). 그래서 본문으로 알린다.
    """
    resp = await json_with_keepalive(slow_boom(), label="slow-boom")
    check("실패해도 일단 스트리밍으로 전환됨", isinstance(resp, StreamingResponse))

    body = await drain(resp)
    parsed = json.loads(body.decode("utf-8"))
    check("에러 본문이 JSON 으로 파싱된다", isinstance(parsed, dict), parsed)
    check("questions 키가 없다(성공과 구분)", "questions" not in parsed, list(parsed))
    check("keepalive_error 표시가 있다", parsed.get("keepalive_error") is True, parsed)
    check("원래 예외 메시지가 담긴다",
          parsed.get("detail") == "재시도 소진" and parsed.get("error_type") == "Boom",
          parsed)


async def t_client_disconnect():
    """수신 측이 끊으면 생성 작업을 취소해 GPU 를 놓아준다."""
    state = {"cancelled": False}

    async def long_job():
        try:
            await asyncio.sleep(10)
            return SAMPLE
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    resp = await json_with_keepalive(long_job(), label="disconnect")
    agen = resp.body_iterator
    first = await agen.__anext__()
    check("첫 청크는 공백", first.strip() == b"", first[:40])
    await agen.aclose()
    await asyncio.sleep(0.05)
    check("끊기면 생성 작업이 취소된다", state["cancelled"])


async def t_disabled():
    """스위치를 끄면 아무것도 감싸지 않는다 (템플릿 env 로 되돌릴 수 있어야 한다)."""
    settings.RESPONSE_KEEPALIVE_ENABLED = False
    try:
        result = await json_with_keepalive(slow_ok(0.4), label="off")
        check("꺼져 있으면 모델 객체 그대로", result is SAMPLE, type(result).__name__)
    finally:
        settings.RESPONSE_KEEPALIVE_ENABLED = True


async def t_settings_loaded():
    check("설정이 환경변수로 읽힌다",
          settings.RESPONSE_KEEPALIVE_DELAY_SEC == 0.3
          and settings.RESPONSE_KEEPALIVE_INTERVAL_SEC == 0.1,
          f"{settings.RESPONSE_KEEPALIVE_DELAY_SEC}/{settings.RESPONSE_KEEPALIVE_INTERVAL_SEC}")
    check("encode_json 이 FastAPI 와 같은 바이트",
          encode_json(SAMPLE) == JSONResponse(content=json.loads(SAMPLE.model_dump_json())).body)


if __name__ == "__main__":
    run(t_settings_loaded())
    run(t_fast_path())
    run(t_slow_path())
    run(t_slow_failure())
    run(t_client_disconnect())
    run(t_disabled())

    print("\n" + "=" * 55)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("ALL PASS")
