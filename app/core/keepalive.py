"""응답이 오래 걸릴 때 연결을 살려 두는 장치.

RunPod HTTP 프록시(Cloudflare)는 **응답 첫 바이트까지 100초**를 기다린 뒤
HTTP 524 로 끊는다. 2026-09-05 파드 검증에서 직접 재현했고, 그때 서버는
요청을 정상 처리하고 있었다 — 클라이언트가 524 를 받은 뒤에도 로그에
`퀴즈 뱅크 저장: 5개` 가 계속 찍혔다. **생성이 실패한 게 아니라 전달이 실패한 것이다.**

일단 바이트가 흐르기 시작하면 제한은 "무응답 구간" 기준으로 바뀐다. 그래서
생성이 끝날 때까지 주기적으로 **공백**을 흘려보내면 100초 벽이 사라진다.
JSON 문법상 값 앞의 공백은 무시되므로(RFC 8259 §2) 클라이언트는 `JSON.parse`
든 axios 기본 동작이든 그대로 파싱한다. **백엔드 코드 변경이 필요 없고
응답 스키마도 Content-Type 도 그대로다.**

`/chat/stream` 이 SSE 로 버퍼링 없이 흐르는 것은 2026-08-30 파드에서 확인했다
(923줄이 12.8초에 걸쳐 분산 도착). 같은 프록시를 `application/json` 으로도
통과하는지는 파드에서 별도 확인이 필요하다.

## 상태 코드에 관한 제약 (중요)

첫 바이트를 보내는 순간 HTTP 상태 코드는 200 으로 확정된다. 그래서:

- **하트비트 시작 전(기본 30초)에 난 예외**는 예전 그대로다. 라우터의
  `except` 가 받아 422 / 500 이 정상적으로 나간다. 실제로 잡히는 에러
  (`PromptTopicError` = 주제 판별 실패)는 분석 단계에서 나므로 약 4초면
  결정된다. 30초는 이 여유를 벌어 두려고 고른 값이다.
- **하트비트가 시작된 뒤에 난 예외**(LLM 재시도 소진 등)는 상태 코드를
  바꿀 수 없다. 이때는 `{"detail": ..., "keepalive_error": true}` 를 본문으로
  내보낸다. `questions` 키가 **없으므로** 성공 응답과 절대 헷갈리지 않는다.

  처음에는 "연결을 끊으면 클라이언트가 알아서 에러로 본다"고 설계했는데
  **틀렸다.** 로컬에서 와이어를 직접 떠 보니(curl -v, 2026-09-08) uvicorn 이
  청크 스트림을 정상 종료해서 클라이언트는 `200 OK` + 공백뿐인 본문을 받고
  curl 은 exit 0 을 냈다. 실패 신호가 하나도 없다. 그래서 본문으로 알린다.

  백엔드는 `questions` 유무만 확인하면 된다. 확인하지 않아도 `questions` 가
  undefined 라 그쪽 코드에서 즉시 터지므로 조용히 잘못되지는 않는다.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable

from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from app.core.config import settings

logger = logging.getLogger(__name__)

# 한 박동에 흘려보내는 공백 덩어리.
# 32바이트로 둔 것은 중간 프록시가 아주 작은 쓰기를 모아 둘 가능성을 피하려는
# 것이고, 끝을 개행으로 둔 것은 raw 응답을 덤프했을 때 박동을 세기 쉬워서다.
_BEAT = b" " * 31 + b"\n"


def encode_json(value: Any) -> bytes:
    """FastAPI 의 JSONResponse 와 같은 방식으로 직렬화한다.

    구분자와 ensure_ascii 까지 맞춰야 하트비트 경로와 일반 경로의 본문이
    바이트 단위로 같아진다. 백엔드가 두 경로를 구분할 수 없어야 한다.
    """
    return json.dumps(
        jsonable_encoder(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


async def json_with_keepalive(coro: Awaitable[Any], *, label: str = "요청") -> Any:
    """`coro` 를 기다리되, 오래 걸리면 공백을 흘리는 스트리밍 응답으로 바꾼다.

    반환값은 둘 중 하나다.

    - 제한 시간 안에 끝났으면 `coro` 의 결과 그대로. 라우터가 평소처럼
      `response_model` 로 직렬화한다. 예외도 그대로 올라간다.
    - 넘겼으면 `StreamingResponse`. FastAPI 는 `Response` 인스턴스를
      그대로 내보내므로 OpenAPI 계약은 변하지 않는다.
    """
    if not settings.RESPONSE_KEEPALIVE_ENABLED:
        return await coro

    delay = settings.RESPONSE_KEEPALIVE_DELAY_SEC
    interval = settings.RESPONSE_KEEPALIVE_INTERVAL_SEC

    task = asyncio.ensure_future(coro)
    done, _ = await asyncio.wait({task}, timeout=delay)
    if task in done:
        # 제시간에 끝났다. 예외라면 여기서 그대로 올라가 라우터의 except 가 받는다.
        return task.result()

    logger.info(f"{label}: {delay:.0f}초를 넘겨 하트비트 응답으로 전환한다")

    async def _stream():
        beats = 0
        try:
            while True:
                finished, _ = await asyncio.wait({task}, timeout=interval)
                if finished:
                    break
                beats += 1
                yield _BEAT
            body = encode_json(task.result())
        except Exception as e:
            # 이미 200 헤더를 내보낸 뒤라 상태 코드를 바꿀 수 없다.
            # 성공 응답과 구분되도록 questions 없는 에러 본문을 내보낸다.
            logger.error(
                f"{label}: 하트비트 {beats}회 뒤 실패 - 에러 본문을 반환한다: {e}",
                exc_info=True,
            )
            body = encode_json({
                "detail": str(e),
                "error_type": type(e).__name__,
                "keepalive_error": True,
            })
        finally:
            # 클라이언트가 먼저 끊으면 여기로 GeneratorExit 가 들어온다.
            # 아무도 안 받을 응답을 위해 GPU 를 더 태울 이유가 없다.
            if not task.done():
                task.cancel()
                logger.warning(f"{label}: 수신 측이 끊어 생성 작업을 취소했다")

        logger.info(f"{label}: 하트비트 {beats}회 뒤 본문 {len(body)}바이트 전송")
        yield body

    return StreamingResponse(_stream(), media_type="application/json")
