import asyncio
import json
import time
from datetime import datetime
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import AskRequest, AskResponse, ChatStreamRequest
from app.services.rag.ingestion import ingest_news, ingest_disclosures, ingest_financials
from app.services.rag.chain import (
    run_rag_chain, run_general_chain, run_quiz_chain, run_rag_evaluate,
    stream_rag_chain, stream_general_chain, stream_quiz_chain,
)
from app.services.rag.ticker_extractor import extract_tickers, extract_tickers_from_history
from app.core.logger import setup_logger

# /chat/evaluate 의 예외 핸들러가 logger 를 쓰는데 정의가 없어
# 에러 발생 시 NameError 로 원래 예외가 가려지고 있었다.
logger = setup_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


# 토큰 사이 침묵이 이보다 길어지면 SSE 주석을 흘려 연결을 유지한다.
# 사고(reasoning) 구간에는 본문 토큰이 한동안 안 나오는데, 그 사이 아무것도
# 보내지 않으면 프록시나 클라이언트가 유휴로 보고 끊을 수 있다.
_HEARTBEAT_INTERVAL_SEC = 10.0

# SSE 주석. EventSource 는 ':' 로 시작하는 줄을 자동으로 무시하므로
# 클라이언트 변경 없이 연결만 살려둔다.
_SSE_KEEPALIVE = ": keepalive\n\n"


async def _ingest_all_sources(tickers: list[str]) -> dict:
    """티커별 뉴스/공시/재무를 **동시에** 적재하고 건수를 dict로 모아 반환한다.

    원래는 티커 × 소스를 이중 for 문으로 순차 호출했다. 전부 외부 API
    (네이버 뉴스 / DART / DART 재무) 왕복이라 서로 기다릴 이유가 전혀 없는데,
    종목 2개면 6번을 줄 세워 기다리고 있었다. 답변 시작 전에 치르는 비용이라
    사용자가 체감하는 지연에 그대로 얹힌다.

    한 소스가 실패해도 나머지로 답변한다 (KIS 시세가 이미 그렇게 동작한다).
    적재는 보조 자료이지, 이것 때문에 채팅 전체가 실패하면 안 된다.
    """
    async def _one(ticker: str) -> tuple[str, dict]:
        news, dart, financials = await asyncio.gather(
            ingest_news(ticker),
            ingest_disclosures(ticker),
            ingest_financials(ticker),
            return_exceptions=True,
        )

        counts: dict = {}
        for key, outcome in (("news", news), ("dart", dart),
                             ("dart_financials", financials)):
            if isinstance(outcome, BaseException):
                logger.warning(f"적재 실패 [{ticker}/{key}] - 건너뜀: {outcome}")
                counts[key] = 0
            else:
                counts[key] = outcome
        return ticker, counts

    pairs = await asyncio.gather(*(_one(t) for t in tickers))
    return dict(pairs)


async def _with_heartbeat(chunks, interval: float = _HEARTBEAT_INTERVAL_SEC):
    """토큰 스트림을 감싸, 침묵이 interval 을 넘으면 None 을 한 번 흘린다.

    None = "아직 살아 있다" 신호이며 호출부가 SSE 주석으로 바꾼다.
    """
    iterator = chunks.__aiter__()
    pending = asyncio.ensure_future(iterator.__anext__())
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                yield None
                continue
            try:
                chunk = pending.result()
            except StopAsyncIteration:
                return
            yield chunk
            pending = asyncio.ensure_future(iterator.__anext__())
    finally:
        if not pending.done():
            pending.cancel()


async def _token_events(chunks):
    """토큰 스트림을 SSE 이벤트 문자열로 바꾼다.

    **빈 토큰은 내보내지 않는다.** 서버가 `reasoning_effort` 를 켠 채로 뜨면
    vLLM 의 qwen3 reasoning 파서가 사고 토큰을 `reasoning_content` 로 보내고
    `content` 는 빈 문자열로 온다. LangChain 은 그 청크를 그대로 흘리므로
    스트림 **초반에** `{"type":"token","content":""}` 가 수십~수백 개 쌓인다
    (2026-08-30 실측: token 이벤트 984개 중 341개가 빈 문자열).

    클라이언트가 무시하면 되긴 하지만, 무의미한 이벤트를 수백 개 보내는 것은
    대역폭과 파싱 낭비이고 "왜 빈 게 오지?"를 매번 설명해야 한다. 서버에서 막는다.

    대신 그 구간이 통째로 침묵이 되므로 주기적으로 SSE 주석을 흘려 연결을 지킨다.
    """
    # 간격을 호출 시점에 읽는다. 기본 인자로 넘기면 함수 정의 시점의 값이
    # 박혀서, 상수를 바꿔도 반영되지 않는다 (테스트가 잡아낸 실제 버그).
    async for chunk in _with_heartbeat(chunks, _HEARTBEAT_INTERVAL_SEC):
        if chunk is None:
            yield _SSE_KEEPALIVE
        elif chunk:
            yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"


@router.post(
    "/stream",
    summary="백엔드 전용 통합 채팅 스트리밍",
    description="""
NestJS 백엔드에서 호출하는 SSE 스트리밍 엔드포인트입니다.

---

**SSE 이벤트 순서**

1. meta 이벤트 - 스트리밍 시작 시 1회 전송
2. token 이벤트 - 답변 토큰 순차 전송
3. done 이벤트 - 스트리밍 완료 시 1회 전송

---

**이벤트별 데이터 형식**

meta:
```json
{
    "type": "meta",
    "tickers": ["삼성전자", "SK하이닉스"],
    "timestamp": "2026-05-27T14:30:00"
}
```

token:
```json
{
    "type": "token",
    "content": "안녕하세요"
}
```

⚠️ **`content`가 빈 문자열인 token 이벤트는 더 이상 오지 않습니다.**
서버가 사고(reasoning) 모드로 뜨면 그 구간의 청크는 `content`가 비어서 오는데,
예전에는 그것까지 그대로 흘려보냈습니다(실측: 984개 중 341개가 빈 문자열).
이제 서버에서 걸러내므로 클라이언트는 빈 문자열 처리를 하지 않아도 됩니다.

대신 그 구간은 통째로 침묵이 되므로, 10초 이상 토큰이 없으면 서버가
**SSE 주석**(`: keepalive`)을 흘려 연결을 유지합니다. `EventSource`는 `:`로
시작하는 줄을 자동으로 무시하므로 클라이언트 변경은 필요 없습니다.
직접 파싱하는 경우에만 `:` 로 시작하는 줄을 건너뛰세요.

done:
```json
{
    "type": "done",
    "ingested": {
        "삼성전자": {
            "news": 5,
            "dart": 3
        }
    },
    "elapsed_sec": 3.42,
    "timestamp": "2026-05-27T14:30:03"
}
```

error (에러 발생 시):
```json
{
    "type": "error",
    "message": "에러 내용",
    "elapsed_sec": 1.23
}
```

---

**일반 채팅 (quiz_context: null)**
- 질문에서 종목명 자동 추출
- 종목 있으면 뉴스/공시 기반 RAG 답변
- 종목 없으면 LLM 직접 답변
- investment_level에 맞게 답변 난이도 조절

**문제 기반 Q&A (quiz_context 포함)**
- RAG 없이 문제 정보 기반으로 답변
- 사용자 수준에 맞게 설명 난이도 조절

---

**investment_level 값**
- 입문 / 초급 / 중급 / 고급 / 미설정

**주의사항**
- history는 최근 10개만 포함
- history 없으면 빈 배열로 전송
- quiz_context.explanation은 null 허용
""",
    response_description="SSE 스트리밍 응답",
    responses={
        200: {
            "description": "SSE 스트리밍 성공",
            "content": {
                "text/event-stream": {
                    "example": 'data: {"type": "token", "content": "안녕하세요"}\n\n'
                }
            },
        }
    },
)
async def chat_stream(req: ChatStreamRequest):
    async def generate():
        start_time = time.time()
        started_at = datetime.now().isoformat()
        investment_level = req.user.investment_level if req.user else "미설정"

        try:
            if req.quiz_context:
                ingested = {}
                yield f"data: {json.dumps({'type': 'meta', 'tickers': [], 'timestamp': started_at}, ensure_ascii=False)}\n\n"

                async for event in _token_events(stream_quiz_chain(
                    req.question, req.history, req.quiz_context, investment_level,
                    ingested_out=ingested,
                )):
                    yield event

            else:
                tickers = extract_tickers(req.question)
                if not tickers and req.history:
                    tickers = extract_tickers_from_history(req.history)

                ingested = {}
                yield f"data: {json.dumps({'type': 'meta', 'tickers': tickers, 'timestamp': started_at}, ensure_ascii=False)}\n\n"

                if tickers:
                    ingested = await _ingest_all_sources(tickers)
                    async for event in _token_events(stream_rag_chain(
                        req.question, tickers, req.history, investment_level
                    )):
                        yield event
                else:
                    async for event in _token_events(stream_general_chain(
                        req.question, req.history, investment_level
                    )):
                        yield event

            elapsed = round(time.time() - start_time, 2)
            yield f"data: {json.dumps({'type': 'done', 'ingested': ingested, 'elapsed_sec': elapsed, 'timestamp': datetime.now().isoformat()}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'elapsed_sec': elapsed}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/ask", response_model=AskResponse, include_in_schema=False)
async def ask(req: AskRequest):
    tickers = extract_tickers(req.question)
    if not tickers and req.history:
        tickers = extract_tickers_from_history(req.history)

    ingested = {}
    if tickers:
        ingested = await _ingest_all_sources(tickers)
        answer = await run_rag_chain(req.question, tickers, req.history)
    else:
        answer = await run_general_chain(req.question, req.history)

    return AskResponse(
        question=req.question,
        tickers=tickers,
        answer=answer,
        ingested=ingested,
    )


@router.post(
    "/evaluate",
    summary="RAG 검색 가시성 디버그 (비스트리밍)",
    description="""
스트리밍 없이 한 번에 응답하며, LLM이 실제로 무엇을 근거로 답했는지 노출한다.
최신성/실시간 정확성 테스트의 기반 엔드포인트.

**응답 필드**
- `tickers`: 추출된 종목
- `mode`: "rag" | "general" (종목 없으면 general)
- `ingested`: 이번 호출에 적재된 뉴스/공시/재무 건수
- `answer`: 최종 답변(비스트리밍)
- `stock_data`: KIS 실시간 시세/지표
- `context_sent_to_llm`: LLM 프롬프트에 실제로 들어간 컨텍스트 전문
- `retrieved`: 검색 후보 목록. 각 항목에 `source`/`score`/`published_at`/`passed_score`/`taken`/`content`
  - `passed_score`: min_score 임계값 통과 여부
  - `taken`: 최종적으로 컨텍스트에 채택됐는지 (score 통과분 중 최신순 상위 take개)
""",
)
async def chat_evaluate(req: ChatStreamRequest):
    # 디버그 엔드포인트라 전역 500 핸들러가 에러를 가리지 않도록 여기서 잡아
    # 실제 예외 메시지와 traceback을 그대로 반환한다(원인 파악용).
    try:
        investment_level = req.user.investment_level if req.user else "미설정"
        tickers = extract_tickers(req.question)
        if not tickers and req.history:
            tickers = extract_tickers_from_history(req.history)

        if not tickers:
            answer = await run_general_chain(req.question, req.history, investment_level)
            return {
                "tickers": [],
                "mode": "general",
                "ingested": {},
                "answer": answer,
                "stock_data": {},
                "retrieved": [],
                "context_sent_to_llm": "",
            }

        ingested = await _ingest_all_sources(tickers)
        result = await run_rag_evaluate(req.question, tickers, req.history, investment_level)
        return {
            "tickers": tickers,
            "mode": "rag",
            "ingested": ingested,
            **result,
        }
    except Exception as e:
        import traceback
        logger.error(f"/chat/evaluate 실패: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }


@router.post("/ask/stream", include_in_schema=False)
async def ask_stream(req: AskRequest):
    async def generate():
        start_time = time.time()
        started_at = datetime.now().isoformat()

        try:
            tickers = extract_tickers(req.question)
            if not tickers and req.history:
                tickers = extract_tickers_from_history(req.history)

            ingested = {}
            yield f"data: {json.dumps({'type': 'meta', 'tickers': tickers, 'timestamp': started_at}, ensure_ascii=False)}\n\n"

            if tickers:
                ingested = await _ingest_all_sources(tickers)
                async for event in _token_events(
                    stream_rag_chain(req.question, tickers, req.history)
                ):
                    yield event
            else:
                async for event in _token_events(
                    stream_general_chain(req.question, req.history)
                ):
                    yield event

            elapsed = round(time.time() - start_time, 2)
            yield f"data: {json.dumps({'type': 'done', 'ingested': ingested, 'elapsed_sec': elapsed, 'timestamp': datetime.now().isoformat()}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'elapsed_sec': elapsed}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")