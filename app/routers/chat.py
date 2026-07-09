import json
import time
from datetime import datetime
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import AskRequest, AskResponse, ChatStreamRequest
from app.services.rag.ingestion import ingest_news, ingest_disclosures, ingest_financials
from app.services.rag.chain import (
    run_rag_chain, run_general_chain, run_quiz_chain,
    stream_rag_chain, stream_general_chain, stream_quiz_chain,
)
from app.services.rag.ticker_extractor import extract_tickers, extract_tickers_from_history

router = APIRouter(prefix="/chat", tags=["chat"])


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

                async for chunk in stream_quiz_chain(
                    req.question, req.history, req.quiz_context, investment_level,
                    ingested_out=ingested,
                ):
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

            else:
                tickers = extract_tickers(req.question)
                if not tickers and req.history:
                    tickers = extract_tickers_from_history(req.history)

                ingested = {}
                yield f"data: {json.dumps({'type': 'meta', 'tickers': tickers, 'timestamp': started_at}, ensure_ascii=False)}\n\n"

                if tickers:
                    for ticker in tickers:
                        news_count = await ingest_news(ticker)
                        dart_count = await ingest_disclosures(ticker)
                        financials_count = await ingest_financials(ticker)
                        ingested[ticker] = {
                            "news": news_count,
                            "dart": dart_count,
                            "dart_financials": financials_count,
                        }

                if tickers:
                    async for chunk in stream_rag_chain(
                        req.question, tickers, req.history, investment_level
                    ):
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"
                else:
                    async for chunk in stream_general_chain(
                        req.question, req.history, investment_level
                    ):
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

            elapsed = round(time.time() - start_time, 2)
            yield f"data: {json.dumps({'type': 'done', 'ingested': ingested, 'elapsed_sec': elapsed, 'timestamp': datetime.now().isoformat()}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'elapsed_sec': elapsed}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/ask", response_model=AskResponse, include_in_schema=False)
async def ask(req: AskRequest):
    tickers = extract_tickers(req.question)
    if not tickers and req.history:
        tickers = extract_tickers_from_history(req.history)

    ingested = {}
    if tickers:
        for ticker in tickers:
            news_count = await ingest_news(ticker)
            dart_count = await ingest_disclosures(ticker)
            financials_count = await ingest_financials(ticker)
            ingested[ticker] = {
                "news": news_count,
                "dart": dart_count,
                "dart_financials": financials_count,
            }
        answer = await run_rag_chain(req.question, tickers, req.history)
    else:
        answer = await run_general_chain(req.question, req.history)

    return AskResponse(
        question=req.question,
        tickers=tickers,
        answer=answer,
        ingested=ingested,
    )


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
                for ticker in tickers:
                    news_count = await ingest_news(ticker)
                    dart_count = await ingest_disclosures(ticker)
                    financials_count = await ingest_financials(ticker)
                    ingested[ticker] = {
                        "news": news_count,
                        "dart": dart_count,
                        "dart_financials": financials_count,
                    }

            if tickers:
                async for chunk in stream_rag_chain(req.question, tickers, req.history):
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"
            else:
                async for chunk in stream_general_chain(req.question, req.history):
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

            elapsed = round(time.time() - start_time, 2)
            yield f"data: {json.dumps({'type': 'done', 'ingested': ingested, 'elapsed_sec': elapsed, 'timestamp': datetime.now().isoformat()}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'elapsed_sec': elapsed}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")