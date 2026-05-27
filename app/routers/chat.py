import json
import time
from datetime import datetime
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import AskRequest, AskResponse, ChatStreamRequest
from app.services.rag.ingestion import ingest_news, ingest_disclosures
from app.services.rag.chain import (
    run_rag_chain, run_general_chain, run_quiz_chain,
    stream_rag_chain, stream_general_chain, stream_quiz_chain,
)
from app.services.rag.ticker_extractor import extract_tickers, extract_tickers_from_history

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    tickers = extract_tickers(req.question)
    if not tickers and req.history:
        tickers = extract_tickers_from_history(req.history)

    ingested = {}
    if tickers:
        for ticker in tickers:
            news_count = await ingest_news(ticker)
            dart_count = await ingest_disclosures(ticker)
            ingested[ticker] = {"news": news_count, "dart": dart_count}
        answer = await run_rag_chain(req.question, tickers, req.history)
    else:
        answer = await run_general_chain(req.question, req.history)

    return AskResponse(
        question=req.question,
        tickers=tickers,
        answer=answer,
        ingested=ingested,
    )


@router.post("/ask/stream")
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
                    ingested[ticker] = {"news": news_count, "dart": dart_count}

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


# 백엔드 전용 통합 엔드포인트
@router.post("/stream")
async def chat_stream(req: ChatStreamRequest):
    async def generate():
        start_time = time.time()
        started_at = datetime.now().isoformat()
        investment_level = req.user.investment_level if req.user else "미설정"

        try:
            # 퀴즈 모드
            if req.quiz_context:
                yield f"data: {json.dumps({'type': 'meta', 'tickers': [], 'timestamp': started_at}, ensure_ascii=False)}\n\n"

                async for chunk in stream_quiz_chain(
                    req.question, req.history, req.quiz_context, investment_level
                ):
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

            # 일반 채팅 모드
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
                        ingested[ticker] = {"news": news_count, "dart": dart_count}

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
            yield f"data: {json.dumps({'type': 'done', 'ingested': {}, 'elapsed_sec': elapsed, 'timestamp': datetime.now().isoformat()}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'elapsed_sec': elapsed}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")