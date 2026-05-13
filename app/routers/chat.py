import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import AskRequest, AskResponse
from app.services.rag.ingestion import ingest_news, ingest_disclosures
from app.services.rag.chain import (
    run_rag_chain, run_general_chain,
    stream_rag_chain, stream_general_chain,
)
from app.services.rag.ticker_extractor import extract_tickers

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """일반 JSON 응답"""
    tickers = extract_tickers(req.question)
    ingested = {}

    if tickers:
        for ticker in tickers:
            news_count = await ingest_news(ticker)
            dart_count = await ingest_disclosures(ticker)
            ingested[ticker] = {"news": news_count, "dart": dart_count}
        answer = await run_rag_chain(req.question, tickers)
    else:
        answer = await run_general_chain(req.question)

    return AskResponse(
        question=req.question,
        tickers=tickers,
        answer=answer,
        ingested=ingested,
    )


@router.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """SSE 스트리밍 응답"""

    async def generate():
        try:
            tickers = extract_tickers(req.question)
            ingested = {}

            # 1. 메타 정보 먼저 전송
            yield f"data: {json.dumps({'type': 'meta', 'tickers': tickers}, ensure_ascii=False)}\n\n"

            # 2. 데이터 적재
            if tickers:
                for ticker in tickers:
                    news_count = await ingest_news(ticker)
                    dart_count = await ingest_disclosures(ticker)
                    ingested[ticker] = {"news": news_count, "dart": dart_count}

            # 3. 토큰 스트리밍
            if tickers:
                async for chunk in stream_rag_chain(req.question, tickers):
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"
            else:
                async for chunk in stream_general_chain(req.question):
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

            # 4. 완료 신호
            yield f"data: {json.dumps({'type': 'done', 'ingested': ingested}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")