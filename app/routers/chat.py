import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.schemas.chat import AskRequest, AskResponse
from app.services.rag.ingestion import ingest_news, ingest_disclosures
from app.services.rag.chain import (
    run_rag_chain, run_general_chain,
    stream_rag_chain, stream_general_chain,
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
        try:
            tickers = extract_tickers(req.question)

            if not tickers and req.history:
                tickers = extract_tickers_from_history(req.history)

            ingested = {}

            yield f"data: {json.dumps({'type': 'meta', 'tickers': tickers}, ensure_ascii=False)}\n\n"

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

            yield f"data: {json.dumps({'type': 'done', 'ingested': ingested}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")