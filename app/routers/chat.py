from fastapi import APIRouter
from app.schemas.chat import AskRequest, AskResponse
from app.services.rag.ingestion import ingest_news, ingest_disclosures
from app.services.rag.chain import run_rag_chain, run_general_chain
from app.services.rag.ticker_extractor import extract_tickers

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    tickers = extract_tickers(req.question)
    ingested = {}

    if tickers:
        # 종목 관련 질문 → 데이터 적재 후 RAG
        for ticker in tickers:
            news_count = await ingest_news(ticker)
            dart_count = await ingest_disclosures(ticker)
            ingested[ticker] = {"news": news_count, "dart": dart_count}

        answer = await run_rag_chain(req.question, tickers)
    else:
        # 일반 질문 → LLM 직접 답변
        answer = await run_general_chain(req.question)

    return AskResponse(
        question=req.question,
        tickers=tickers,
        answer=answer,
        ingested=ingested,
    )