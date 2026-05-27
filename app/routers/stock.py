from fastapi import APIRouter, HTTPException
from app.services.external.kis import get_stock_info
from app.services.external.dart import _stock_code_map

router = APIRouter(prefix="/stock", tags=["stock"])

@router.get("/test/{ticker_name}")
async def test_stock(ticker_name: str):
    stock_code = _stock_code_map.get(ticker_name)
    if not stock_code:
        raise HTTPException(status_code=404, detail=f"{ticker_name} 종목 코드를 찾을 수 없습니다.")
    
    result = await get_stock_info(ticker_name, stock_code)
    return result

from app.services.external.dart import get_disclosures

@router.get("/test-dart/{ticker}")
async def test_dart(ticker: str):
    result = await get_disclosures(ticker, days=90)
    return {"count": len(result), "data": result}


@router.get("/test-ingest/{ticker}")
async def test_ingest(ticker: str):
    from app.services.rag.ingestion import ingest_news, ingest_disclosures
    news = await ingest_news(ticker)
    dart = await ingest_disclosures(ticker)
    return {"news": news, "dart": dart}