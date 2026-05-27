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