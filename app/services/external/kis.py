import httpx
from datetime import datetime, timedelta
from app.core.config import settings

KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"

# 토큰 캐시 (24시간 유효)
_token: str | None = None
_token_expired_at: datetime | None = None


async def _get_access_token() -> str:
    global _token, _token_expired_at

    if _token and _token_expired_at and datetime.now() < _token_expired_at:
        return _token

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{KIS_BASE_URL}/oauth2/tokenP",
            json={
                "grant_type": "client_credentials",
                "appkey": settings.KIS_APP_KEY,
                "appsecret": settings.KIS_APP_SECRET,
            },
        )
        response.raise_for_status()
        data = response.json()

    _token = data["access_token"]
    _token_expired_at = datetime.now() + timedelta(hours=23)
    return _token


async def get_stock_info(ticker_name: str, stock_code: str) -> dict:
    """현재가 + PER/PBR/EPS 조회"""
    token = await _get_access_token()

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{KIS_BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price",
            headers={
                "authorization": f"Bearer {token}",
                "appkey": settings.KIS_APP_KEY,
                "appsecret": settings.KIS_APP_SECRET,
                "tr_id": "FHKST01010100",
            },
            params={
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": stock_code,
            },
        )
        response.raise_for_status()
        data = response.json()["output"]

    return {
        "ticker": ticker_name,
        "stock_code": stock_code,
        "current_price": int(data["stck_prpr"]),
        "change_rate": float(data["prdy_ctrt"]),
        "volume": int(data["acml_vol"]),
        "per": float(data["per"]) if data["per"] else None,
        "pbr": float(data["pbr"]) if data["pbr"] else None,
        "eps": int(float(data["eps"])) if data["eps"] else None,
    }


async def get_stocks_info(ticker_names: list[str]) -> dict[str, dict]:
    """여러 종목 한번에 조회"""
    from app.services.external.dart import _stock_code_map

    results = {}
    for ticker_name in ticker_names:
        stock_code = _stock_code_map.get(ticker_name)
        if not stock_code:
            continue
        try:
            results[ticker_name] = await get_stock_info(ticker_name, stock_code)
        except Exception:
            continue
    return results
