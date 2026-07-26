import asyncio
import httpx
from datetime import datetime, timedelta
from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger(__name__)

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
        payload = response.json()

    # KIS는 HTTP 200이어도 rt_cd로 성공/실패를 구분한다("0"=성공).
    # 확인하지 않으면 output이 비어 KeyError가 나고, 호출부에서 조용히 삼켜져 시세만 누락된다.
    if payload.get("rt_cd") != "0":
        raise ValueError(
            f"KIS 시세 조회 실패 [{ticker_name}/{stock_code}]: "
            f"{payload.get('msg_cd')} {payload.get('msg1')}"
        )
    data = payload["output"]

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
    """여러 종목을 병렬로 조회. 실패한 종목은 로그를 남기고 건너뛴다."""
    from app.services.external.dart import _stock_code_map

    targets = [
        (name, _stock_code_map[name])
        for name in ticker_names
        if _stock_code_map.get(name)
    ]
    if not targets:
        return {}

    fetched = await asyncio.gather(
        *[get_stock_info(name, code) for name, code in targets],
        return_exceptions=True,
    )

    results: dict[str, dict] = {}
    for (name, _code), outcome in zip(targets, fetched):
        if isinstance(outcome, Exception):
            logger.warning(f"KIS 시세 누락 [{name}]: {outcome}")
            continue
        results[name] = outcome
    return results
