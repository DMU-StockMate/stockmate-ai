import io
import zipfile
import xml.etree.ElementTree as ET
import httpx
from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger(__name__)

DART_BASE_URL = "https://opendart.fss.or.kr/api"

_corp_code_map: dict[str, str] = {}
_stock_code_map: dict[str, str] = {}


async def load_corp_codes() -> None:
    global _corp_code_map, _stock_code_map
    if _corp_code_map:
        return

    try:
        params = {"crtfc_key": settings.DART_API_KEY}
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{DART_BASE_URL}/corpCode.xml", params=params)
            response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open("CORPCODE.xml") as f:
                tree = ET.parse(f)

        for item in tree.getroot().findall("list"):
            name = item.findtext("corp_name", "").strip()
            code = item.findtext("corp_code", "").strip()
            stock_code = item.findtext("stock_code", "").strip()
            if name and code and stock_code:
                _corp_code_map[name] = code
                _stock_code_map[name] = stock_code

        logger.info(f"DART 종목 코드 로드 완료: {len(_corp_code_map)}개")

    except Exception as e:
        logger.error(f"DART 종목 코드 로드 실패: {e}")


async def get_corp_code(corp_name: str) -> str | None:
    await load_corp_codes()
    return _corp_code_map.get(corp_name)


async def get_disclosures(corp_name: str, days: int = 30) -> list[dict]:
    corp_code = await get_corp_code(corp_name)
    if not corp_code:
        logger.warning(f"DART corp_code 없음: {corp_name}")
        return []

    params = {
        "crtfc_key": settings.DART_API_KEY,
        "corp_code": corp_code,
        "bgn_de": _days_ago(days),
        "page_count": 10,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{DART_BASE_URL}/list.json", params=params)
            response.raise_for_status()
            data = response.json()

        if data.get("status") != "000":
            logger.warning(f"DART API 응답 이상 [{corp_name}]: {data.get('status')}")
            return []

        items = data.get("list", [])
        logger.info(f"DART [{corp_name}]: {len(items)}건 조회")
        return [
            {
                "title": item["report_nm"].strip(),
                "corp_name": item["corp_name"],
                "date": item["rcept_dt"],
                "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={item['rcept_no']}",
            }
            for item in items
        ]
    except httpx.TimeoutException:
        logger.error(f"DART API 타임아웃 [{corp_name}]")
        return []
    except Exception as e:
        logger.error(f"DART API 예외 [{corp_name}]: {e}")
        return []


def _days_ago(days: int) -> str:
    from datetime import datetime, timedelta
    return (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")