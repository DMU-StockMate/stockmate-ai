import io
import zipfile
import xml.etree.ElementTree as ET
import httpx
from app.core.config import settings

DART_BASE_URL = "https://opendart.fss.or.kr/api"

# 앱 시작 시 한 번만 로드
_corp_code_map: dict[str, str] = {}

async def load_corp_codes() -> None:
    """DART 전체 회사 코드 목록 로드"""
    global _corp_code_map
    if _corp_code_map:
        return

    params = {"crtfc_key": settings.DART_API_KEY}
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{DART_BASE_URL}/corpCode.xml", params=params)
        response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open("CORPCODE.xml") as f:
            tree = ET.parse(f)

    for item in tree.getroot().findall("list"):
        name = item.findtext("corp_name", "").strip()
        code = item.findtext("corp_code", "").strip()
        stock_code = item.findtext("stock_code", "").strip()
        # 상장사만 저장 (stock_code 있는 것)
        if name and code and stock_code:
            _corp_code_map[name] = code

async def get_corp_code(corp_name: str) -> str | None:
    await load_corp_codes()
    return _corp_code_map.get(corp_name)

async def get_disclosures(corp_name: str, days: int = 30) -> list[dict]:
    corp_code = await get_corp_code(corp_name)
    if not corp_code:
        return []

    params = {
        "crtfc_key": settings.DART_API_KEY,
        "corp_code": corp_code,
        "bgn_de": _days_ago(days),
        "page_count": 10,
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{DART_BASE_URL}/list.json", params=params)
        response.raise_for_status()
        data = response.json()

    if data.get("status") != "000":
        return []

    return [
        {
            "title": item["report_nm"].strip(),
            "corp_name": item["corp_name"],
            "date": item["rcept_dt"],
            "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={item['rcept_no']}",
        }
        for item in data.get("list", [])
    ]

def _days_ago(days: int) -> str:
    from datetime import datetime, timedelta
    return (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")