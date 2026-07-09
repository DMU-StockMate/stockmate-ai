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


_REPRT_NAMES = {
    "11011": "사업보고서",
    "11012": "반기보고서",
    "11013": "1분기보고서",
    "11014": "3분기보고서",
}

# 최신 정보 우선: 사업보고서(연간, 가장 포괄적) -> 3분기 -> 반기 -> 1분기
_REPRT_CODE_ORDER = ["11011", "11014", "11012", "11013"]


def _recent_report_periods(lookback: int = 8) -> list[tuple[str, str]]:
    """최신순으로 시도할 (사업연도, 보고서코드) 후보 목록을 생성한다.

    아직 제출되지 않은 기간은 DART가 status=013(조회된 데이터 없음)으로 응답하므로,
    호출 측에서 후보를 순서대로 시도하며 첫 성공 결과를 사용하면 된다.
    """
    from datetime import date
    this_year = date.today().year

    candidates: list[tuple[str, str]] = []
    for year in (this_year, this_year - 1, this_year - 2):
        for code in _REPRT_CODE_ORDER:
            candidates.append((str(year), code))
    return candidates[:lookback]


async def get_financial_statements(corp_name: str, lookback: int = 8) -> list[dict]:
    """가장 최근에 제출된 정기보고서의 주요 재무계정을 조회한다.

    연결재무제표(CFS)가 있으면 우선 사용하고, 없으면 개별재무제표(OFS)를 사용한다.
    """
    corp_code = await get_corp_code(corp_name)
    if not corp_code:
        logger.warning(f"DART corp_code 없음: {corp_name}")
        return []

    for bsns_year, reprt_code in _recent_report_periods(lookback):
        params = {
            "crtfc_key": settings.DART_API_KEY,
            "corp_code": corp_code,
            "bsns_year": bsns_year,
            "reprt_code": reprt_code,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{DART_BASE_URL}/fnlttSinglAcnt.json", params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            logger.error(f"DART 재무제표 타임아웃 [{corp_name}] {bsns_year}/{reprt_code}")
            continue
        except Exception as e:
            logger.error(f"DART 재무제표 예외 [{corp_name}] {bsns_year}/{reprt_code}: {e}")
            continue

        status = data.get("status")
        if status == "013":
            # 해당 기간 보고서 미제출 - 더 이전 기간 시도
            continue
        if status != "000":
            logger.warning(f"DART 재무제표 응답 이상 [{corp_name}] {bsns_year}/{reprt_code}: {status}")
            continue

        items = data.get("list", [])
        if not items:
            continue

        logger.info(f"DART 재무제표 [{corp_name}] {bsns_year}/{reprt_code}: {len(items)}건")
        return _select_financial_items(items, corp_name, bsns_year, reprt_code)

    logger.warning(f"DART 재무제표 조회 실패(가용 기간 없음) [{corp_name}]")
    return []


def _select_financial_items(
    items: list[dict], corp_name: str, bsns_year: str, reprt_code: str
) -> list[dict]:
    fs_div_priority = "CFS" if any(i.get("fs_div") == "CFS" for i in items) else "OFS"
    filtered = [i for i in items if i.get("fs_div") == fs_div_priority]

    return [
        {
            "corp_name": corp_name,
            "bsns_year": bsns_year,
            "reprt_code": reprt_code,
            "reprt_name": _REPRT_NAMES.get(reprt_code, reprt_code),
            "fs_nm": item.get("fs_nm", ""),
            "sj_nm": item.get("sj_nm", ""),
            "account_nm": item.get("account_nm", ""),
            "thstrm_nm": item.get("thstrm_nm", ""),
            "thstrm_amount": item.get("thstrm_amount", ""),
            "frmtrm_nm": item.get("frmtrm_nm", ""),
            "frmtrm_amount": item.get("frmtrm_amount", ""),
            "currency": item.get("currency", "KRW"),
        }
        for item in filtered
    ]