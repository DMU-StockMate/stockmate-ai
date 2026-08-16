import io
import re
import html
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
        # 10 이면 삼성전자처럼 임원 소유상황보고서가 매일 나오는 대형주는 최근 10건이
        # 전부 그걸로 채워져, 자기주식처분결정·잠정실적 같은 핵심 공시가 아예 적재되지 않는다.
        # (실측: 삼성전자 공시 후보 10건이 100% 저정보 공시였다)
        # 100 은 DART list.json 의 상한. 호출 횟수는 그대로 1회다.
        "page_count": 100,
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
                "rcept_no": item["rcept_no"],
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


def _decode_dart_bytes(raw: bytes) -> str:
    """DART 원문 XML은 UTF-8이 아닌 경우(CP949/EUC-KR)가 있어 순차 시도 후 디코딩한다."""
    for enc in ("utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


# 정보량이 낮은 공시 유형.
# 임원 개인이 200주 팔았다는 보고서 같은 것들은 대형주에서 거의 매일 나온다.
# 검색 채택(chain.py)에서 후순위로 밀고, 적재 시 원문 조회도 건너뛴다
# (본문을 받아봐야 컨텍스트에 들어갈 일이 거의 없어 API 호출만 낭비된다).
_LOW_INFO_REPORT_KEYWORDS = (
    "특정증권등소유상황보고서",
    "소유주식변동신고서",
    "대량보유상황보고서",
    "임원ㆍ주요주주",
    "임원·주요주주",
)


def is_low_info_report(title: str) -> bool:
    """공시 보고서명이 저정보 유형인지 판정한다 (공백 제거 후 부분일치)."""
    compact = re.sub(r"\s+", "", title or "")
    return any(re.sub(r"\s+", "", kw) in compact for kw in _LOW_INFO_REPORT_KEYWORDS)


def format_krw_human(amount: int | str) -> str:
    """원 단위 금액을 "약 306.22조원"처럼 사람이 읽기 쉬운 형태로 변환.

    LLM에게 큰 원화 숫자를 그대로 주고 조/억 단위 변환을 맡기면 자릿수를 잘못 읽어
    1000배씩 틀리는 경우가 실측(RAGAS faithfulness 평가)에서 확인됐다. 여기서 미리 변환해
    컨텍스트에 함께 넣어주면 LLM이 계산 대신 그대로 인용만 하면 되어 오류가 줄어든다.
    """
    try:
        value = int(str(amount).replace(",", ""))
    except (ValueError, TypeError):
        return ""

    JO = 1_000_000_000_000  # 1조
    EOK = 100_000_000  # 1억

    sign = "-" if value < 0 else ""
    abs_amount = abs(value)

    if abs_amount >= JO:
        return f"약 {sign}{abs_amount / JO:,.2f}조원"
    if abs_amount >= EOK:
        return f"약 {sign}{abs_amount / EOK:,.1f}억원"
    return f"{sign}{abs_amount:,}원"


# 공시 표는 "단위 : 백만원, %" 처럼 단위를 선언하고 숫자는 그 단위로만 적는다.
# 이걸 LLM 이 조원으로 환산하게 두면 자릿수를 틀린다
# (실측: 79,318,746백만원(=79.3조원)을 "약 793조 원"이라고 10배 틀리게 씀).
# 그래서 단위 선언을 읽어 원 단위로 되돌린 뒤, 숫자 옆에 변환값을 미리 적어준다.
_UNIT_DECL = re.compile(r"단위\s*[:：]\s*([백천]?만?)\s*원")
_UNIT_MULTIPLIER = {"백만": 1_000_000, "천": 1_000, "만": 10_000, "": 1}
# 콤마 3자리 그룹이 2개 이상인 정수만 (소수점이 붙은 증감율 '1,306.8' 등은 제외).
# 백만원 단위 기준 1,000,000 이상 = 1조원 이상만 손대므로 오탐 여지가 작다.
_BIG_NUMBER = re.compile(r"(?<![\d,.])\d{1,3}(?:,\d{3}){2,}(?![\d,]*\.\d)(?![\d,])")
_AMOUNT_HINT = ("매출", "영업이익", "순이익", "자산", "부채", "자본", "이익", "손실", "금액", "규모")


def annotate_amounts(text: str) -> str:
    """공시 본문이 선언한 금액 단위를 읽어, 큰 숫자 옆에 조/억원 환산값을 붙인다.

    단위 선언이 없거나 금액 관련 표가 아니면 아무것도 하지 않는다 (주식수·수량 오탐 방지).
    """
    decl = _UNIT_DECL.search(text or "")
    if not decl:
        return text
    raw_unit = decl.group(1)
    multiplier = _UNIT_MULTIPLIER.get(raw_unit, 1)
    if multiplier == 1:
        return text  # 이미 원 단위면 환산할 게 없다
    if not any(h in text for h in _AMOUNT_HINT):
        return text  # 금액 표가 아니면 건드리지 않는다

    def repl(m: re.Match) -> str:
        human = format_krw_human(int(m.group(0).replace(",", "")) * multiplier)
        return f"{m.group(0)}({human})" if human else m.group(0)

    annotated = _BIG_NUMBER.sub(repl, text)
    if annotated == text:
        return text
    return (
        f"(숫자 뒤 괄호의 '약 N조원/억원'은 '{raw_unit}원' 단위를 원 단위로 미리 환산해둔 "
        "값이니, 다시 계산하지 말고 그대로 인용할 것)\n" + annotated
    )


_DART_TAG = re.compile(r"<[^>]+>")

# <style>/<script>는 태그만 지우면 '안쪽 내용'(CSS/JS 텍스트)이 본문처럼 남는다.
# DART XFORMS 문서는 문서 앞머리가 통째로 스타일 정의라, 이걸 안 지우면
# max_chars 로 자르는 구간이 전부 CSS 로 채워져 실제 공시 본문이 한 글자도 안 들어간다.
# (실측: SK하이닉스 '연결재무제표기준영업(잠정)실적' 공시가 검색 1위로 채택됐는데
#  LLM 이 받은 내용은 '.xforms * { font-family: 돋움체;} ...' 뿐이었다)
_DART_STYLE_BLOCK = re.compile(
    r"<(style|script)\b[^>]*>.*?</\s*\1\s*>", re.IGNORECASE | re.DOTALL
)
# 닫는 태그가 깨진 문서 대비 — 태그 제거 후 남은 'selector { ... }' 형태의 CSS 룰을 정리한다.
# 셀렉터 부분은 ASCII 만 매칭해서, 한글 본문이 중괄호 앞에 있어도 같이 지워지지 않게 한다.
_DART_CSS_RULE = re.compile(r"[A-Za-z0-9_.#@*\-\s>:,\[\]=\"'()]{0,120}\{[^{}]*\}")


def _strip_dart_markup(text: str) -> str:
    """DART 문서의 태그/엔티티를 제거하고 공백을 정리해 읽을 수 있는 본문 텍스트로 만든다."""
    text = _DART_STYLE_BLOCK.sub(" ", text)
    text = _DART_TAG.sub(" ", text)
    text = html.unescape(text)
    text = _DART_CSS_RULE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# 2000: 단위 환산 주석(숫자당 ~12자 + 안내문 1줄)이 붙는 만큼 여유를 준 값.
# 1500 그대로 두면 환산값이 실제 본문을 밀어내 버린다.
async def get_disclosure_document(rcept_no: str, max_chars: int = 2000) -> str:
    """공시 원본 문서(document.xml)를 받아 태그를 제거한 본문 텍스트를 반환한다.

    list.json은 공시 '제목'만 주므로, "공시 내용"을 물으면 LLM이 제목만 보고 내용을
    지어내는 문제(예: 자기주식처분결정 공시를 'IR 개최+잠정실적'으로 오설명)가 있었다.
    원문 본문을 함께 저장해 실제 근거를 제공한다.
    응답이 ZIP이 아니거나 실패하면 빈 문자열을 반환해 호출부가 제목만으로 진행하도록 한다.
    """
    if not rcept_no:
        return ""
    params = {"crtfc_key": settings.DART_API_KEY, "rcept_no": rcept_no}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{DART_BASE_URL}/document.xml", params=params)
            response.raise_for_status()
        content = response.content
        # 정상 응답은 ZIP(PK 시그니처). 에러 시 DART는 XML/JSON 텍스트를 준다.
        if content[:2] != b"PK":
            logger.warning(f"공시 원문 비정상 응답 [{rcept_no}]")
            return ""
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            xml_names = [n for n in z.namelist() if n.lower().endswith(".xml")]
            if not xml_names:
                return ""
            raw = z.read(xml_names[0])
        text = _strip_dart_markup(_decode_dart_bytes(raw))
        # 자르기 전에 환산값을 붙인다 (자른 뒤에 붙이면 잘린 숫자에 잘못된 값이 달릴 수 있다).
        text = annotate_amounts(text)
        return text[:max_chars]
    except httpx.TimeoutException:
        logger.error(f"공시 원문 타임아웃 [{rcept_no}]")
        return ""
    except Exception as e:
        logger.error(f"공시 원문 조회 실패 [{rcept_no}]: {e}")
        return ""


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