import uuid
import hashlib
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from langchain_core.documents import Document
from app.services.rag.vectorstore import get_vectorstore
from app.services.external.naver_news import search_news
from app.services.external.dart import get_disclosures, get_financial_statements
from app.core.logger import setup_logger

logger = setup_logger(__name__)

_last_fetched_news: dict[str, datetime] = {}
_last_fetched_dart: dict[str, datetime] = {}
_last_fetched_financials: dict[str, datetime] = {}
CACHE_MINUTES = 30
# 재무제표는 분기 단위로만 갱신되므로 뉴스/공시보다 훨씬 긴 캐시 주기를 둔다
FINANCIALS_CACHE_MINUTES = 60 * 24


def _is_news_cache_valid(ticker: str) -> bool:
    if ticker not in _last_fetched_news:
        return False
    return datetime.now() - _last_fetched_news[ticker] < timedelta(minutes=CACHE_MINUTES)


def _is_dart_cache_valid(ticker: str) -> bool:
    if ticker not in _last_fetched_dart:
        return False
    return datetime.now() - _last_fetched_dart[ticker] < timedelta(minutes=CACHE_MINUTES)


def _is_financials_cache_valid(ticker: str) -> bool:
    if ticker not in _last_fetched_financials:
        return False
    return datetime.now() - _last_fetched_financials[ticker] < timedelta(minutes=FINANCIALS_CACHE_MINUTES)


def _update_news_cache(ticker: str) -> None:
    _last_fetched_news[ticker] = datetime.now()


def _update_dart_cache(ticker: str) -> None:
    _last_fetched_dart[ticker] = datetime.now()


def _update_financials_cache(ticker: str) -> None:
    _last_fetched_financials[ticker] = datetime.now()


def _make_id(text: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, text))


def _format_krw_human(amount_str: str) -> str:
    """"306,220,075,000,000" 같은 원 단위 문자열을 "약 306.22조원"처럼 사람이 읽기 쉬운 형태로 변환.

    LLM에게 큰 원화 숫자를 그대로 주고 조/억 단위 변환을 맡기면 자릿수를 잘못 읽어
    1000배씩 틀리는 경우가 실측(RAGAS faithfulness 평가)에서 확인됐다. 여기서 미리 변환해
    컨텍스트에 함께 넣어주면 LLM이 계산 대신 그대로 인용만 하면 되어 오류가 줄어든다.
    """
    try:
        amount = int(str(amount_str).replace(",", ""))
    except (ValueError, TypeError):
        return ""

    JO = 1_000_000_000_000  # 1조
    EOK = 100_000_000  # 1억

    sign = "-" if amount < 0 else ""
    abs_amount = abs(amount)

    if abs_amount >= JO:
        return f"약 {sign}{abs_amount / JO:,.2f}조원"
    if abs_amount >= EOK:
        return f"약 {sign}{abs_amount / EOK:,.1f}억원"
    return f"{sign}{abs_amount:,}원"


def _parse_pub_date(pub_date: str) -> int:
    dt = parsedate_to_datetime(pub_date)
    return int(dt.strftime("%Y%m%d"))


async def ingest_news(ticker: str, display: int = 10) -> int:
    if _is_news_cache_valid(ticker):
        logger.info(f"뉴스 캐시 유효 [{ticker}] - 스킵")
        return 0

    items = await search_news(ticker, display=display)
    if not items:
        return 0

    vs = get_vectorstore()
    docs, ids = [], []

    for item in items:
        content = f"{item['title']}\n{item['description']}"
        doc_id = _make_id(item["link"])
        docs.append(Document(
            page_content=content,
            metadata={
                "ticker": ticker,
                "source": "naver_news",
                "published_at": _parse_pub_date(item["pub_date"]),
                "link": item["link"],
            }
        ))
        ids.append(doc_id)

    client = vs.client
    new_docs, new_ids = [], []
    for doc, id_ in zip(docs, ids):
        try:
            results = client.retrieve(
                collection_name=vs.collection_name,
                ids=[id_],
            )
            if not results:
                new_docs.append(doc)
                new_ids.append(id_)
        except Exception:
            new_docs.append(doc)
            new_ids.append(id_)

    if new_docs:
        try:
            vs.add_documents(documents=new_docs, ids=new_ids)
            logger.info(f"뉴스 적재 [{ticker}]: {len(new_docs)}건")
        except Exception as e:
            logger.error(f"뉴스 적재 실패 [{ticker}]: {e}")
            return 0

    _update_news_cache(ticker)
    return len(new_docs)


async def ingest_disclosures(ticker: str, days: int = 90) -> int:
    if _is_dart_cache_valid(ticker):
        logger.info(f"공시 캐시 유효 [{ticker}] - 스킵")
        return 0

    items = await get_disclosures(ticker, days=days)
    if not items:
        return 0

    vs = get_vectorstore()
    docs, ids = [], []

    for item in items:
        content = f"{item['corp_name']} 공시: {item['title']}\n날짜: {item['date']}"
        doc_id = _make_id(item["url"])
        docs.append(Document(
            page_content=content,
            metadata={
                "ticker": ticker,
                "source": "dart",
                "published_at": int(item["date"]),
                "link": item["url"],
            }
        ))
        ids.append(doc_id)

    client = vs.client
    new_docs, new_ids = [], []
    for doc, id_ in zip(docs, ids):
        try:
            results = client.retrieve(
                collection_name=vs.collection_name,
                ids=[id_],
            )
            if not results:
                new_docs.append(doc)
                new_ids.append(id_)
        except Exception:
            new_docs.append(doc)
            new_ids.append(id_)

    if new_docs:
        try:
            vs.add_documents(documents=new_docs, ids=new_ids)
            logger.info(f"공시 적재 [{ticker}]: {len(new_docs)}건")
        except Exception as e:
            logger.error(f"공시 적재 실패 [{ticker}]: {e}")
            return 0

    _update_dart_cache(ticker)
    return len(new_docs)


async def ingest_financials(ticker: str) -> int:
    """DART 정기보고서의 주요 재무계정(매출액/영업이익/자산총계 등)을 Qdrant에 적재한다.

    list.json은 공시 제목만 담고 있어 LLM이 실제 실적 수치를 답할 수 없었던 문제를
    보완하기 위해 fnlttSinglAcnt.json 기반 재무 수치를 별도 source로 저장한다.
    """
    if _is_financials_cache_valid(ticker):
        logger.info(f"재무제표 캐시 유효 [{ticker}] - 스킵")
        return 0

    items = await get_financial_statements(ticker)
    if not items:
        return 0

    corp_name = items[0]["corp_name"]
    bsns_year = items[0]["bsns_year"]
    reprt_code = items[0]["reprt_code"]
    reprt_name = items[0]["reprt_name"]
    fs_nm = items[0]["fs_nm"]
    thstrm_nm = items[0]["thstrm_nm"]

    lines = [
        f"{corp_name} {bsns_year}년 {reprt_name} 주요 재무계정 ({fs_nm}, {thstrm_nm} 기준)",
        "(금액 뒤 괄호의 '약 N조원/억원'은 이미 변환된 값이니 다시 계산하지 말고 그대로 인용할 것)",
    ]
    for item in items:
        thstrm_human = _format_krw_human(item["thstrm_amount"]) if item["currency"] == "KRW" else ""
        frmtrm_human = _format_krw_human(item["frmtrm_amount"]) if item["currency"] == "KRW" else ""
        lines.append(
            f"- {item['account_nm']}: 당기 {item['thstrm_amount']}{item['currency']}"
            f"{f' ({thstrm_human})' if thstrm_human else ''} "
            f"(전기 {item['frmtrm_nm']} {item['frmtrm_amount']}{item['currency']}"
            f"{f', {frmtrm_human}' if frmtrm_human else ''})"
        )
    content = "\n".join(lines)

    doc_id = _make_id(f"dart_financials:{ticker}:{bsns_year}:{reprt_code}")
    doc = Document(
        page_content=content,
        metadata={
            "ticker": ticker,
            "source": "dart_financials",
            # 접수일자 대신 사업연도 1/1을 기준시점으로 사용 (정확한 접수일은 list.json 쪽에서 관리)
            "published_at": int(f"{bsns_year}0101"),
            "bsns_year": bsns_year,
            "reprt_code": reprt_code,
        },
    )

    # news/dart와 달리 티커+기간당 문서가 딱 1개뿐이라 exists-면-skip 방식을 쓰지 않고
    # 항상 upsert한다 (doc_id가 고정이라 add_documents가 같은 id면 덮어쓴다).
    # 그래야 포맷팅/프롬프트 개선으로 content가 바뀌었을 때도 반영된다.
    vs = get_vectorstore()
    try:
        vs.add_documents(documents=[doc], ids=[doc_id])
        logger.info(f"재무제표 적재(upsert) [{ticker}] {bsns_year}/{reprt_code}: {len(items)}개 계정")
    except Exception as e:
        logger.error(f"재무제표 적재 실패 [{ticker}]: {e}")
        return 0

    _update_financials_cache(ticker)
    return 1