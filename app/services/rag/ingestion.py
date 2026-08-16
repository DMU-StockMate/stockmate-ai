import asyncio
import uuid
import hashlib
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from langchain_core.documents import Document
from app.services.rag.vectorstore import get_vectorstore
from app.services.external.naver_news import search_news
from app.services.external.dart import (
    get_disclosures, get_financial_statements, get_disclosure_document, is_low_info_report,
    format_krw_human,
)
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


# 공시 본문(dart.py)에서도 같은 변환이 필요해져 dart.py 로 옮겼다. 호출부 호환용 별칭.
_format_krw_human = format_krw_human


def _parse_pub_date(pub_date: str) -> int | None:
    """RFC822 형식 pubDate → YYYYMMDD int. 파싱 실패 시 None.

    parsedate_to_datetime는 형식이 어긋나면 None을 반환하거나 예외를 던진다.
    이전엔 방어가 없어 None.strftime()에서 죽고 요청 전체가 500이 났다
    (네이버가 이따금 비정상 pubDate를 섞어 보내면 특정 종목에서만 재현).
    """
    try:
        dt = parsedate_to_datetime(pub_date)
        if dt is None:
            return None
        return int(dt.strftime("%Y%m%d"))
    except (TypeError, ValueError):
        return None


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
        published_at = _parse_pub_date(item["pub_date"])
        if published_at is None:
            logger.warning(f"뉴스 날짜 파싱 실패 - 스킵 [{ticker}]: {item.get('pub_date')!r}")
            continue
        content = f"{item['title']}\n{item['description']}"
        doc_id = _make_id(item["link"])
        docs.append(Document(
            page_content=content,
            metadata={
                "ticker": ticker,
                "source": "naver_news",
                "published_at": published_at,
                "link": item["link"],
            }
        ))
        ids.append(doc_id)

    # 이미 적재된 문서 id를 한 번의 retrieve로 배치 조회한다.
    # (기존엔 문서마다 retrieve를 호출해 뉴스 10건이면 10번 왕복 → 지연 컸음)
    client = vs.client
    try:
        existing = client.retrieve(collection_name=vs.collection_name, ids=ids)
        existing_ids = {str(p.id) for p in existing}
    except Exception:
        existing_ids = set()

    new_docs, new_ids = [], []
    for doc, id_ in zip(docs, ids):
        if id_ not in existing_ids:
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

    # 1) 먼저 id를 계산하고 기존재 문서를 배치 retrieve로 걸러낸다.
    #    (공시 원문 조회는 비싸므로 '새 공시'에 대해서만 본문을 받는다)
    ids = [_make_id(item["url"]) for item in items]
    client = vs.client
    try:
        existing = client.retrieve(collection_name=vs.collection_name, ids=ids)
        existing_ids = {str(p.id) for p in existing}
    except Exception:
        existing_ids = set()

    new_items = [(item, id_) for item, id_ in zip(items, ids) if id_ not in existing_ids]
    if not new_items:
        _update_dart_cache(ticker)
        return 0

    # 2) 새 공시의 원문 본문을 병렬로 조회 (실패 시 빈 문자열 → 제목만 저장)
    #    저정보 공시(임원 개인의 수백 주 매매 보고서 등)는 본문을 받지 않는다.
    #    검색에서 어차피 후순위로 밀려 컨텍스트에 들어갈 일이 거의 없는데,
    #    page_count 를 100 으로 올린 뒤로는 이 유형이 대부분이라 원문 조회가 그만큼 낭비된다.
    async def _body(item: dict) -> str:
        if is_low_info_report(item["title"]):
            return ""
        return await get_disclosure_document(item["rcept_no"])

    bodies = await asyncio.gather(*[_body(item) for item, _ in new_items])

    # 3) 제목·날짜 + (있으면) 본문을 담아 Document 구성.
    #    제목만 있던 기존 방식은 LLM이 '공시 내용'을 지어내게 만들었다 - 본문을 근거로 제공한다.
    new_docs, new_ids = [], []
    for (item, id_), body in zip(new_items, bodies):
        content = f"{item['corp_name']} 공시: {item['title']}\n날짜: {item['date']}"
        if body:
            content += f"\n내용: {body}"
        new_docs.append(Document(
            page_content=content,
            metadata={
                "ticker": ticker,
                "source": "dart",
                "published_at": int(item["date"]),
                "link": item["url"],
            }
        ))
        new_ids.append(id_)

    try:
        vs.add_documents(documents=new_docs, ids=new_ids)
        logger.info(f"공시 적재 [{ticker}]: {len(new_docs)}건 (본문 포함 {sum(1 for b in bodies if b)}건)")
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