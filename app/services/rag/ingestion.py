import hashlib
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from langchain_core.documents import Document
from app.services.rag.vectorstore import get_vectorstore
from app.services.external.naver_news import search_news
from app.services.external.dart import get_disclosures

# 종목별 마지막 fetch 시간 캐시
_last_fetched: dict[str, datetime] = {}
CACHE_MINUTES = 30


def _is_cache_valid(ticker: str) -> bool:
    if ticker not in _last_fetched:
        return False
    return datetime.now() - _last_fetched[ticker] < timedelta(minutes=CACHE_MINUTES)


def _update_cache(ticker: str) -> None:
    _last_fetched[ticker] = datetime.now()


def _make_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _parse_pub_date(pub_date: str) -> int:
    dt = parsedate_to_datetime(pub_date)
    return int(dt.strftime("%Y%m%d"))


async def ingest_news(ticker: str, display: int = 10) -> int:
    if _is_cache_valid(ticker):
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

    existing = vs.get(ids=ids)["ids"]
    new_docs = [(doc, id_) for doc, id_ in zip(docs, ids) if id_ not in existing]

    if new_docs:
        vs.add_documents(
            documents=[d for d, _ in new_docs],
            ids=[i for _, i in new_docs],
        )

    return len(new_docs)


async def ingest_disclosures(ticker: str, days: int = 90) -> int:
    if _is_cache_valid(ticker):
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

    existing = vs.get(ids=ids)["ids"]
    new_docs = [(doc, id_) for doc, id_ in zip(docs, ids) if id_ not in existing]

    if new_docs:
        vs.add_documents(
            documents=[d for d, _ in new_docs],
            ids=[i for _, i in new_docs],
        )

    _update_cache(ticker)
    return len(new_docs)