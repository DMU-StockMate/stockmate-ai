import hashlib
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from langchain_core.documents import Document
from app.services.rag.vectorstore import get_vectorstore
from app.services.external.naver_news import search_news
from app.services.external.dart import get_disclosures
import uuid

import logging
logger = logging.getLogger(__name__)

_last_fetched_news: dict[str, datetime] = {}
_last_fetched_dart: dict[str, datetime] = {}
CACHE_MINUTES = 30


def _is_news_cache_valid(ticker: str) -> bool:
    if ticker not in _last_fetched_news:
        return False
    return datetime.now() - _last_fetched_news[ticker] < timedelta(minutes=CACHE_MINUTES)


def _is_dart_cache_valid(ticker: str) -> bool:
    if ticker not in _last_fetched_dart:
        return False
    return datetime.now() - _last_fetched_dart[ticker] < timedelta(minutes=CACHE_MINUTES)


def _update_news_cache(ticker: str) -> None:
    _last_fetched_news[ticker] = datetime.now()


def _update_dart_cache(ticker: str) -> None:
    _last_fetched_dart[ticker] = datetime.now()


def _make_id(text: str) -> str:
    """UUID 형식으로 ID 생성"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, text))


def _parse_pub_date(pub_date: str) -> int:
    dt = parsedate_to_datetime(pub_date)
    return int(dt.strftime("%Y%m%d"))

async def ingest_news(ticker: str, display: int = 10) -> int:
    if _is_news_cache_valid(ticker):
        return 0

    items = await search_news(ticker, display=display)
    if not items:
        return 0

    vs = get_vectorstore()
    docs, ids = [], []

    for item in items:
        content = f"{item['title']}\n{item['description']}"
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, item["link"]))
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
        vs.add_documents(documents=new_docs, ids=new_ids)

    _update_news_cache(ticker)
    return len(new_docs)


async def ingest_disclosures(ticker: str, days: int = 90) -> int:
    if _is_dart_cache_valid(ticker):
        return 0

    items = await get_disclosures(ticker, days=days)
    if not items:
        return 0

    vs = get_vectorstore()
    docs, ids = [], []

    for item in items:
        content = f"{item['corp_name']} 공시: {item['title']}\n날짜: {item['date']}"
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, item["url"]))
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

    # 기존 ID 조회해서 새것만 추가
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
        vs.add_documents(documents=new_docs, ids=new_ids)

    _update_dart_cache(ticker)
    return len(new_docs)