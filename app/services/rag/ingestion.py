import hashlib
from email.utils import parsedate_to_datetime
from langchain_core.documents import Document
from app.services.rag.vectorstore import get_vectorstore
from app.services.external.naver_news import search_news
from app.services.external.dart import get_disclosures


def _make_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _parse_pub_date(pub_date: str) -> int:
    """'Wed, 13 May 2026 23:30:00 +0900' → 20260513"""
    dt = parsedate_to_datetime(pub_date)
    return int(dt.strftime("%Y%m%d"))


async def ingest_news(ticker: str, display: int = 10) -> int:
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

    return len(new_docs)