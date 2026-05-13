import httpx
from app.core.config import settings

NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"

async def search_news(query: str, display: int = 10, days: int = 7) -> list[dict]:
    headers = {
        "X-Naver-Client-Id": settings.NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": settings.NAVER_CLIENT_SECRET,
    }
    params = {
        "query": query,
        "display": display,
        "sort": "date",
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(NAVER_NEWS_URL, headers=headers, params=params)
        response.raise_for_status()
        items = response.json().get("items", [])

    return [
        {
            "title": item["title"],
            "description": item["description"],
            "pub_date": item["pubDate"],
            "link": item["link"],
        }
        for item in items
    ]