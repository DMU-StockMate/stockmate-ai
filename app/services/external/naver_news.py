import httpx
from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger(__name__)

NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"

async def search_news(query: str, display: int = 10) -> list[dict]:
    headers = {
        "X-Naver-Client-Id": settings.NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": settings.NAVER_CLIENT_SECRET,
    }
    params = {
        "query": query,
        "display": display,
        "sort": "date",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(NAVER_NEWS_URL, headers=headers, params=params)
            response.raise_for_status()
            items = response.json().get("items", [])
            logger.info(f"Naver News [{query}]: {len(items)}건 조회")
            return [
                {
                    "title": item["title"],
                    "description": item["description"],
                    "pub_date": item["pubDate"],
                    "link": item["link"],
                }
                for item in items
            ]
    except httpx.TimeoutException:
        logger.error(f"Naver News API 타임아웃 [{query}]")
        return []
    except httpx.HTTPStatusError as e:
        logger.error(f"Naver News API 에러 [{query}]: {e.response.status_code}")
        return []
    except Exception as e:
        logger.error(f"Naver News API 예외 [{query}]: {e}")
        return []