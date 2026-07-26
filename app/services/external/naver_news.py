import html
import re
import httpx
from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger(__name__)

NAVER_NEWS_URL = "https://openapi.naver.com/v1/search/news.json"

_TAG_PATTERN = re.compile(r"<[^>]+>")


def _clean_html(text: str) -> str:
    """네이버 뉴스 title/description에 섞여오는 <b> 태그와 &quot; 같은 HTML 엔티티를 제거한다.

    태그/엔티티가 그대로 임베딩·프롬프트 컨텍스트에 저장되면 검색 품질과 답변 가독성을
    떨어뜨리므로 적재 전에 정리한다.
    """
    if not text:
        return ""
    return html.unescape(_TAG_PATTERN.sub("", text)).strip()

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
                    "title": _clean_html(item["title"]),
                    "description": _clean_html(item["description"]),
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