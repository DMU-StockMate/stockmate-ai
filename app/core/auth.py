"""API 키 인증 미들웨어.

FastAPI 를 외부(포트포워딩/공인IP)로 열 때 무단 호출을 막는다.
인증이 없으면 누구나 /chat/stream 을 호출해 GPU 와 외부 API 쿼터
(KIS/DART/네이버)를 소모시킬 수 있다.

동작
    settings.AI_API_KEY 가 비어 있으면 인증을 끈다 (로컬 개발용 기본값).
    값이 있으면 모든 요청에 다음 중 하나를 요구한다.
        X-API-Key: <키>
        Authorization: Bearer <키>

예외 경로
    /health, /docs, /redoc, /openapi.json 과 CORS 프리플라이트(OPTIONS).
"""
import secrets

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger(__name__)

EXEMPT_PATHS = {"/health", "/docs", "/redoc", "/openapi.json"}


def _extract_key(request: Request) -> str:
    key = request.headers.get("x-api-key")
    if key:
        return key
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        expected = settings.AI_API_KEY

        # 키가 설정되지 않았으면 통과 (로컬 개발)
        if not expected:
            return await call_next(request)

        # 프리플라이트와 공개 경로는 통과
        if request.method == "OPTIONS" or request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        provided = _extract_key(request)
        # compare_digest 로 타이밍 공격 방지
        if not provided or not secrets.compare_digest(provided, expected):
            logger.warning(
                "인증 실패: %s %s from %s",
                request.method, request.url.path,
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "유효한 API 키가 필요합니다. X-API-Key 헤더를 확인하세요."},
            )

        return await call_next(request)