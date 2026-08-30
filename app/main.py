import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.routers import chat, quiz
from app.services.external.dart import load_corp_codes
from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger(__name__)

if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN



@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("서버 시작 중...")
    await load_corp_codes()
    logger.info("DART 종목 코드 로드 완료")
    await _warm_up()
    yield
    logger.info("서버 종료")


async def _warm_up() -> None:
    """임베딩 모델을 미리 올린다.

    bge-m3 는 첫 사용 시점에 지연 로딩되는데, 서버(GPU)에서 40초쯤 걸린다.
    그 비용을 첫 사용자의 요청이 뒤집어쓰면 RunPod HTTP 프록시의 80초 제한에
    걸려 500 이 나간다 (2026-08-30 리허설 실측: 서버는 86초에 정상 완료했는데
    클라이언트는 79.7초에 500 수신). 기동 시점에 미리 올려 그 창을 없앤다.

    실패해도 서버는 떠야 한다 - 첫 요청이 느려질 뿐 기능은 동작한다.
    """
    import asyncio

    try:
        from app.services.rag.vectorstore import get_embeddings

        started = asyncio.get_running_loop().time()
        embeddings = await asyncio.to_thread(get_embeddings)
        await asyncio.to_thread(embeddings.embed_query, "워밍업")
        elapsed = asyncio.get_running_loop().time() - started
        logger.info(f"임베딩 모델 워밍업 완료 ({elapsed:.1f}s, device={settings.EMBEDDING_DEVICE})")
    except Exception as e:
        logger.warning(f"임베딩 워밍업 실패 - 첫 요청이 느려질 수 있다: {e}")


app = FastAPI(title="StockMate AI", version="0.1.0", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware
from app.core.auth import APIKeyMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(APIKeyMiddleware)

app.include_router(chat.router)
app.include_router(quiz.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"처리되지 않은 에러: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "서버 내부 오류가 발생했습니다."}
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================================================
# OpenAPI 스키마에 인증 표기
# =========================================================
# 인증은 미들웨어(app/core/auth.py)로 처리하므로 FastAPI 가 자동으로
# 스키마에 넣어주지 않는다. docs/openapi.json 을 계약서로 쓰는
# NestJS 팀이 헤더 필요 여부를 알 수 있도록 직접 주입한다.
from fastapi.openapi.utils import get_openapi  # noqa: E402

_NO_AUTH_PATHS = {"/health"}


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=(
            "StockMate AI 마이크로서비스.\n\n"
            "**인증**: `/health` 를 제외한 모든 엔드포인트는 `X-API-Key` 헤더가 필요합니다.\n"
            "(`Authorization: Bearer <키>` 형식도 허용)\n\n"
            "서버의 `AI_API_KEY` 환경변수가 비어 있으면 인증이 비활성화됩니다(로컬 개발용)."
        ),
        routes=app.routes,
    )

    schema.setdefault("components", {})["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "팀 공유 API 키. 백엔드 .env 의 AI_API_KEY 값을 그대로 보낸다.",
        }
    }

    for path, operations in schema.get("paths", {}).items():
        if path in _NO_AUTH_PATHS:
            continue
        for operation in operations.values():
            if not isinstance(operation, dict):
                continue
            operation["security"] = [{"ApiKeyAuth": []}]
            operation.setdefault("responses", {})["401"] = {
                "description": "API 키가 없거나 일치하지 않음",
                "content": {
                    "application/json": {
                        "example": {"detail": "유효한 API 키가 필요합니다. X-API-Key 헤더를 확인하세요."}
                    }
                },
            }

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi
