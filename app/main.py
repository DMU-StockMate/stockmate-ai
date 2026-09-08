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


API_DESCRIPTION = """
백엔드(NestJS)가 호출하는 AI 마이크로서비스입니다.
**연동 전에 아래 4가지를 먼저 읽어 주세요. 실제로 사고가 났던 부분입니다.**

---

### 1. 인증

모든 요청에 `X-API-Key` 헤더가 필요합니다.
예외: `/health`, `/docs`, `/redoc`, `/openapi.json` (키 없이 열림)

```
X-API-Key: <AI_API_KEY>
```

키가 없으면 **401** 이고, 이때는 문제 생성이 아예 시작되지 않습니다.
`Authorization: Bearer <키>` 형식도 허용합니다.
서버의 `AI_API_KEY` 환경변수가 비어 있으면 인증이 비활성화됩니다(로컬 개발용).

### 2. 응답이 오래 걸립니다 — 그리고 200 이어도 실패일 수 있습니다

문제 생성은 **30초~3분**이 걸립니다. RunPod 프록시(Cloudflare)가
응답 첫 바이트까지 100초만 기다렸다가 `HTTP 524` 로 끊기 때문에,
서버는 **30초가 지나면 응답을 먼저 열고 10초마다 공백을 흘려보낸 뒤**
마지막에 본문을 채웁니다.

- JSON 값 앞의 공백은 문법상 무시되므로(RFC 8259) **파싱 코드는 그대로 두면 됩니다.**
  `JSON.parse`, axios 기본 동작 모두 정상 동작합니다.
- 다만 **헤더를 먼저 보낸 뒤에는 상태 코드를 바꿀 수 없습니다.**
  그래서 30초가 지난 뒤에 생성이 실패하면 500 이 아니라
  **200 에 실패 본문**이 옵니다.

```jsonc
// 성공
{ "questions": [ ... ] }

// 실패 (30초 넘긴 뒤 실패한 경우)
{ "detail": "묶음 문제 생성 실패: ...", "error_type": "ValueError", "keepalive_error": true }
```

**`questions` 키의 유무로 판정하세요.** 30초 안에 난 실패는 예전 그대로
422 / 500 로 옵니다.

```ts
if (!res.data?.questions) {
  throw new InternalServerErrorException(res.data?.detail ?? 'AI 문제 생성 실패');
}
```

### 3. 재시도하지 마세요

- **500**: 서버가 이미 LLM 을 3번 호출하고 실패한 결과입니다.
- **524**: 서버는 그 요청을 **계속 처리 중**입니다. 재시도하면 GPU 를 두 배로 씁니다.

둘 다 사용자에게 **"다시 시도" 버튼**을 주는 쪽이 맞습니다.

### 4. 타임아웃은 넉넉히

클라이언트 타임아웃은 **180초 이상**으로 두세요. 동시 사용자가 몰리면
프롬프트 퀴즈가 300초를 넘길 수도 있습니다. 앞단에 nginx 등 리버스 프록시가
있다면 **거기 타임아웃도 같이 늘려야 합니다** (기본 60초에서 잘립니다).

| 엔드포인트 | 파드 실측 |
|---|---|
| `POST /quiz/generate` count=1 | 18~30초 |
| `POST /quiz/generate` count=5 | 43~62초 |
| `POST /quiz/generate/prompt` | 60~140초 |
| `POST /quiz/generate/review` | 33~50초 |
| `POST /chat/stream` 첫 글자 | 2.4초 |
| `POST /chat/evaluate` | 약 35초 |
"""

app = FastAPI(
    title="StockMate AI",
    version="0.1.0",
    description=API_DESCRIPTION,
    lifespan=lifespan,
)

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
        # app.description(= API_DESCRIPTION)을 그대로 쓴다. 예전에는 여기에
        # 설명을 따로 하드코딩해서, FastAPI() 에 넣은 설명이 조용히 버려졌다.
        description=app.description,
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
