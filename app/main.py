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
    yield
    logger.info("서버 종료")


app = FastAPI(title="StockMate AI", version="0.1.0", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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