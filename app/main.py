from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers import chat, quiz, stock
from app.services.external.dart import load_corp_codes


@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_corp_codes()
    yield


app = FastAPI(title="StockMate AI", version="0.1.0", lifespan=lifespan)

app.include_router(chat.router)
app.include_router(quiz.router)
app.include_router(stock.router)


@app.get("/health")
def health():
    return {"status": "ok"}