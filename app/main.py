from fastapi import FastAPI
from app.routers import chat

app = FastAPI(title="StockMate AI", version="0.1.0")

app.include_router(chat.router)

@app.get("/health")
def health():
    return {"status": "ok"}