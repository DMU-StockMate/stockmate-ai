from pydantic import BaseModel

class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str

class AskRequest(BaseModel):
    question: str
    history: list[Message] = []

class AskResponse(BaseModel):
    question: str
    tickers: list[str]
    answer: str
    ingested: dict