from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    tickers: list[str]
    answer: str
    ingested: dict