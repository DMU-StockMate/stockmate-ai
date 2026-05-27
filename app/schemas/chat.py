from pydantic import BaseModel
from typing import Optional


class Message(BaseModel):
    role: str
    content: str


class UserContext(BaseModel):
    user_id: int
    investment_level: str = "미설정"


class Choice(BaseModel):
    choice_no: int
    text: str
    is_correct: bool


class QuizContext(BaseModel):
    question_id: int
    question_type: str  # "OX", "MULTIPLE_CHOICE"
    question_text: str
    explanation: Optional[str] = None
    topic: str = "미분류"
    level: str = "미설정"
    choices: list[Choice] = []


# 백엔드에서 호출하는 통합 엔드포인트용
class ChatStreamRequest(BaseModel):
    question: str
    history: list[Message] = []
    user: Optional[UserContext] = None
    quiz_context: Optional[QuizContext] = None


# 기존 프론트 직접 호출용 (유지)
class AskRequest(BaseModel):
    question: str
    history: list[Message] = []


class AskResponse(BaseModel):
    question: str
    tickers: list[str]
    answer: str
    ingested: dict