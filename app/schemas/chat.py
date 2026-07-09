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
    # NestJS가 category/level을 null로 보내는 경우가 있어 Optional로 둔다.
    # (필드가 str이고 null이 오면 Pydantic이 422로 요청 전체를 거부해버림)
    level: Optional[str] = None
    category: Optional[str] = None  # "concept" | "metric" | "news" | "disclosure"
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