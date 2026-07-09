from pydantic import BaseModel
from typing import Optional
from app.schemas.chat import UserContext


class QuizGenerateRequest(BaseModel):
    user: UserContext
    quiz_type: str  # "OX" or "MULTIPLE_CHOICE"
    topic: Optional[str] = None  # 없으면 랜덤
    count: int = 1  # 한 번에 생성할 문제 개수 (같은 topic으로 count개, 순차 생성)


class QuizChoice(BaseModel):
    choice_no: int
    text: str
    is_correct: bool


class QuizGenerateResponse(BaseModel):
    question_type: str
    question_text: str
    choices: list[QuizChoice]
    explanation: str
    topic: str
    level: str


class QuizGenerateBatchResponse(BaseModel):
    questions: list[QuizGenerateResponse]