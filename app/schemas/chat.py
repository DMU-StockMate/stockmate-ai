from pydantic import BaseModel, Field
from typing import Optional


class Message(BaseModel):
    role: str
    content: str


class UserContext(BaseModel):
    user_id: int = Field(
        description="사용자 ID. 생성된 문제를 이 사용자 앞으로 기록하는 데 쓴다.",
        examples=[1],
    )
    investment_level: str = Field(
        default="미설정",
        description=(
            "사용자 투자 등급. **아래 다섯 값 중 하나여야 한다.**\n\n"
            "`입문` | `초급` | `중급` | `고급` | `미설정`\n\n"
            "문제 난이도와, 프롬프트 퀴즈에서 주제를 찾을 카테고리 범위를 결정한다. "
            "다른 문자열(`unset`, `BEGINNER` 등)을 보내면 오류는 나지 않지만 "
            "**전체 카탈로그를 뒤지게 되어 주제 매핑 품질이 떨어진다.** "
            "DB 의 등급명이 이 다섯 값과 다르면 백엔드에서 변환해 보낼 것."
        ),
        examples=["초급"],
    )


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