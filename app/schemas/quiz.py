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


# =========================================================
# 프롬프트 기반 문제 생성 (POST /quiz/generate/prompt)
# =========================================================

class QuizCategoryDetailIn(BaseModel):
    """quiz_category_details 테이블의 한 행 (내부 카탈로그용 모델).

    카탈로그 데이터는 app/services/quiz/categories.py에 하드코딩되어 있다.
    """
    detail_code: str
    detail_name: str
    description: Optional[str] = None


class QuizCategoryIn(BaseModel):
    """quiz_categories 테이블의 한 행 + 소속 상세 목록 (내부 카탈로그용 모델)."""
    category_code: str
    category_name: str
    description: Optional[str] = None
    problem_direction: Optional[str] = None
    investment_level: Optional[str] = None  # "초급" | "중급" | "고급"
    details: list[QuizCategoryDetailIn] = []


class PromptQuizGenerateRequest(BaseModel):
    prompt: str
    user: UserContext


class PromptQuizQuestion(QuizGenerateResponse):
    """생성된 문제 + DB 저장용 카테고리 매핑.

    category_code / detail_codes는 요청으로 받은 목록 안의 코드만 반환한다.
    NestJS는 코드로 category_id / detail_id를 조회해
    quiz_questions.category_id 와 quiz_question_details 에 저장하면 된다.
    primary_detail_code는 quiz_question_details.is_primary=TRUE 대상.
    """
    category_code: Optional[str] = None
    detail_codes: list[str] = []
    primary_detail_code: Optional[str] = None


class PromptQuizGenerateResponse(BaseModel):
    prompt: str
    questions: list[PromptQuizQuestion]