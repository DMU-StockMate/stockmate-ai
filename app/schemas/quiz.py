from pydantic import BaseModel, Field
from typing import Optional
from app.schemas.chat import UserContext


class QuizGenerateRequest(BaseModel):
    user: UserContext
    quiz_type: str = Field(
        description="`OX` 또는 `MULTIPLE_CHOICE`. 이 경로에서는 지정한 유형으로만 생성한다.",
        examples=["MULTIPLE_CHOICE"],
    )
    topic: Optional[str] = Field(
        default=None,
        description=(
            "출제 주제. **반드시 짧은 주제어여야 한다** (`PER`, `배당수익률`, `부채비율`).\n\n"
            "⚠️ **문장이나 지시문을 넣지 말 것.** 생성된 문제가 이 문자열을 실제로 "
            "다루는지 검사하는 주제 이탈 가드가 있어서, `\"...문제를 만들어줘\"` 같은 "
            "긴 텍스트를 넣으면 매번 이탈 판정 → 재시도 3회 소진 → **500** 이 난다. "
            "생성 자체는 되는데 마지막에 실패하는 형태라 원인 파악이 어렵다.\n\n"
            "자유 프롬프트로 만들고 싶으면 `POST /quiz/generate/prompt` 를, "
            "오답 기반이면 `POST /quiz/generate/review` 를 쓸 것.\n\n"
            "생략하면 사용자 수준에 맞는 주제를 서버가 고른다."
        ),
        examples=["PER"],
    )
    count: int = Field(
        default=1,
        description=(
            "생성할 문제 수. 같은 `topic` 으로 count개를 만든다.\n\n"
            "**이 경로에서만 동작한다** (`/prompt`, `/review` 는 받지 않는다).\n"
            "count 를 키우는 것보다 **count=1 요청을 여러 번 병렬로 보내는 편이 빠르다** — "
            "GPU 총 작업량은 같은데 요청 하나하나가 짧아진다."
        ),
        examples=[1],
    )


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
    """자유 프롬프트로 문제를 만든다.

    ⚠️ **`quiz_type` 은 받지 않는다.** 보내도 무시되고, 문제 유형은 서버가
    프롬프트를 읽어서 정한다. `count` 는 받는다(아래 참고).
    """
    prompt: str = Field(
        description=(
            "사용자가 입력한 자연어 요청. 한국어 문장을 그대로 넣으면 된다.\n\n"
            "서버가 이 문장을 읽어 **주제 / 문제 유형 / 개수**를 뽑아낸다.\n\n"
            "- 개수: 문장에 `5개` 처럼 적혀 있으면 그 수, 없으면 **3개** (최대 10)\n"
            "- 유형: 문장에 OX/객관식 언급이 있으면 그대로, 없으면 서버가 판단\n"
            "- 주제: 사용자 `investment_level` 에 해당하는 카테고리 안에서 찾는다.\n"
            "  그 범위에 없는 주제면 **422** 가 난다 (예: 초급 사용자가 `당기순이익`)\n\n"
            "카테고리 설명 같은 참고 문맥을 앞에 붙여 보내도 되지만, "
            "**요청 문장이 문맥에 묻히지 않게** 짧게 유지하는 편이 주제 인식률이 높다."
        ),
        examples=["PER과 PBR의 차이를 확인하는 문제를 5개 만들어줘"],
    )
    user: UserContext
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description=(
            "만들 문제 수. **주면 프롬프트에서 읽은 개수보다 우선한다.**\n\n"
            "화면에서 사용자가 개수를 고르는 UI 가 있다면 그 값을 그대로 넣으면 된다. "
            "생략하면 프롬프트 문장에서 읽고, 문장에도 없으면 서버 기본값(3)을 쓴다.\n\n"
            "⚠️ **개수가 곧 대기 시간이다.** 1개는 약 40초, 3개는 약 60~105초다. "
            "앞단(nginx 등) 타임아웃이 짧다면 개수를 줄이는 것이 가장 확실한 완화책이다."
        ),
        examples=[1],
    )


class PromptQuizQuestion(QuizGenerateResponse):
    """생성된 문제 + DB 저장용 카테고리 매핑.

    category_code / detail_codes는 요청으로 받은 목록 안의 코드만 반환한다.
    NestJS는 코드로 category_id / detail_id를 조회해
    quiz_questions.category_id 와 quiz_question_details 에 저장하면 된다.
    primary_detail_code는 quiz_question_details.is_primary=TRUE 대상.
    """
    category_code: Optional[str] = Field(
        default=None,
        description=(
            "이 문제가 속한 카테고리 코드. 백엔드는 이 코드로 `quiz_categories.category_id` 를 "
            "조회해 `quiz_questions.category_id` 에 저장하면 된다."
        ),
        examples=["BEGINNER_FINANCIAL_BASIC"],
    )
    detail_codes: list[str] = Field(
        default=[],
        description=(
            "이 문제에 연결할 상세 코드 목록. `quiz_question_details` 에 행으로 저장한다."
        ),
        examples=[["PER_BASIC"]],
    )
    primary_detail_code: Optional[str] = Field(
        default=None,
        description=(
            "대표 상세 코드. `quiz_question_details.is_primary = TRUE` 로 저장할 대상이다.\n\n"
            "**이 값을 저장해 두면** 다음에 그 문제가 오답이 됐을 때 "
            "`/quiz/generate/review` 의 `detail_code` 로 그대로 넣을 수 있다."
        ),
        examples=["PER_BASIC"],
    )


class PromptQuizGenerateResponse(BaseModel):
    prompt: str
    questions: list[PromptQuizQuestion]


# =========================================================
# 오답 기반 문제 생성 (POST /quiz/generate/review)
# =========================================================

class WrongAnswerChoice(BaseModel):
    choice_no: int
    text: str
    is_correct: bool = False


class WrongAnswerItem(BaseModel):
    """유저가 틀린 문제 1개 (quiz_attempts + quiz_questions + quiz_choices 조인 결과).

    NestJS는 `WHERE user_id = ? AND is_correct = FALSE` 로 조회한 오답들을
    이 형태로 담아 보내면 된다.

    detail_code는 quiz_question_details(is_primary=TRUE) → quiz_category_details.detail_code.
    이 값이 있으면 주제를 정확히 알 수 있어 생성 품질이 크게 올라간다.
    없으면 topic 문자열로 대체하고, 둘 다 없으면 문제 본문에서 주제를 추론한다.
    """
    question_text: str = Field(description="유저가 틀린 문제의 본문.")
    question_type: str = Field(
        default="MULTIPLE_CHOICE", description="`OX` | `MULTIPLE_CHOICE`"
    )
    explanation: Optional[str] = Field(
        default=None, description="원래 문제의 해설. 있으면 오해 지점을 더 정확히 겨냥한다."
    )
    topic: Optional[str] = Field(
        default=None,
        description="원래 문제의 주제명. `detail_code` 가 없을 때의 대체 수단이다.",
        examples=["PER"],
    )
    detail_code: Optional[str] = Field(
        default=None,
        description=(
            "**카탈로그 상세 코드.** `quiz_question_details` 에서 `is_primary=TRUE` 인 행의 "
            "`quiz_category_details.detail_code` 를 넣는다.\n\n"
            "⚠️ **주제 이름이 아니라 코드다.** `PER` ❌ / `PER_BASIC` ⭕, "
            "`ROE` ❌ / `ROE_BASIC` ⭕. 코드 목록은 AI 서버의 "
            "`app/services/quiz/categories.py` 에 있고 DB `quiz_category_details` 와 같아야 한다.\n\n"
            "이 값이 맞으면 서버가 카탈로그에서 바로 역추적해 **LLM 매핑 호출 1회를 통째로 "
            "건너뛴다** (파드 실측 약 20초 단축).\n\n"
            "틀린 코드를 넣으면 오류는 안 나지만 조용히 무시되고 문제 본문에서 주제를 "
            "추론하게 되는데, 이때 엉뚱한 주제로 흘러 **500** 이 날 수 있다."
        ),
        examples=["PER_BASIC"],
    )
    choices: list[WrongAnswerChoice] = Field(
        default=[], description="원래 문제의 선택지 전체. `is_correct` 로 정답을 표시한다."
    )
    selected_choice_no: Optional[int] = Field(
        default=None,
        description=(
            "유저가 고른 (틀린) 선택지 번호. **이게 있어야 '무엇을 오해했는지'를 "
            "겨냥한 문제를 만들 수 있다.** 가능하면 꼭 채워 보낼 것."
        ),
        examples=[1],
    )


class ReviewQuizGenerateRequest(BaseModel):
    """오답 기록으로 복습 문제를 만든다.

    ⚠️ **`quiz_type` 은 받지 않는다.** 유형은 오답의 성격에 맞춰 서버가 정한다.
    `count` 는 받는다 — 생략하면 5문제다.
    """
    user: UserContext
    wrong_answers: list[WrongAnswerItem] = Field(
        description=(
            "유저가 틀린 문제들. `WHERE user_id = ? AND is_correct = FALSE` 로 조회한 결과를 "
            "이 형태로 담는다. 빈 배열이면 **422**.\n\n"
            "여러 개를 넣으면 서버가 주제별로 묶어 문제 수를 배분한다 "
            "(많이 틀린 주제가 더 많이 가져간다)."
        )
    )
    count: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description=(
            "만들 문제 수. 생략하면 **5개**다.\n\n"
            "⚠️ **개수가 곧 대기 시간이다.** 앞단 타임아웃이 짧다면 줄이는 것이 "
            "가장 확실한 완화책이다."
        ),
        examples=[1],
    )


class ReviewQuizGenerateResponse(BaseModel):
    """응답 형태는 PromptQuizQuestion과 동일하다.

    NestJS 입장에서 /quiz/generate/prompt 와 저장 로직을 그대로 공유할 수 있게
    의도적으로 같은 스키마를 사용한다 (category_code/detail_codes/primary_detail_code).
    """
    questions: list[PromptQuizQuestion]