from fastapi import APIRouter, HTTPException
from app.schemas.quiz import (
    QuizGenerateRequest, QuizGenerateResponse, QuizGenerateBatchResponse,
    PromptQuizGenerateRequest, PromptQuizGenerateResponse, PromptQuizQuestion,
    ReviewQuizGenerateRequest, ReviewQuizGenerateResponse,
)
from app.services.quiz.generator import (
    generate_quiz_batch, generate_quiz_from_prompt, PromptTopicError,
)
from app.services.quiz.review import (
    generate_quiz_from_wrong_answers, NoWrongAnswerError, REVIEW_QUIZ_COUNT,
)
from app.core.keepalive import json_with_keepalive

router = APIRouter(prefix="/quiz", tags=["quiz"])


@router.post(
    "/generate",
    response_model=QuizGenerateBatchResponse,
    summary="AI 퀴즈 문제 생성",
    description="""
LLM을 사용해 사용자 수준에 맞는 주식 투자 퀴즈 문제를 생성합니다.

**지원 문제 유형**

OX 문제 (quiz_type: "OX")
- O/X 2개 선택지
- choice_no 1 = O, choice_no 2 = X

객관식 4지선다 (quiz_type: "MULTIPLE_CHOICE")
- 4개 선택지
- choice_no 1~4

**investment_level별 난이도**
- 입문: 주식, 배당 등 아주 기초 개념
- 초급: PER, PBR, ROE 등 기본 지표
- 중급: 재무제표 해석, 투자 전략
- 고급: 심화 분석, 고급 전략

**topic 미입력시** 수준에 맞는 주제 자동 선택

**count** (기본값 1)
- 같은 topic으로 count개 문제를 한 번에 생성합니다.
- 응답은 항상 `questions` 배열입니다 (count=1이어도 배열 안에 1개).
- 순차 생성이라 count가 클수록 응답 시간이 비례해서 늘어납니다.

**주의사항**
- LLM이 생성하므로 문제 1개당 최대 3번 재시도
- 재시도 후에도 실패시 500 에러 반환
    """,
    responses={
        200: {
            "description": "문제 생성 성공",
            "content": {
                "application/json": {
                    "examples": {
                        "OX": {
                            "summary": "OX 문제 예시 (count=1)",
                            "value": {
                                "questions": [
                                    {
                                        "question_type": "OX",
                                        "question_text": "PER이 낮을수록 주식이 저평가되어 있다고 볼 수 있다.",
                                        "choices": [
                                            {"choice_no": 1, "text": "O", "is_correct": False},
                                            {"choice_no": 2, "text": "X", "is_correct": True}
                                        ],
                                        "explanation": "PER은 업종별로 달리 해석해야 합니다.",
                                        "topic": "PER",
                                        "level": "초급"
                                    }
                                ]
                            }
                        },
                        "MULTIPLE_CHOICE": {
                            "summary": "객관식 문제 예시 (count=1)",
                            "value": {
                                "questions": [
                                    {
                                        "question_type": "MULTIPLE_CHOICE",
                                        "question_text": "PER을 구하는 공식은?",
                                        "choices": [
                                            {"choice_no": 1, "text": "주가 / 주당순이익", "is_correct": True},
                                            {"choice_no": 2, "text": "주가 / 주당순자산", "is_correct": False},
                                            {"choice_no": 3, "text": "순이익 / 자기자본", "is_correct": False},
                                            {"choice_no": 4, "text": "주가 / 매출액", "is_correct": False}
                                        ],
                                        "explanation": "PER = 주가 / EPS(주당순이익)입니다.",
                                        "topic": "PER",
                                        "level": "초급"
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        },
        500: {"description": "문제 생성 실패 (LLM 재시도 초과)"}
    }
)
async def generate(req: QuizGenerateRequest):
    async def build():
        results = await generate_quiz_batch(
            user=req.user,
            quiz_type=req.quiz_type,
            topic=req.topic,
            count=req.count,
        )
        return QuizGenerateBatchResponse(questions=[QuizGenerateResponse(**r) for r in results])

    try:
        return await json_with_keepalive(build(), label="/quiz/generate")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/generate/prompt",
    response_model=PromptQuizGenerateResponse,
    summary="프롬프트 기반 AI 퀴즈 문제 생성",
    description="""
사용자가 입력한 자유 프롬프트를 AI가 분석해 주제를 파악하고 퀴즈 문제를 생성합니다.

**동작 방식 (2단계)**

1. 분석: 프롬프트에서 출제 주제·문제 개수·문제 유형을 파악하고,
   요청으로 받은 카테고리/상세 코드 목록에 매핑한 출제 계획을 세웁니다.
2. 생성: 계획의 각 항목을 **동시에** 실제 문제로 생성합니다.

**문제 개수**
- 프롬프트에 개수가 명시되면 그 개수로 생성 (예: "PER 문제 3개 만들어줘" → 3개)
- 명시가 없으면 기본 3개 (서버 설정 `PROMPT_QUIZ_DEFAULT_COUNT`)
- 최대 10개 (초과 요청 시 10개로 제한)

⚠️ **개수가 많으면 프록시 타임아웃 위험이 있습니다.**
이 경로는 LLM을 `1(분석) + N(생성) + 1(검증)`번 호출하고, 생성은 서버 설정
(`QUIZ_GEN_CONCURRENCY`, 기본 5)만큼씩 나눠서 동시에 돕니다. 파드 실측 기준:

| 문제 수 | 라운드 | 실측/추정 |
|---|---|---|
| 3개 | 3 | 약 60초 |
| 5개 | 3 | **84.9초 (실측)** |
| 10개 | 4 (생성 2파도) | 약 88초 |

RunPod HTTP 프록시(Cloudflare)는 **100초에서 524로 끊습니다**(2026-09-05 실측).
사용자가 고를 수 있는 개수를 **5 이하로 제한**하는 것을 권장합니다.

**여러 주제**
- 프롬프트에 여러 주제가 있으면 주제별로 고르게 나눠 생성합니다.
  (예: "PER이랑 분산투자 문제 내줘" → PER/분산투자 문제 혼합)

**문제 유형**
- 프롬프트에서 파악: "OX로 내줘" → OX, 언급 없으면 전부 객관식(4지선다)
- "섞어서 내줘" 같은 요청이면 문제별로 유형이 다를 수 있습니다.

**카테고리 등급 필터링**
- 카테고리 매핑 시 사용자의 investment_level과 같은 등급의 카테고리를 우선 사용합니다.
  (초급 유저의 문제가 불필요하게 중급/고급 카테고리로 분류되지 않도록)
- 입문 유저는 초급 카테고리를 사용합니다 (DB에 입문용 카테고리가 없음).
- 미설정 유저는 전체 카테고리를 사용합니다.
- 유저 등급에 없는 주제도 문제 생성이 가능합니다: 예를 들어 초급 유저가
  "볼린저밴드 문제"를 요청하면, 해당 주제만 전체 카테고리 대상으로 재매핑해
  다른 등급의 category_code(예: INTERMEDIATE_INDICATOR_CONCEPT)로 반환합니다.
  이때 문제 난이도는 여전히 사용자 수준(초급)에 맞춰 조절됩니다.

**카테고리 매핑 (응답 필드)**
- 문제마다 `category_code`(1개)와 `detail_codes`(관련 상세 코드 목록)를 반환합니다.
- `primary_detail_code`: 대표 상세 코드 (`quiz_question_details.is_primary=TRUE` 저장용)
- 카테고리/상세 코드 목록은 AI 서버에 내장되어 있습니다 (DB `quiz_categories` /
  `quiz_category_details` 초기 데이터의 미러 — `app/services/quiz/categories.py`).
  ⚠️ DB에서 카테고리를 추가/수정하면 이 파일도 같이 수정해야 합니다.
- 내장 목록에 존재하는 코드만 반환합니다 (AI가 지어낸 코드는 검증 단계에서 제거).
- 적합한 카테고리가 없으면 `category_code`는 null, `detail_codes`는 빈 배열일 수 있습니다.
- NestJS는 코드로 category_id/detail_id를 조회해 `quiz_questions`,
  `quiz_question_details`, `quiz_choices`에 저장하면 됩니다.

**주의사항**
- LLM 생성 특성상 문제 1개당 최대 3회 재시도, 실패 시 500 에러
- 순차 생성이라 응답 시간은 문제 개수에 비례합니다 (1개당 수 초)
""",
    responses={
        200: {
            "description": "문제 생성 성공",
            "content": {
                "application/json": {
                    "example": {
                        "prompt": "PER이랑 분산투자에 대한 문제 4개 내줘",
                        "questions": [
                            {
                                "question_type": "MULTIPLE_CHOICE",
                                "question_text": "PER을 구하는 공식은?",
                                "choices": [
                                    {"choice_no": 1, "text": "주가 / 주당순이익", "is_correct": True},
                                    {"choice_no": 2, "text": "주가 / 주당순자산", "is_correct": False},
                                    {"choice_no": 3, "text": "순이익 / 자기자본", "is_correct": False},
                                    {"choice_no": 4, "text": "주가 / 매출액", "is_correct": False},
                                ],
                                "explanation": "PER = 주가 / EPS(주당순이익)입니다.",
                                "topic": "PER",
                                "level": "초급",
                                "category_code": "BEGINNER_FINANCIAL_BASIC",
                                "detail_codes": ["PER_BASIC"],
                                "primary_detail_code": "PER_BASIC",
                            },
                            {
                                "question_type": "MULTIPLE_CHOICE",
                                "question_text": "분산투자의 주된 목적은?",
                                "choices": [
                                    {"choice_no": 1, "text": "투자 위험을 줄이기 위해", "is_correct": True},
                                    {"choice_no": 2, "text": "수수료를 아끼기 위해", "is_correct": False},
                                    {"choice_no": 3, "text": "세금을 줄이기 위해", "is_correct": False},
                                    {"choice_no": 4, "text": "거래량을 늘리기 위해", "is_correct": False},
                                ],
                                "explanation": "여러 자산에 나누어 투자하면 특정 자산의 손실 위험을 줄일 수 있습니다.",
                                "topic": "분산투자",
                                "level": "초급",
                                "category_code": "BEGINNER_RISK_BASIC",
                                "detail_codes": ["DIVERSIFICATION"],
                                "primary_detail_code": "DIVERSIFICATION",
                            },
                        ],
                    }
                }
            },
        },
        422: {
            "description": "프롬프트에서 퀴즈 주제를 파악할 수 없음 (주식/투자와 무관한 프롬프트 등)",
            "content": {
                "application/json": {
                    "example": {"detail": "프롬프트에서 퀴즈 주제를 파악할 수 없습니다."}
                }
            },
        },
        500: {"description": "문제 생성 실패 (LLM 재시도 초과)"},
    },
)
async def generate_from_prompt(req: PromptQuizGenerateRequest):
    async def build():
        results = await generate_quiz_from_prompt(
            prompt=req.prompt,
            user=req.user,
        )
        return PromptQuizGenerateResponse(
            prompt=req.prompt,
            questions=[PromptQuizQuestion(**r) for r in results],
        )

    try:
        return await json_with_keepalive(build(), label="/quiz/generate/prompt")
    except PromptTopicError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/generate/review",
    response_model=ReviewQuizGenerateResponse,
    summary="오답 기반 AI 퀴즈 문제 생성",
    description=f"""
사용자가 그동안 틀린 문제들을 받아, 자주 틀린 주제 위주로 새 문제 {REVIEW_QUIZ_COUNT}개를 생성합니다.

**동작 방식**

1. 받은 오답들을 주제별로 묶습니다 (`detail_code` 우선, 없으면 `topic`).
2. 많이 틀린 주제 순으로 정렬한 뒤 {REVIEW_QUIZ_COUNT}개 슬롯을 배분합니다.
   많이 틀린 주제가 더 많은 문제를 가져갑니다 (주제 2개면 3:2).
3. 슬롯마다 해당 주제의 실제 오답을 프롬프트에 넣어 문제를 생성합니다.
   같은 개념은 유지하고 종목·수치·상황만 바꾸며, 유저가 골랐던 오답의
   오해를 선택지로 재배치하고 해설에서 짚어줍니다.
4. 생성된 문제 전체를 한 번에 검증합니다 (정답/해설 모순, 지표 정의 오류,
   OX 이중부정 등). 불합격 문제만 같은 조건으로 1회 재생성합니다.

**같은 주제에 여러 문제가 배정될 때**

슬롯마다 출제 관점을 강제로 돌립니다 (정의·계산 → 흔한 오해 → 함께 볼 지표
→ 실제 적용 → 예외 상황). 같은 주제에서 사실상 같은 문제가 반복되는 것을 막습니다.

**문제 개수**
- 항상 {REVIEW_QUIZ_COUNT}개 고정입니다 (요청으로 조절 불가).

**NestJS가 보낼 데이터**

`quiz_attempts`에서 `WHERE user_id = ? AND is_correct = FALSE`로 조회한 오답을
`quiz_questions` / `quiz_choices`와 조인해서 `wrong_answers` 배열에 담아주세요.

- `detail_code` (권장): `quiz_question_details`에서 `is_primary = TRUE`인 행의
  `quiz_category_details.detail_code`. 주제를 정확히 알 수 있어 품질이 크게 올라갑니다.
  없으면 `topic` 문자열로 대체되고, 둘 다 없으면 문제 본문에서 추론합니다.
- `selected_choice_no` (권장): 유저가 고른 오답 번호.
  이게 있어야 "무엇을 오해했는지"를 겨냥한 문제가 나옵니다.
  없으면 그냥 같은 주제의 평범한 문제가 생성됩니다.
- 오답 개수 제한은 없지만, 주제당 최대 2개까지만 프롬프트에 사용합니다.

**문제 유형**
- 주제별로 틀린 문제의 유형을 다수결로 따릅니다.
  (OX만 틀린 주제는 OX로, 그 외에는 객관식 4지선다)

**응답 형태**
- `/quiz/generate/prompt`와 **완전히 동일한 스키마**입니다.
  NestJS는 기존 저장 로직(`category_code`/`detail_codes`/`primary_detail_code` →
  `quiz_questions`, `quiz_question_details`, `quiz_choices`)을 그대로 재사용하면 됩니다.
- 저장 시 `source_type='WRONG_ANSWER'`, `visibility='PRIVATE'`,
  `owner_user_id`=요청 유저로 넣으면 됩니다.
- ⚠️ AI 생성 문제에는 `reward_cash`를 넣지 마세요. 보상 현금이 모의투자 총자산 →
  랭킹 수익률로 이어져서, 문제를 반복 생성하면 거래 없이 수익률을 올릴 수 있습니다.

**주의사항**
- LLM 생성 특성상 문제 1개당 최대 3회 재시도, 실패 시 500 에러
- 순차 생성이라 응답 시간이 깁니다. LLM 호출은 최소 {REVIEW_QUIZ_COUNT + 1}회
  (생성 {REVIEW_QUIZ_COUNT} + 검증 1)이며, 중복·검증 불합격 시 더 늘어납니다.
  `detail_code`를 보내면 카테고리 매핑 호출 1회를 아낄 수 있습니다.
  프론트에 로딩 처리가 필요합니다.
""",
    responses={
        200: {
            "description": "문제 생성 성공",
            "content": {
                "application/json": {
                    "example": {
                        "questions": [
                            {
                                "question_type": "MULTIPLE_CHOICE",
                                "question_text": "업종 평균보다 PER이 낮은 기업을 평가할 때 함께 확인해야 하는 것은?",
                                "choices": [
                                    {"choice_no": 1, "text": "PER이 낮으면 무조건 저평가이므로 확인할 필요가 없다", "is_correct": False},
                                    {"choice_no": 2, "text": "성장성과 일회성 이익 여부", "is_correct": True},
                                    {"choice_no": 3, "text": "거래량이 많은지 여부", "is_correct": False},
                                    {"choice_no": 4, "text": "상장 시기가 언제인지", "is_correct": False},
                                ],
                                "explanation": "PER이 낮다는 것만으로 저평가라고 볼 수 없습니다. 성장성이 낮거나 일회성 이익으로 순이익이 부풀려진 경우에도 PER은 낮게 나옵니다.",
                                "topic": "PER",
                                "level": "초급",
                                "category_code": "BEGINNER_FINANCIAL_BASIC",
                                "detail_codes": ["PER_BASIC"],
                                "primary_detail_code": "PER_BASIC",
                            }
                        ]
                    }
                }
            },
        },
        422: {
            "description": "오답 데이터가 비어 있음",
            "content": {
                "application/json": {
                    "example": {"detail": "오답 데이터가 없습니다."}
                }
            },
        },
        500: {"description": "문제 생성 실패 (LLM 재시도 초과)"},
    },
)
async def generate_from_wrong_answers(req: ReviewQuizGenerateRequest):
    async def build():
        results = await generate_quiz_from_wrong_answers(
            user=req.user,
            wrong_answers=req.wrong_answers,
        )
        return ReviewQuizGenerateResponse(
            questions=[PromptQuizQuestion(**r) for r in results],
        )

    try:
        return await json_with_keepalive(build(), label="/quiz/generate/review")
    except NoWrongAnswerError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))