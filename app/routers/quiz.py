from fastapi import APIRouter, HTTPException
from app.schemas.quiz import QuizGenerateRequest, QuizGenerateResponse, QuizGenerateBatchResponse
from app.services.quiz.generator import generate_quiz_batch

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
    try:
        results = await generate_quiz_batch(
            user=req.user,
            quiz_type=req.quiz_type,
            topic=req.topic,
            count=req.count,
        )
        return QuizGenerateBatchResponse(questions=[QuizGenerateResponse(**r) for r in results])
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))