import json
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
from app.schemas.chat import UserContext

LEVEL_GUIDE = {
    "입문": "주식 투자를 막 시작한 사람. 아주 기초적인 용어 (주식, 배당, 시가총액 등) 수준. 문제는 짧고 단순하게.",
    "초급": "주식 기본 개념을 아는 사람. PER, PBR, ROE 같은 기본 지표 수준. 문제는 간단한 계산이나 개념 이해 수준.",
    "중급": "투자 경험이 있는 사람. 재무제표 기본 해석, 투자 전략 수준. 문제는 실제 투자 상황에 적용하는 수준.",
    "고급": "전문 투자자. 심화 재무 분석, 파생상품, 고급 전략 수준. 문제는 복잡한 분석이 필요한 수준.",
    "미설정": "주식 기본 개념을 아는 사람. 문제는 간단하게.",
}

OX_TOPICS = [
    "PER", "PBR", "ROE", "EPS", "배당", "시가총액",
    "코스피", "코스닥", "ETF", "공매도", "분산투자",
    "손절매", "익절매", "포트폴리오", "리스크 관리",
]

# _get_random_topic()에서 MULTIPLE_CHOICE용으로 참조하는데 정의가 누락돼 있었음
# (topic 없이 객관식 문제 생성 요청하면 NameError로 500 에러가 나던 버그)
MC_TOPICS = [
    "PER", "PBR", "ROE", "EPS", "배당수익률", "시가총액",
    "코스피", "코스닥", "ETF", "공매도", "분산투자",
    "손절매", "익절매", "포트폴리오", "리스크 관리",
    "재무제표", "영업이익", "당기순이익", "부채비율", "유상증자",
]

MC_PROMPT = ChatPromptTemplate.from_template("""
당신은 주식 투자 교육 전문가입니다.
아래 조건에 맞는 4지선다 객관식 문제를 1개 생성하세요.

조건:
- 주제: {topic}
- 난이도: {level} ({level_guide})
- 문제는 반드시 한 문장으로 짧고 명확하게 작성하세요.
- 복잡한 수치나 여러 개념을 동시에 묻는 문제는 절대 만들지 마세요.
- 사용자 수준에 맞는 쉬운 용어만 사용하세요.
- 반드시 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.

예시 (초급 수준):
{{
  "question_text": "PER이 낮을수록 주식이 저평가되어 있다고 볼 수 있는 이유는?",
  "choices": [
    {{"no": 1, "text": "주가가 이익에 비해 낮기 때문"}},
    {{"no": 2, "text": "배당금이 많기 때문"}},
    {{"no": 3, "text": "시가총액이 크기 때문"}},
    {{"no": 4, "text": "매출이 높기 때문"}}
  ],
  "correct_no": 1,
  "explanation": "PER은 주가를 주당순이익으로 나눈 값으로, PER이 낮다는 것은 이익에 비해 주가가 낮다는 의미입니다.",
  "topic": "{topic}"
}}

JSON 형식:
{{
  "question_text": "문제 내용 (한 문장)",
  "choices": [
    {{"no": 1, "text": "선택지1"}},
    {{"no": 2, "text": "선택지2"}},
    {{"no": 3, "text": "선택지3"}},
    {{"no": 4, "text": "선택지4"}}
  ],
  "correct_no": 정답번호(1~4 중 하나),
  "explanation": "해설 내용 (2~3문장)",
  "topic": "{topic}"
}}
""")

OX_PROMPT = ChatPromptTemplate.from_template("""
당신은 주식 투자 교육 전문가입니다.
아래 조건에 맞는 OX 문제를 1개 생성하세요.

조건:
- 주제: {topic}
- 난이도: {level} ({level_guide})
- 반드시 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.

JSON 형식:
{{
  "question_text": "문제 내용",
  "answer": "O" 또는 "X",
  "explanation": "해설 내용 (2~3문장)",
  "topic": "{topic}"
}}
""")


def _get_llm() -> ChatOllama:
    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=0.7,  # 문제 다양성을 위해 높게
        reasoning=False,
    )


def _extract_json(text: str) -> dict:
    """LLM 응답에서 JSON 추출"""
    # 마크다운 코드블록 제거
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)


def _get_random_topic(quiz_type: str) -> str:
    import random
    if quiz_type == "OX":
        return random.choice(OX_TOPICS)
    return random.choice(MC_TOPICS)


async def generate_ox_question(user: UserContext, topic: str) -> dict:
    level_guide = LEVEL_GUIDE.get(user.investment_level, LEVEL_GUIDE["미설정"])
    chain = OX_PROMPT | _get_llm()

    for attempt in range(3):  # 최대 3번 재시도
        try:
            response = await chain.ainvoke({
                "topic": topic,
                "level": user.investment_level,
                "level_guide": level_guide,
            })
            data = _extract_json(response.content)

            # 필수 필드 검증
            assert "question_text" in data
            assert "answer" in data and data["answer"] in ["O", "X"]
            assert "explanation" in data

            return {
                "question_type": "OX",
                "question_text": data["question_text"],
                "choices": [
                    {"choice_no": 1, "text": "O", "is_correct": data["answer"] == "O"},
                    {"choice_no": 2, "text": "X", "is_correct": data["answer"] == "X"},
                ],
                "explanation": data["explanation"],
                "topic": data.get("topic", topic),
                "level": user.investment_level,
            }
        except Exception as e:
            if attempt == 2:
                raise ValueError(f"OX 문제 생성 실패: {e}")
            continue


async def generate_mc_question(user: UserContext, topic: str) -> dict:
    level_guide = LEVEL_GUIDE.get(user.investment_level, LEVEL_GUIDE["미설정"])
    chain = MC_PROMPT | _get_llm()

    for attempt in range(3):
        try:
            response = await chain.ainvoke({
                "topic": topic,
                "level": user.investment_level,
                "level_guide": level_guide,
            })
            data = _extract_json(response.content)

            # 필수 필드 검증
            assert "question_text" in data
            assert "choices" in data and len(data["choices"]) == 4
            assert "correct_no" in data and 1 <= data["correct_no"] <= 4
            assert "explanation" in data

            choices = [
                {
                    "choice_no": c["no"],
                    "text": c["text"],
                    "is_correct": c["no"] == data["correct_no"],
                }
                for c in data["choices"]
            ]

            return {
                "question_type": "MULTIPLE_CHOICE",
                "question_text": data["question_text"],
                "choices": choices,
                "explanation": data["explanation"],
                "topic": data.get("topic", topic),
                "level": user.investment_level,
            }
        except Exception as e:
            if attempt == 2:
                raise ValueError(f"객관식 문제 생성 실패: {e}")
            continue


async def generate_quiz(user: UserContext, quiz_type: str, topic: str = None) -> dict:
    if not topic:
        topic = _get_random_topic(quiz_type)

    if quiz_type == "OX":
        return await generate_ox_question(user, topic)
    elif quiz_type == "MULTIPLE_CHOICE":
        return await generate_mc_question(user, topic)
    else:
        raise ValueError(f"지원하지 않는 문제 유형: {quiz_type}")


async def generate_quiz_batch(user: UserContext, quiz_type: str, topic: str = None, count: int = 1) -> list[dict]:
    """같은 topic으로 count개의 문제를 순차 생성한다.

    로컬 Ollama가 단일 모델 인스턴스라 동시 요청을 병렬로 못 받는 경우가 많아
    순차 생성으로 처리한다 (팀 논의 결과 - 속도보단 안정성 우선).
    같은 topic으로 여러 번 생성하면 LLM이 같은 문제를 반복할 수 있어서,
    문제 텍스트가 이전에 나온 것과 겹치면 한 번 더 재생성을 시도한다
    (그래도 겹치면 마지막 결과를 그대로 채택 - count는 항상 맞춰야 하므로).
    """
    if not topic:
        topic = _get_random_topic(quiz_type)

    questions: list[dict] = []
    seen_texts: set[str] = set()

    for _ in range(count):
        question = await generate_quiz(user, quiz_type, topic)
        if question["question_text"] in seen_texts:
            question = await generate_quiz(user, quiz_type, topic)  # 중복이면 1회 재시도
        seen_texts.add(question["question_text"])
        questions.append(question)

    return questions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            