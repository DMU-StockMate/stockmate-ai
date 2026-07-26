import json
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
from app.schemas.chat import UserContext
from app.schemas.quiz import QuizCategoryIn
from app.core.logger import setup_logger

logger = setup_logger(__name__)


class PromptTopicError(ValueError):
    """프롬프트에서 퀴즈 주제를 파악할 수 없을 때 (라우터에서 422로 매핑)."""

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
- 문제·선택지·해설의 모든 텍스트는 한국어로 작성하세요.
- 실제로 존재하는 개념·지표·용어만 사용하고, 존재하지 않는 용어를 지어내지 마세요.
- 정답은 반드시 하나만 명확한 정답이어야 하고, 오답 선택지는 그럴듯하지만 분명히 틀린 내용이어야 합니다.
- 정답 번호(correct_no)와 해설이 서로 모순되지 않아야 하며, 해설은 정답이 왜 맞고 오답이 왜 틀린지 설명하세요.
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
- 문제는 반드시 한 문장으로 짧고 명확하게 작성하세요.
- 사용자 수준에 맞는 쉬운 용어만 사용하세요.
- 문제·해설의 모든 텍스트는 한국어로 작성하세요.
- 실제로 존재하는 개념·용어만 사용하고, 존재하지 않는 용어를 지어내지 마세요.
- 정답(answer)과 해설이 서로 모순되지 않아야 하며, 해설은 왜 그 답이 맞는지 설명하세요.
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
        # 문제+선택지+해설 JSON이 중간에 잘리면 파싱 실패로 재시도를 소진하므로 출력 토큰 확보.
        # num_ctx는 채팅 체인과 동일하게 맞춰 Ollama가 모델을 재로드(지연)하지 않게 한다.
        num_predict=1024,
        num_ctx=6144,
    )


def _extract_json(text: str) -> dict:
    """LLM 응답에서 JSON 추출.

    코드펜스만 제거하던 기존 방식은 LLM이 JSON 앞뒤에 설명 문장을 붙이면 곧바로 실패해
    재시도 3회를 소진하고 500이 나기 쉬웠다. 그래서 먼저 통짜 파싱을 시도하고, 실패하면
    첫 번째 균형 잡힌 중괄호 블록 `{...}`을 찾아 파싱한다.
    """
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("응답에서 JSON 객체를 찾지 못함")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("응답의 JSON 중괄호가 닫히지 않음")


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
            # 선택지 번호가 1~4로 중복 없이 존재하고 정답 번호가 그 안에 있는지 검증.
            # (LLM이 no를 중복/누락하면 is_correct 매핑이 깨지므로 여기서 걸러 재시도한다)
            nos = sorted(c["no"] for c in data["choices"])
            assert nos == [1, 2, 3, 4], f"선택지 번호 이상: {nos}"

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

    return questions


# =========================================================
# 프롬프트 기반 문제 생성
# =========================================================

# 순차 생성이라 개수에 비례해 응답이 느려지므로 상한을 둔다 (팀 결정: 10개).
MAX_PROMPT_QUIZ_COUNT = 10
DEFAULT_PROMPT_QUIZ_COUNT = 5

# 1단계(분석) 프롬프트: 사용자 프롬프트에서 주제/개수/유형을 파악하고
# NestJS가 전달한 카테고리/상세 코드 목록에 매핑한 '출제 계획'을 JSON으로 만든다.
# 2단계에서 계획의 각 항목을 기존 OX/MC 생성 프롬프트로 실제 문제로 만든다.
ANALYZE_PROMPT = ChatPromptTemplate.from_template("""
당신은 주식 투자 교육 전문가입니다.
사용자의 요청 프롬프트를 분석해서 퀴즈 출제 계획을 JSON으로 작성하세요.

[사용자 요청 프롬프트]
{prompt}

[사용 가능한 카테고리 목록]
{catalog}

규칙:
- 요청이 주식/투자/금융 학습과 관련이 없거나 출제 주제를 파악할 수 없으면
  {{"relevant": false}} 만 출력하세요.
- count: 프롬프트에 문제 개수가 명시되어 있으면 그 숫자, 없으면 {default_count}.
- items: 정확히 count개의 출제 항목을 만드세요.
  - 프롬프트에 여러 주제가 있으면 항목들을 주제별로 고르게 나누세요.
  - 주제가 하나면 모든 항목이 같은 주제여도 됩니다 (문제 내용은 서로 다르게 생성됩니다).
- topic: 문제를 출제할 구체적인 주제 (한국어, 예: "PER", "분산투자").
  반드시 프롬프트에 언급된 주제를 그대로 사용하세요.
  프롬프트에 없는 주제로 임의로 확장하거나 다른 개념과 조합하거나 변형하지 마세요.
  (예: 프롬프트가 "PER"이면 topic은 "PER" - "PER + ROE 조합"으로 바꾸면 안 됨)
- question_type: 프롬프트에 "OX"라는 표현이 명시적으로 있을 때만 "OX".
  그 외에는 반드시 전부 "MULTIPLE_CHOICE"로 하세요.
  프롬프트에 유형 언급이 없는데 임의로 OX를 섞는 것은 금지입니다.
  단, "섞어서"라고 요청한 경우에만 항목별로 OX와 MULTIPLE_CHOICE를 나누세요.
- category_code: 위 카테고리 목록에서 topic과 가장 잘 맞는 카테고리의 code.
  목록에 있는 code만 사용하세요. 맞는 것이 없으면 null.
- detail_codes: 해당 카테고리의 상세 목록 중 topic과 관련된 code들 (보통 1개).
  목록에 있는 code만 사용하세요. 없으면 빈 배열.
- 반드시 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.

JSON 형식:
{{
  "relevant": true,
  "count": 문제개수,
  "items": [
    {{
      "topic": "주제",
      "question_type": "OX 또는 MULTIPLE_CHOICE",
      "category_code": "카테고리 코드 또는 null",
      "detail_codes": ["상세 코드"]
    }}
  ]
}}
""")


# 2차 매핑 프롬프트: 1차 분석(수준별 카탈로그)에서 매핑에 실패한 주제만
# 전체 카탈로그를 상대로 다시 매핑한다. 문제 출제 계획은 이미 확정된 상태라
# 카테고리 매핑만 하면 되는 가벼운 호출이다.
MAP_TOPICS_PROMPT = ChatPromptTemplate.from_template("""
당신은 주식 투자 교육 전문가입니다.
아래 주제들을 카테고리 목록에 매핑하세요.

[주제 목록]
{topics}

[사용 가능한 카테고리 목록]
{catalog}

규칙:
- 각 주제마다 가장 잘 맞는 카테고리의 category_code 1개와
  관련된 detail_codes(보통 1개)를 고르세요.
- 목록에 있는 code만 사용하세요. 맞는 것이 없으면 category_code를 null로 하세요.
- topic은 주제 목록의 문자열을 그대로 사용하세요.
- 반드시 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.

JSON 형식:
{{
  "mappings": [
    {{"topic": "주제", "category_code": "카테고리 코드 또는 null", "detail_codes": ["상세 코드"]}}
  ]
}}
""")


def _format_catalog(categories: list[QuizCategoryIn]) -> str:
    """카테고리/상세 목록을 분석 프롬프트에 넣을 텍스트로 변환한다.

    코드와 이름만 넣으면 LLM이 매핑을 잘 못해서 description(예: "PER, ROE, EPS, PBR")도
    함께 넣는다 — 상세 코드가 없는 주제도 카테고리 설명으로 매핑할 수 있게.
    같은 주제(예: PER)가 난이도별로 여러 카테고리에 있어 등급 표시로 구분을 돕는다.
    """
    lines = []
    for cat in categories:
        level = f"({cat.investment_level}) " if cat.investment_level else ""
        desc = f" - {cat.description}" if cat.description else ""
        lines.append(f"{level}[{cat.category_code}] {cat.category_name}{desc}")
        for d in cat.details:
            d_desc = f": {d.description}" if d.description else ""
            lines.append(f"  - {d.detail_code} ({d.detail_name}){d_desc}")
    return "\n".join(lines)


def _validate_plan_item(item: dict, categories: list[QuizCategoryIn]) -> dict | None:
    """분석 결과의 항목 1개를 검증/정제한다. topic이 없으면 None(버림).

    - question_type이 이상하면 MULTIPLE_CHOICE로 보정.
    - category_code/detail_codes는 요청으로 받은 목록에 실제로 존재하는 코드만 통과시킨다
      (LLM이 코드를 지어내는 경우 방지). detail이 다른 카테고리 소속이면 버린다.
    """
    topic = (item.get("topic") or "").strip()
    if not topic:
        return None

    question_type = item.get("question_type")
    if question_type not in ("OX", "MULTIPLE_CHOICE"):
        question_type = "MULTIPLE_CHOICE"

    cat_map = {c.category_code: c for c in categories}
    category_code = item.get("category_code")
    if category_code not in cat_map:
        category_code = None

    detail_codes: list[str] = []
    raw_details = item.get("detail_codes") or []
    if category_code:
        valid_details = {d.detail_code for d in cat_map[category_code].details}
        detail_codes = [d for d in raw_details if d in valid_details]
    elif raw_details:
        # 카테고리 없이 상세 코드만 온 경우: 상세 코드로 소속 카테고리를 역추적한다.
        for cat in categories:
            matched = [d for d in raw_details if d in {x.detail_code for x in cat.details}]
            if matched:
                category_code = cat.category_code
                detail_codes = matched
                break

    return {
        "topic": topic,
        "question_type": question_type,
        "category_code": category_code,
        "detail_codes": detail_codes,
    }


async def _analyze_once(prompt: str, categories: list[QuizCategoryIn]) -> list[dict]:
    """주어진 카탈로그로 분석 1회 실행 → 검증된 출제 계획 반환."""
    chain = ANALYZE_PROMPT | _get_llm()
    catalog = _format_catalog(categories)

    data = None
    for attempt in range(3):
        try:
            response = await chain.ainvoke({
                "prompt": prompt,
                "catalog": catalog,
                "default_count": DEFAULT_PROMPT_QUIZ_COUNT,
            })
            data = _extract_json(response.content)
            break
        except Exception as e:
            if attempt == 2:
                raise PromptTopicError(f"프롬프트 분석 실패: {e}")
            continue

    if not data.get("relevant") or not data.get("items"):
        raise PromptTopicError("프롬프트에서 퀴즈 주제를 파악할 수 없습니다.")

    items = [
        valid for item in data["items"]
        if isinstance(item, dict) and (valid := _validate_plan_item(item, categories))
    ]
    if not items:
        raise PromptTopicError("프롬프트에서 퀴즈 주제를 파악할 수 없습니다.")

    # count와 items 길이가 어긋나면 items를 기준으로 맞춘다:
    # 부족하면 기존 항목을 순환 복제해 채우고, 초과분은 자른다.
    try:
        count = int(data.get("count") or len(items))
    except (TypeError, ValueError):
        count = len(items)
    count = max(1, min(count, MAX_PROMPT_QUIZ_COUNT))

    if len(items) < count:
        base = list(items)
        while len(items) < count:
            items.append(dict(base[len(items) % len(base)]))
    return items[:count]


async def _map_topics_to_catalog(topics: list[str], categories: list[QuizCategoryIn]) -> dict[str, dict]:
    """주제 목록을 카탈로그에 매핑한다. {topic: {"category_code", "detail_codes"}} 반환.

    실패해도 예외를 던지지 않고 빈 dict를 반환한다 — 매핑은 부가 정보라
    실패했다고 문제 생성 전체를 막을 이유가 없다 (category_code=null 허용).
    """
    chain = MAP_TOPICS_PROMPT | _get_llm()
    data = None
    for attempt in range(3):
        try:
            response = await chain.ainvoke({
                "topics": "\n".join(f"- {t}" for t in topics),
                "catalog": _format_catalog(categories),
            })
            data = _extract_json(response.content)
            break
        except Exception as e:
            if attempt == 2:
                logger.warning(f"주제 재매핑 실패 - 매핑 없이 진행: {e}")
                return {}
            continue

    result: dict[str, dict] = {}
    mappings = data.get("mappings") or []
    for m in mappings:
        if not isinstance(m, dict):
            continue
        validated = _validate_plan_item({
            "topic": m.get("topic"),
            "question_type": "MULTIPLE_CHOICE",  # 매핑만 쓰므로 유형은 의미 없음
            "category_code": m.get("category_code"),
            "detail_codes": m.get("detail_codes"),
        }, categories)
        if validated and validated["category_code"]:
            result[validated["topic"]] = {
                "category_code": validated["category_code"],
                "detail_codes": validated["detail_codes"],
            }

    # LLM이 topic 문자열을 바꿔 보내면 exact 매칭이 깨진다.
    # 요청 주제가 1개뿐이고 매핑도 1개면 문자열이 달라도 그 매핑을 적용한다.
    if not any(t in result for t in topics) and len(topics) == 1 and len(result) == 1:
        result = {topics[0]: next(iter(result.values()))}
    return result


async def analyze_quiz_prompt(prompt: str, investment_level: str = "미설정") -> list[dict]:
    """프롬프트를 분석해 검증된 출제 계획(항목 리스트)을 반환한다.

    1차: 사용자 수준에 해당하는 등급의 카테고리만 LLM에 보여주고 분석한다
    (초급 유저 문제가 불필요하게 중급/고급 카테고리로 분류되는 것을 차단).

    2차: 유저가 자기 등급에 없는 주제를 요청하면(예: 초급 유저가 "볼린저밴드")
    해당 항목만 매핑이 비므로, 매핑 실패 항목의 주제만 모아 전체 카탈로그로
    재매핑한다 — 초급 유저도 고급 카테고리 주제의 문제를 풀 수 있게 한다.
    ("PER이랑 볼린저밴드"처럼 섞인 요청도 볼린저밴드만 재매핑되어 처리됨)

    실패 시 PromptTopicError를 던진다 (라우터가 422로 변환).
    """
    from app.services.quiz.categories import QUIZ_CATEGORY_CATALOG, get_catalog_for_level

    level_catalog = get_catalog_for_level(investment_level)
    items = await _analyze_once(prompt, level_catalog)

    unmapped = [item for item in items if item["category_code"] is None]
    if unmapped and len(level_catalog) != len(QUIZ_CATEGORY_CATALOG):
        topics = list(dict.fromkeys(item["topic"] for item in unmapped))
        logger.info(f"등급({investment_level}) 밖 주제 재매핑 시도: {topics}")
        mappings = await _map_topics_to_catalog(topics, QUIZ_CATEGORY_CATALOG)
        for item in unmapped:
            mapping = mappings.get(item["topic"])
            if mapping:
                item["category_code"] = mapping["category_code"]
                item["detail_codes"] = mapping["detail_codes"]

    return items


async def generate_quiz_from_prompt(prompt: str, user: UserContext) -> list[dict]:
    """프롬프트 기반 문제 생성 파이프라인 (분석 → 순차 생성).

    generate_quiz_batch와 같은 이유(로컬 Ollama 단일 인스턴스)로 순차 생성하고,
    같은 topic 항목이 연속되면 중복 문제가 나올 수 있어 동일한 중복 1회 재시도를 적용한다.
    개별 문제 생성 실패(재시도 3회 소진)는 ValueError로 전파된다 (라우터에서 500).
    """
    plan = await analyze_quiz_prompt(prompt, user.investment_level)
    logger.info(
        f"프롬프트 퀴즈 계획: {len(plan)}개 "
        f"[{', '.join(item['topic'] for item in plan)}]"
    )

    questions: list[dict] = []
    seen_texts: set[str] = set()

    for item in plan:
        question = await generate_quiz(user, item["question_type"], item["topic"])
        if question["question_text"] in seen_texts:
            question = await generate_quiz(user, item["question_type"], item["topic"])
        seen_texts.add(question["question_text"])

        question["category_code"] = item["category_code"]
        question["detail_codes"] = item["detail_codes"]
        question["primary_detail_code"] = item["detail_codes"][0] if item["detail_codes"] else None
        questions.append(question)

    return questions