import re
from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate
from app.schemas.chat import UserContext
from app.schemas.quiz import QuizCategoryIn
from app.core.logger import setup_logger
from app.services.quiz import bank
from app.services.quiz.quality import (
    ANGLE_BLOCK,
    ANSWER_POSITION_BLOCK,
    MC_QUALITY_RULES,
    OX_QUALITY_RULES,
    QUALITY_RULES,
    angle_for,
    answer_pos_for,
    answer_pos_offset,
    check_mc_question,
    check_ox_question,
    extract_json,
    format_avoid_block,
    get_llm,
    is_near_duplicate,
    validate_questions,
)

# 하위 호환 별칭 (기존 코드/테스트가 _get_llm, _extract_json 이름을 참조한다)
_get_llm = get_llm
_extract_json = extract_json

logger = setup_logger(__name__)


class PromptTopicError(ValueError):
    """프롬프트에서 퀴즈 주제를 파악할 수 없을 때 (라우터에서 422로 매핑)."""

# 주의: 여기에 구체적인 지표명(PER, ROE 등)을 쓰면 안 된다.
# "초급"에 "PER, PBR, ROE 같은 기본 지표 수준"이라고 적혀 있던 탓에,
# 주제가 "주식"이어도 모델이 PER 문제를 만들었다(실측 이탈률 32%).
# 난이도는 '어느 수준까지 깊이 들어갈지'만 설명하고 소재는 언급하지 않는다.
LEVEL_GUIDE = {
    "입문": "주식 투자를 막 시작한 사람. 용어의 뜻을 아는 정도. 문제는 짧고 단순하게.",
    "초급": "기본 개념을 아는 사람. 개념의 정의와 단순한 계산을 이해하는 수준.",
    "중급": "투자 경험이 있는 사람. 여러 요소를 함께 해석하고 전략에 적용하는 수준.",
    "고급": "전문 투자자. 심화 분석과 복합적인 판단이 필요한 수준.",
    "미설정": "기본 개념을 아는 사람. 문제는 간단하게.",
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
- 주제: {topic}{topic_hint}
  이 주제의 범위를 벗어나지 마세요. 다른 지표나 개념을 주인공으로 삼으면 안 됩니다.
- 난이도: {level} ({level_guide})
""" + QUALITY_RULES + """
""" + MC_QUALITY_RULES + """
- 해설은 정답이 왜 맞고 오답이 왜 틀린지 설명하세요.
""" + ANGLE_BLOCK + """
참고용 예시 (형식만 보세요):
{{
  "question_text": "OOO에 대한 설명으로 옳은 것은?",
  "choices": [
    {{"no": 1, "text": "선택지 내용 (4개를 모두 채웁니다)"}},
    {{"no": 2, "text": "선택지 내용"}},
    {{"no": 3, "text": "선택지 내용"}},
    {{"no": 4, "text": "선택지 내용"}}
  ],
  "correct_no": 아래에 지시된 정답 번호,
  "explanation": "정답이 왜 맞는지, 오답이 왜 틀렸는지 설명합니다.",
  "topic": "{topic}"
}}

⚠️ 위 예시는 **JSON 구조만** 보여주는 것입니다.
반드시 "{topic}" 주제로 출제하세요. 다른 지표나 개념으로 새지 마세요.
""" + ANSWER_POSITION_BLOCK + """
JSON 형식:
{{
  "question_text": "문제 내용 (한 문장)",
  "choices": [
    {{"no": 1, "text": "선택지1"}},
    {{"no": 2, "text": "선택지2"}},
    {{"no": 3, "text": "선택지3"}},
    {{"no": 4, "text": "선택지4"}}
  ],
  "correct_no": {answer_pos},
  "explanation": "해설 내용 (2~3문장)",
  "topic": "{topic}"
}}
""")

OX_PROMPT = ChatPromptTemplate.from_template("""
당신은 주식 투자 교육 전문가입니다.
아래 조건에 맞는 OX 문제를 1개 생성하세요.

조건:
- 주제: {topic}{topic_hint}
  이 주제의 범위를 벗어나지 마세요. 다른 지표나 개념을 주인공으로 삼으면 안 됩니다.
- 난이도: {level} ({level_guide})
""" + QUALITY_RULES + """
""" + OX_QUALITY_RULES + """
- 해설은 왜 그 답이 맞는지 설명하세요.
""" + ANGLE_BLOCK + """
JSON 형식:
{{
  "question_text": "문제 내용",
  "answer": "O" 또는 "X",
  "explanation": "해설 내용 (2~3문장)",
  "topic": "{topic}"
}}
""")


def _get_random_topic(quiz_type: str) -> str:
    import random
    if quiz_type == "OX":
        return random.choice(OX_TOPICS)
    return random.choice(MC_TOPICS)


async def generate_ox_question(
    user: UserContext, topic: str, angle: str = "", avoid: list[str] | None = None,
    topic_desc: str = "",
) -> dict:
    level_guide = LEVEL_GUIDE.get(user.investment_level, LEVEL_GUIDE["미설정"])
    chain = OX_PROMPT | _get_llm()
    prompt_vars = {
        "topic": topic,
        "level": user.investment_level,
        "level_guide": level_guide,
        "angle": angle or angle_for(0),
        "avoid_block": format_avoid_block(avoid or []),
        # 주제명만 주면 범위가 모호해 다른 개념으로 샌다(실측). 카탈로그 설명을 함께 준다.
        "topic_hint": f" ({topic_desc})" if topic_desc else "",
    }

    for attempt in range(3):  # 최대 3번 재시도
        try:
            response = await chain.ainvoke(prompt_vars)
            data = _extract_json(response.content)

            check_ox_question(data, topic=topic, topic_desc=topic_desc)

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


async def generate_mc_question(
    user: UserContext, topic: str, angle: str = "", avoid: list[str] | None = None,
    topic_desc: str = "", answer_pos: int | None = None,
) -> dict:
    level_guide = LEVEL_GUIDE.get(user.investment_level, LEVEL_GUIDE["미설정"])
    chain = MC_PROMPT | _get_llm()
    # 지정하지 않으면 정답이 1번에 몰린다(실측 78.7%). 반드시 자리를 정해준다.
    answer_pos = answer_pos or answer_pos_for(0)
    prompt_vars = {
        "topic": topic,
        "level": user.investment_level,
        "level_guide": level_guide,
        "angle": angle or angle_for(0),
        "avoid_block": format_avoid_block(avoid or []),
        # 주제명만 주면 범위가 모호해 다른 개념으로 샌다(실측). 카탈로그 설명을 함께 준다.
        "topic_hint": f" ({topic_desc})" if topic_desc else "",
        "answer_pos": answer_pos,
    }

    for attempt in range(3):
        try:
            response = await chain.ainvoke(prompt_vars)
            data = _extract_json(response.content)

            check_mc_question(data, topic=topic, topic_desc=topic_desc,
                              answer_pos=answer_pos)

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


async def generate_quiz(
    user: UserContext, quiz_type: str, topic: str = None,
    angle: str = "", avoid: list[str] | None = None, topic_desc: str = "",
    answer_pos: int | None = None,
) -> dict:
    """문제 1개 생성.

    angle/avoid는 같은 주제로 여러 문제를 만들 때 중복을 막기 위한 것이다.
    (angle = 이번 문제의 출제 방향, avoid = 이미 만든 문제 목록)
    생략하면 첫 번째 관점을 쓴다 - 단건 생성에서는 중복 걱정이 없다.
    """
    if not topic:
        topic = _get_random_topic(quiz_type)

    if quiz_type == "OX":
        return await generate_ox_question(user, topic, angle, avoid, topic_desc)
    elif quiz_type == "MULTIPLE_CHOICE":
        return await generate_mc_question(
            user, topic, angle, avoid, topic_desc, answer_pos)
    else:
        raise ValueError(f"지원하지 않는 문제 유형: {quiz_type}")


async def generate_quiz_batch(
    user: UserContext, quiz_type: str, topic: str = None, count: int = 1,
    topic_desc: str = "",
) -> list[dict]:
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
    seen_texts: list[str] = []
    gen_args: list[tuple] = []

    # 과거에 이 주제로 만든 문제 수만큼 관점을 밀어, 재요청 시 다른 관점부터 시작한다
    offset = await bank.angle_offset(topic, user.user_id)

    for i in range(count):
        # 같은 topic이 반복되므로 문제마다 출제 관점을 돌려 중복을 막는다.
        # 이미 만든 문제도 프롬프트에 넘겨 같은 내용을 피하게 한다.
        angle = angle_for(offset + i)
        # 관점은 5주기, 정답 위치는 4주기라 서로소다. 20문제가 지나야 조합이 반복된다.
        # 주제마다 시작점을 밀어, 5문제 중 두 번 걸리는 자리가 주제별로 달라지게 한다.
        answer_pos = answer_pos_for(offset + i + answer_pos_offset(topic))
        question = await generate_quiz(
            user, quiz_type, topic, angle, seen_texts, topic_desc, answer_pos)
        # 세트 안 중복(문자 유사도) + 과거 요청과의 중복(임베딩 유사도)
        if await _is_duplicate(question["question_text"], seen_texts, user):
            question = await generate_quiz(
                user, quiz_type, topic, angle, seen_texts, topic_desc, answer_pos)
        seen_texts.append(question["question_text"])
        questions.append(question)
        gen_args.append((quiz_type, topic, angle, list(seen_texts), topic_desc, answer_pos))

    await _revalidate_and_fix(questions, gen_args, user)
    # 검증·재생성이 끝난 확정본만 저장한다
    await bank.register(questions, user.user_id)
    return questions


async def _is_duplicate(text: str, seen_texts: list[str], user: UserContext) -> bool:
    """세트 안 중복(무료) 먼저 보고, 통과하면 과거 문제와 대조한다.

    문자 유사도가 먼저인 이유는 비용이 0이기 때문이다.
    여기서 걸리면 Qdrant 호출 자체를 아낀다.
    """
    if is_near_duplicate(text, seen_texts):
        logger.info(f"세트 내 중복 감지 - 재생성: {text[:40]}")
        return True
    return await bank.is_duplicate_of_past(text, user.user_id)


async def _revalidate_and_fix(
    questions: list[dict], gen_args: list[tuple], user: UserContext,
) -> None:
    """정답/해설이 모순된 문제를 찾아 같은 조건으로 1회 재생성한다 (in-place 수정).

    세트 전체를 LLM 1회 호출로 검사한다. 재생성본은 다시 검증하지 않는다
    (검증 호출이 계속 늘어나는 것을 막기 위한 타협).
    재생성에 실패하면 원본을 그대로 둔다 - 문제 개수는 항상 맞춰야 한다.
    """
    failed = await validate_questions(questions)

    if not failed:
        logger.info("문제 검증 통과 (전체 합격)")
        return

    logger.warning(f"검증 불합격 {len(failed)}건 - 재생성: {list(failed.values())}")
    for idx, _reason in failed.items():
        quiz_type, topic, angle, avoid, topic_desc, answer_pos = gen_args[idx]
        try:
            fixed = await generate_quiz(
                user, quiz_type, topic, angle, avoid, topic_desc, answer_pos)
        except ValueError as e:
            logger.warning(f"{idx + 1}번 재생성 실패 - 원본 유지: {e}")
            continue
        # 카테고리 매핑 등 생성 이후 붙은 필드는 원본 것을 승계한다
        for field in ("category_code", "detail_codes", "primary_detail_code"):
            if field in questions[idx]:
                fixed[field] = questions[idx][field]
        questions[idx] = fixed


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
- **같은 개념이 여러 등급에 걸쳐 있으면 반드시 "{preferred_level}" 등급을 먼저 고르세요.**
  (예: 사용자가 초급인데 "배당수익률" 주제라면, 중급의 "배당 공시"가 아니라
   초급의 "배당"을 선택해야 합니다. 등급을 올려 잡으면 안 됩니다)
  해당 등급에 정말 맞는 것이 없을 때만 다른 등급을 고르세요.
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


# 프롬프트에서 문제 유형 의도를 읽어내는 패턴.
# 분석 LLM이 "OX 3개 내줘"에도 객관식을 섞어 내는 경우가 있어(실측)
# 계획 단계 결과를 코드로 한 번 더 강제한다.
# \bOX\b 를 쓰면 "OX로", "OX만" 처럼 조사가 붙었을 때 매칭되지 않는다
# (한글도 \w 라서 단어 경계가 생기지 않음). 앞뒤 영문자만 배제한다.
_OX_MENTION = re.compile(r"(?<![A-Za-z])OX(?![A-Za-z])|O/X|오엑스|오·엑스", re.IGNORECASE)
_MIX_MENTION = re.compile(r"섞|혼합|골고루|반반")


def _enforce_question_type(prompt: str, items: list[dict]) -> list[dict]:
    """출제 계획의 문제 유형을 사용자 의도에 맞춰 통일한다.

    - "섞어서" 류 요청이면 그대로 둔다 (혼합이 의도된 경우).
    - 프롬프트에 OX가 명시되면 전부 OX로 강제한다.
    - 그 외에 유형이 섞여 나왔으면 전부 객관식으로 통일한다
      (유형 언급이 없으면 객관식이 기본이라는 기존 정책과 일치).
    """
    if not items or _MIX_MENTION.search(prompt):
        return items

    if _OX_MENTION.search(prompt):
        forced = "OX"
    elif len({item["question_type"] for item in items}) > 1:
        forced = "MULTIPLE_CHOICE"
    else:
        return items

    changed = sum(1 for item in items if item["question_type"] != forced)
    if changed:
        logger.info(f"문제 유형 통일: {changed}개 항목을 {forced}로 보정")
    for item in items:
        item["question_type"] = forced
    return items


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


async def _map_topics_to_catalog(
    topics: list[str], categories: list[QuizCategoryIn], preferred_level: str = "초급",
) -> dict[str, dict]:
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
                "preferred_level": preferred_level,
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
    # 분석 LLM이 유형을 섞어 내는 경우가 있어 사용자 의도대로 통일한다
    items = _enforce_question_type(prompt, items)

    unmapped = [item for item in items if item["category_code"] is None]
    if unmapped and len(level_catalog) != len(QUIZ_CATEGORY_CATALOG):
        topics = list(dict.fromkeys(item["topic"] for item in unmapped))
        logger.info(f"등급({investment_level}) 밖 주제 재매핑 시도: {topics}")
        # 입문 유저는 초급 카테고리를 쓰므로 그에 맞춰 선호 등급을 정한다
        preferred = "초급" if investment_level in ("입문", "미설정") else investment_level
        mappings = await _map_topics_to_catalog(
            topics, QUIZ_CATEGORY_CATALOG, preferred_level=preferred,
        )
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
    seen_texts: list[str] = []
    gen_args: list[tuple] = []
    # 같은 주제가 여러 항목에 걸쳐 나오므로 주제별로 출제 관점을 돌린다
    # (관점 없이 돌리면 "배당수익률 = 배당금/주가" 하나를 표현만 바꿔 반복함 - 실측)
    topic_slot: dict[str, int] = defaultdict(int)
    # 주제별로 과거 생성 이력만큼 관점을 밀어둔다 (재요청 시 중복 방지)
    for t in {item["topic"] for item in plan}:
        topic_slot[t] = await bank.angle_offset(t, user.user_id)

    for set_index, item in enumerate(plan):
        topic = item["topic"]
        angle = angle_for(topic_slot[topic])
        topic_slot[topic] += 1
        # 정답 위치는 주제별이 아니라 세트 전체 순번으로 돌린다.
        # 주제별로 돌리면 주제가 전부 다를 때 모두 0번째라 정답이 1번에 몰린다.
        answer_pos = answer_pos_for(set_index)

        question = await generate_quiz(
            user, item["question_type"], topic, angle, seen_texts,
            answer_pos=answer_pos)
        # 세트 안 중복(문자 유사도) + 과거 요청과의 중복(임베딩 유사도)
        if await _is_duplicate(question["question_text"], seen_texts, user):
            question = await generate_quiz(
                user, item["question_type"], topic, angle, seen_texts,
                answer_pos=answer_pos)
        seen_texts.append(question["question_text"])

        question["category_code"] = item["category_code"]
        question["detail_codes"] = item["detail_codes"]
        question["primary_detail_code"] = item["detail_codes"][0] if item["detail_codes"] else None
        questions.append(question)
        gen_args.append(
            (item["question_type"], topic, angle, list(seen_texts), "", answer_pos))

    await _revalidate_and_fix(questions, gen_args, user)
    await bank.register(questions, user.user_id)
    return questions