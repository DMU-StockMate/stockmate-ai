import asyncio
import re
from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate
from app.schemas.chat import UserContext
from app.schemas.quiz import QuizCategoryIn
from app.core.config import settings
from app.core.logger import setup_logger
from app.services.quiz import bank
from app.services.quiz.categories import (
    find_topic_context,
    get_detail_description,
    get_problem_direction,
)
from app.services.quiz.concurrency import (
    fire_and_forget,
    generate_all,
    regenerate_failed,
    resolve_duplicates,
)
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
    validate_questions,
)

# 하위 호환 별칭 (기존 코드/테스트가 _get_llm, _extract_json 이름을 참조한다)
_get_llm = get_llm
_extract_json = extract_json

logger = setup_logger(__name__)


def _analyze_llm():
    """프롬프트 분석·카테고리 매핑용 LLM.

    문제 생성과 사고 강도를 분리한다. 생성은 용어를 정확히 인출해야 해서
    medium 이 필요하지만(모델 선정 문서 참고), 분석은 "프롬프트에서 주제·개수·
    유형을 뽑아 카탈로그 코드에 맞추는" 구조화 추출이라 사고가 덜 필요하다.
    계측상 분석 한 번이 19.8초로 전체의 24% 였다.
    """
    return get_llm(reasoning_effort=settings.ANALYZE_REASONING_EFFORT or None)


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


# 등급별 문제 '구조' 지시 (학습 기준표의 목적 / 데이터 난이도 / 판단 방식 / 문제 형태).
#
# LEVEL_GUIDE 가 "어느 수준까지 깊이 들어갈지"를 말한다면, 여기는 "문제를 어떤 모양으로
# 만들지"를 말한다. 초급은 정보 1개로 정답이 명확하게, 중급은 정보 2~3개를 비교해서,
# 고급은 여러 자료를 놓고 근거의 타당성으로 갈리게 만든다.
#
# ⚠️ LEVEL_GUIDE 와 같은 이유로 여기에도 구체적인 지표명(PER, ROE 등)을 쓰면 안 된다.
# 지표명을 넣으면 주제가 "주식"이어도 모델이 그 지표로 새버린다(실측 이탈률 32%).
# 그래서 예시 형태는 개념 자리를 "(개념)", "(상황)" 으로 비워두고 문장 모양만 보여준다.
LEVEL_STYLE = {
    "입문": """- 목적: 투자 기본 개념을 익히게 하는 것입니다.
- 제시 정보: 정보를 하나만 제시하세요. 두 가지 이상을 조합하게 만들지 마세요.
- 판단 방식: 정답이 분명하게 갈리도록 만드세요.
- 문제 형태 예시 (모양만 참고하고 그대로 베끼지 마세요):
  "(개념)에 대한 설명으로 옳은 것은?" 처럼 용어의 뜻을 바로 묻는 형태.""",

    "초급": """- 목적: 용어와 기본 판단 기준을 익히게 하는 것입니다.
- 제시 정보: 정보를 하나만 제시하세요. 여러 정보를 조합해야 풀리는 문제는 만들지 마세요.
- 판단 방식: 정답이 비교적 명확하게 갈리도록 만드세요.
- 문제 형태 예시 (모양만 참고하고 그대로 베끼지 마세요):
  "(개념)이 낮다는 것은 일반적으로 어떤 의미인가?"
  "(개념)이 높을수록 어떤 점에서 긍정적인가?"
  "(상품)은 (다른 상품)보다 일반적으로 위험이 높은가 낮은가?" """,

    "중급": """- 목적: 여러 정보를 조합해서 투자 판단을 하게 하는 것입니다.
- 제시 정보: 서로 다른 정보를 2~3개 함께 제시하세요.
  (예: 가격 흐름과 재무 상태를 함께, 또는 재무 상태와 뉴스를 함께)
- 판단 방식: 단순 암기가 아니라 근거들을 비교해서 고르게 만드세요.
- 단, **제시하는 정보가 여러 개여도 질문은 하나여야 합니다.**
  상황 설명에 정보를 여러 개 담되, 마지막에 묻는 것은 하나로 좁히세요.
- 문제 형태 예시 (모양만 참고하고 그대로 베끼지 마세요):
  "A기업은 (상황1)이고 (상황2)이지만 (상황3)이다. 가장 적절한 판단은?"
  "(사건)이 있었지만 이미 (상황)이라면 어떤 위험이 있는가?" """,

    "고급": """- 목적: 실제 투자 전략을 세우고 리스크까지 판단하게 하는 것입니다.
- 제시 정보: 여러 자료를 복합적으로 제시하세요.
  (예: 가격 흐름과 재무 상태와 공시·뉴스를 한 상황 안에 함께)
- 판단 방식: 정답 하나를 외워서 맞히는 문제가 아니라 근거의 타당성에서 갈리는
  문제로 만드세요. 오답도 언뜻 그럴듯해 보이게 쓰되, 실제로는 틀린 내용이어야 합니다.
- 단, **제시하는 자료가 여러 개여도 질문은 하나여야 합니다.**
  상황 설명에 자료를 여러 개 담되, 마지막에 묻는 것은 하나로 좁히세요.
- 문제 형태 예시 (모양만 참고하고 그대로 베끼지 마세요):
  "(시점) 당시 정보만으로 판단할 때 매수·보유·매도 중 가장 적절한 것은?"
  "(긍정 요인)이 있지만 (부정 요인)도 함께 있을 때 가장 적절한 전략은?" """,

    "미설정": """- 목적: 투자 기본 개념을 익히게 하는 것입니다.
- 제시 정보: 정보를 하나만 제시하세요.
- 판단 방식: 정답이 비교적 명확하게 갈리도록 만드세요.""",
}


def level_style_for(investment_level: str) -> str:
    """사용자 등급에 맞는 문제 구조 지시. 모르는 등급이면 미설정 기준."""
    return LEVEL_STYLE.get(investment_level, LEVEL_STYLE["미설정"])


def format_direction_block(direction: str) -> str:
    """카테고리의 출제 방향을 프롬프트 조각으로 만든다. 없으면 빈 문자열.

    카탈로그의 problem_direction("주식 용어 이해", "차트 흐름을 단순 해석" 등)은
    그동안 categories.py 에 데이터로만 있고 실제 생성 프롬프트에는 전달되지 않았다.
    같은 주제라도 카테고리가 의도한 방향이 다르므로(예: 중급 "보조지표 개념"은
    지표의 의미를, 고급 "보조지표 활용 판단"은 매매 판단을 묻는다) 여기서 넣어준다.
    """
    if not direction:
        return ""
    return (
        f"- 출제 방향: {direction}\n"
        f"  이 방향에 맞는 문제를 만드세요. 같은 주제라도 이 방향에서 벗어나면 안 됩니다.\n"
    )


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
{direction_block}- 난이도: {level} ({level_guide})

[이 난이도에서 문제를 만드는 방식]
{level_style}

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
{direction_block}- 난이도: {level} ({level_guide})

[이 난이도에서 문제를 만드는 방식]
{level_style}

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


# =========================================================
# 한 번의 호출로 여러 문제 생성 (중복 대책)
# =========================================================
#
# 왜 만들었나 - 2026-09-08 측정 (n=30씩, 같은 주제 3문제):
#
#   중복 발동률 77%.  30세트 중 11건은 유사도 0.95 이상(거의 글자까지 동일).
#
# 문제를 하나씩 병렬로 만들면 서로의 결과를 볼 수 없어서, 모델이 같은 주제에
# 대해 각자 가장 자연스러운 첫 문장("X에 대한 설명으로 옳은 것은?")을 쓴다.
# 관점(angle)을 다르게 줘도, "다른 문제가 어느 방향을 맡았는지" 알려줘도
# 바뀌지 않았다 (형제 관점 힌트 실험: 23/30 -> 22/30, 기각).
#
# 부탁으로 될 일이 아니라서 구조를 바꿨다. **한 번의 호출로 N개를 쓰게 하면**
# 모델이 N개를 나란히 놓고 쓰므로 스스로 다르게 만든다. 덤으로 LLM 호출이
# N번 -> 1번이 되고, 2,900자짜리 프롬프트를 N번 보내던 것도 1번이 된다.
#
# 안전장치: 출력 토큰이 개수에 비례해 늘어나므로 한 번에 만들 수를 제한한다
# (settings.QUIZ_BATCH_MAX). 그보다 많으면 여러 덩어리로 나눠 병렬 호출한다.

MC_BATCH_PROMPT = ChatPromptTemplate.from_template("""
당신은 주식 투자 교육 전문가입니다.
아래 조건에 맞는 4지선다 객관식 문제를 **정확히 {count}개** 생성하세요.

조건:
- 주제: {topic}{topic_hint}
  이 주제의 범위를 벗어나지 마세요. 다른 지표나 개념을 주인공으로 삼으면 안 됩니다.
{direction_block}- 난이도: {level} ({level_guide})

[이 난이도에서 문제를 만드는 방식]
{level_style}

""" + QUALITY_RULES + """
""" + MC_QUALITY_RULES + """
- 해설은 정답이 왜 맞고 오답이 왜 틀린지 설명하세요.

=========================================
가장 중요한 지시 — 문제별 출제 방향과 정답 위치
=========================================
{slot_block}

- 각 문제는 **배정된 방향 하나만** 다루세요.
- **{count}개 문제는 서로 확실히 달라야 합니다.** 표현만 바꿔서 같은 것을 묻는 것도
  중복입니다. 특히 여러 문제를 "{topic}에 대한 설명으로 옳은 것은?" 같은 같은
  형태로 시작하지 마세요. 문제를 쓰기 전에 {count}개가 서로 무엇이 다른지 정하세요.
- **correct_no 는 위에 지정된 번호를 그대로 쓰세요.**

JSON 형식 (questions 배열의 길이는 반드시 {count}):
{{
  "questions": [
    {{
      "question_text": "문제 내용 (한 문장)",
      "choices": [
        {{"no": 1, "text": "선택지1"}},
        {{"no": 2, "text": "선택지2"}},
        {{"no": 3, "text": "선택지3"}},
        {{"no": 4, "text": "선택지4"}}
      ],
      "correct_no": 지정된 정답 번호,
      "explanation": "해설 내용 (2~3문장)",
      "topic": "{topic}"
    }}
  ]
}}
""")

OX_BATCH_PROMPT = ChatPromptTemplate.from_template("""
당신은 주식 투자 교육 전문가입니다.
아래 조건에 맞는 OX 문제를 **정확히 {count}개** 생성하세요.

조건:
- 주제: {topic}{topic_hint}
  이 주제의 범위를 벗어나지 마세요. 다른 지표나 개념을 주인공으로 삼으면 안 됩니다.
{direction_block}- 난이도: {level} ({level_guide})

[이 난이도에서 문제를 만드는 방식]
{level_style}

""" + QUALITY_RULES + """
""" + OX_QUALITY_RULES + """
- 해설은 왜 그 답이 맞는지 설명하세요.

=========================================
가장 중요한 지시 — 문제별 출제 방향
=========================================
{slot_block}

- 각 문제는 **배정된 방향 하나만** 다루세요.
- **{count}개 문제는 서로 확실히 달라야 합니다.** 표현만 바꿔서 같은 것을 묻는 것도
  중복입니다. 문제를 쓰기 전에 {count}개가 서로 무엇이 다른지 정하세요.
- 정답이 전부 O 이거나 전부 X 가 되지 않게 섞으세요.

JSON 형식 (questions 배열의 길이는 반드시 {count}):
{{
  "questions": [
    {{
      "question_text": "문제 내용",
      "answer": "O" 또는 "X",
      "explanation": "해설 내용 (2~3문장)",
      "topic": "{topic}"
    }}
  ]
}}
""")


def _format_slot_block(slots: list[dict], quiz_type: str) -> str:
    """문제별 출제 방향(과 정답 위치)을 프롬프트 조각으로 만든다."""
    lines = []
    for i, slot in enumerate(slots, 1):
        lines.append(f"[{i}번 문제]")
        lines.append(slot["angle"])
        if quiz_type == "MULTIPLE_CHOICE":
            lines.append(f"→ 이 문제의 정답은 반드시 {slot['answer_pos']}번 자리에 두세요.")
        lines.append("")
    return "\n".join(lines).rstrip()


async def generate_quiz_chunk(
    user: UserContext, quiz_type: str, topic: str, slots: list[dict],
    topic_desc: str = "", direction: str = "",
) -> list[dict]:
    """한 번의 LLM 호출로 slots 개수만큼 문제를 만든다.

    slots: [{"angle": ..., "answer_pos": ...}, ...]

    개수가 맞지 않거나 한 문제라도 결정론적 검사에 걸리면 **세트 전체를**
    다시 만든다(최대 3회). 개별 재시도보다 비싸 보이지만, 한 번에 쓰게 하는
    것이 중복을 없애는 방법이라 그 대가를 받아들인다.
    """
    count = len(slots)
    level_guide = LEVEL_GUIDE.get(user.investment_level, LEVEL_GUIDE["미설정"])
    prompt = OX_BATCH_PROMPT if quiz_type == "OX" else MC_BATCH_PROMPT

    # 출력 예산은 문제 수에 비례해야 한다. 기본값(1문항 기준)으로 3문항을
    # 만들게 하면 JSON 이 중간에 잘리고 재시도 3회를 소진한다 (실측).
    # 사고 토큰은 호출당 한 번이므로 기본값을 두고 (count-1) 만큼만 더한다.
    base = settings.LLM_MAX_TOKENS or 4096
    budget = base + (count - 1) * settings.QUIZ_BATCH_TOKENS_PER_ITEM
    chain = prompt | get_llm(max_tokens=budget)

    prompt_vars = {
        "count": count,
        "topic": topic,
        "level": user.investment_level,
        "level_guide": level_guide,
        "level_style": level_style_for(user.investment_level),
        "topic_hint": f" ({topic_desc})" if topic_desc else "",
        "direction_block": format_direction_block(direction),
        "slot_block": _format_slot_block(slots, quiz_type),
    }

    for attempt in range(3):
        try:
            response = await chain.ainvoke(prompt_vars)
            data = _extract_json(response.content)
            items = data.get("questions")
            if not isinstance(items, list):
                raise ValueError("questions 배열이 없음")
            if len(items) != count:
                raise ValueError(f"문제 수 불일치: {len(items)}개 (요구 {count}개)")

            results = []
            for item, slot in zip(items, slots):
                if quiz_type == "OX":
                    check_ox_question(item, topic=topic, topic_desc=topic_desc)
                    results.append({
                        "question_type": "OX",
                        "question_text": item["question_text"],
                        "choices": [
                            {"choice_no": 1, "text": "O", "is_correct": item["answer"] == "O"},
                            {"choice_no": 2, "text": "X", "is_correct": item["answer"] == "X"},
                        ],
                        "explanation": item["explanation"],
                        "topic": item.get("topic", topic),
                        "level": user.investment_level,
                    })
                else:
                    check_mc_question(item, topic=topic, topic_desc=topic_desc,
                                      answer_pos=slot["answer_pos"])
                    results.append({
                        "question_type": "MULTIPLE_CHOICE",
                        "question_text": item["question_text"],
                        "choices": [
                            {"choice_no": c["no"], "text": c["text"],
                             "is_correct": c["no"] == item["correct_no"]}
                            for c in item["choices"]
                        ],
                        "explanation": item["explanation"],
                        "topic": item.get("topic", topic),
                        "level": user.investment_level,
                    })
            return results
        except Exception as e:
            logger.warning(
                f"묶음 생성 시도 {attempt + 1}/3 실패 "
                f"(topic={topic}, {quiz_type}, {count}개): {type(e).__name__}: {e}"
            )
            if attempt == 2:
                logger.error(f"묶음 생성 3회 소진 (topic={topic}, {count}개): {e}")
                raise ValueError(f"묶음 문제 생성 실패: {e}")

    raise ValueError("묶음 문제 생성 실패")


def chunk_slots(slots: list[dict], size: int) -> list[list[dict]]:
    """슬롯을 한 번에 만들 수 있는 크기로 나눈다.

    출력 토큰이 개수에 비례하므로 무한정 키울 수 없다. 나뉜 덩어리끼리는
    서로를 못 보므로 중복 가능성이 남지만, resolve_duplicates 가 그대로
    안전망 역할을 한다.
    """
    size = max(1, size)
    return [slots[i:i + size] for i in range(0, len(slots), size)]


def _get_random_topic(quiz_type: str) -> str:
    import random
    if quiz_type == "OX":
        return random.choice(OX_TOPICS)
    return random.choice(MC_TOPICS)


async def generate_ox_question(
    user: UserContext, topic: str, angle: str = "", avoid: list[str] | None = None,
    topic_desc: str = "", direction: str = "",
) -> dict:
    level_guide = LEVEL_GUIDE.get(user.investment_level, LEVEL_GUIDE["미설정"])
    chain = OX_PROMPT | _get_llm()
    prompt_vars = {
        "topic": topic,
        "level": user.investment_level,
        "level_guide": level_guide,
        "level_style": level_style_for(user.investment_level),
        "angle": angle or angle_for(0),
        "avoid_block": format_avoid_block(avoid or []),
        # 주제명만 주면 범위가 모호해 다른 개념으로 샌다(실측). 카탈로그 설명을 함께 준다.
        "topic_hint": f" ({topic_desc})" if topic_desc else "",
        "direction_block": format_direction_block(direction),
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
            # 재시도 사유를 반드시 남긴다. 남기지 않으면 서버 로그에 LLM 호출만
            # 반복해서 찍히고 왜 실패했는지는 어디에도 안 남는다 (2026-08-30 실측:
            # MULTIPLE_CHOICE + 중급 이 3회 소진으로 500 이 났는데 원인 추적 불가).
            # 최종 실패는 라우터가 HTTPException 으로 바꿔 던져 전역 핸들러를
            # 타지 않으므로, 여기서 로그를 남기지 않으면 영영 알 수 없다.
            logger.warning(
                f"OX 생성 시도 {attempt + 1}/3 실패 "
                f"(topic={topic}, level={user.investment_level}): {type(e).__name__}: {e}"
            )
            if attempt == 2:
                logger.error(
                    f"OX 생성 3회 소진 - 500 반환 "
                    f"(topic={topic}, level={user.investment_level}): {e}"
                )
                raise ValueError(f"OX 문제 생성 실패: {e}")
            continue


async def generate_mc_question(
    user: UserContext, topic: str, angle: str = "", avoid: list[str] | None = None,
    topic_desc: str = "", answer_pos: int | None = None, direction: str = "",
) -> dict:
    level_guide = LEVEL_GUIDE.get(user.investment_level, LEVEL_GUIDE["미설정"])
    chain = MC_PROMPT | _get_llm()
    # 지정하지 않으면 정답이 1번에 몰린다(실측 78.7%). 반드시 자리를 정해준다.
    answer_pos = answer_pos or answer_pos_for(0)
    prompt_vars = {
        "topic": topic,
        "level": user.investment_level,
        "level_guide": level_guide,
        "level_style": level_style_for(user.investment_level),
        "angle": angle or angle_for(0),
        "avoid_block": format_avoid_block(avoid or []),
        # 주제명만 주면 범위가 모호해 다른 개념으로 샌다(실측). 카탈로그 설명을 함께 준다.
        "topic_hint": f" ({topic_desc})" if topic_desc else "",
        "direction_block": format_direction_block(direction),
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
            # 재시도 사유를 반드시 남긴다. 남기지 않으면 서버 로그에 LLM 호출만
            # 반복해서 찍히고 왜 실패했는지는 어디에도 안 남는다 (2026-08-30 실측:
            # MULTIPLE_CHOICE + 중급 이 3회 소진으로 500 이 났는데 원인 추적 불가).
            # 최종 실패는 라우터가 HTTPException 으로 바꿔 던져 전역 핸들러를
            # 타지 않으므로, 여기서 로그를 남기지 않으면 영영 알 수 없다.
            logger.warning(
                f"객관식 생성 시도 {attempt + 1}/3 실패 "
                f"(topic={topic}, level={user.investment_level}): {type(e).__name__}: {e}"
            )
            if attempt == 2:
                logger.error(
                    f"객관식 생성 3회 소진 - 500 반환 "
                    f"(topic={topic}, level={user.investment_level}): {e}"
                )
                raise ValueError(f"객관식 문제 생성 실패: {e}")
            continue


async def generate_quiz(
    user: UserContext, quiz_type: str, topic: str = None,
    angle: str = "", avoid: list[str] | None = None, topic_desc: str = "",
    answer_pos: int | None = None, direction: str = "",
) -> dict:
    """문제 1개 생성.

    angle/avoid는 같은 주제로 여러 문제를 만들 때 중복을 막기 위한 것이다.
    (angle = 이번 문제의 출제 방향, avoid = 이미 만든 문제 목록)
    생략하면 첫 번째 관점을 쓴다 - 단건 생성에서는 중복 걱정이 없다.

    topic_desc/direction 을 주지 않으면 카탈로그에서 주제명으로 찾아 채운다.
    (topic_desc = 주제 범위를 좁히는 설명, direction = 카테고리의 출제 방향)
    """
    if not topic:
        topic = _get_random_topic(quiz_type)

    # 호출자가 명시하지 않았으면 카탈로그에서 문맥을 찾는다 (LLM 호출 없는 이름 매칭).
    # 못 찾으면 빈 문자열이라 기존 동작 그대로다.
    if not topic_desc and not direction:
        topic_desc, direction = find_topic_context(topic, user.investment_level)

    if quiz_type == "OX":
        return await generate_ox_question(
            user, topic, angle, avoid, topic_desc, direction)
    elif quiz_type == "MULTIPLE_CHOICE":
        return await generate_mc_question(
            user, topic, angle, avoid, topic_desc, answer_pos, direction)
    else:
        raise ValueError(f"지원하지 않는 문제 유형: {quiz_type}")


async def generate_quiz_batch(
    user: UserContext, quiz_type: str, topic: str = None, count: int = 1,
    topic_desc: str = "",
) -> list[dict]:
    """같은 topic으로 count개의 문제를 **동시에** 생성한다.

    예전에는 순차 생성이었다. 근거는 "로컬 Ollama 가 단일 모델 인스턴스라 동시
    요청을 병렬로 못 받는다" 였는데, 서빙이 vLLM(연속 배칭)으로 바뀌면서 무효가
    됐다 - 리허설 실측으로 동시 20건이 p50=p95=15.0초에 전부 성공했고 KV 캐시
    여유가 49배였다. 순차 생성은 응답 시간만 개수에 비례해 늘리고 있었다.

    출제 관점(angle)과 정답 위치(answer_pos)는 인덱스만으로 정해지므로 앞
    문제의 결과를 기다릴 이유가 애초에 없었다. 순차 생성이 주던 유일한 이점은
    "이미 만든 문제를 피하라"는 avoid 힌트인데, 그건 세트가 나온 뒤 실제로
    겹친 것만 골라 되갚는다(concurrency.resolve_duplicates).

    자세한 배경은 concurrency.py 모듈 독스트링 참고.
    """
    if not topic:
        topic = _get_random_topic(quiz_type)

    # 주제명으로 카탈로그를 찾아 설명·출제 방향을 채운다 (못 찾으면 빈 문자열).
    # 세트 전체가 같은 주제이므로 한 번만 찾으면 된다.
    catalog_desc, direction = find_topic_context(topic, user.investment_level)
    topic_desc = topic_desc or catalog_desc

    # 과거에 이 주제로 만든 문제 수만큼 관점을 밀어, 재요청 시 다른 관점부터 시작한다
    offset = await bank.angle_offset(topic, user.user_id)

    # 문제마다 관점과 정답 위치를 미리 확정한다. 둘 다 인덱스의 함수라
    # 생성 순서와 무관하다 - 이것이 병렬화가 성립하는 근거다.
    # 관점은 5주기, 정답 위치는 4주기라 서로소다. 20문제가 지나야 조합이 반복된다.
    # 주제마다 시작점을 밀어, 5문제 중 두 번 걸리는 자리가 주제별로 달라지게 한다.
    plan = [
        (angle_for(offset + i), answer_pos_for(offset + i + answer_pos_offset(topic)))
        for i in range(count)
    ]
    # 1단계는 **묶음 호출**이다. 한 번에 여러 문제를 쓰게 해야 모델이 서로를
    # 보고 다르게 만든다 (하나씩 병렬로 만들면 중복 발동률 77% - 위 주석 참고).
    # QUIZ_BATCH_MAX 를 넘으면 여러 덩어리로 나눠 병렬 호출한다.
    slots = [{"angle": a, "answer_pos": p} for a, p in plan]
    chunks = chunk_slots(slots, settings.QUIZ_BATCH_MAX)

    def chunk_maker(chunk: list[dict]):
        async def _make() -> list[dict]:
            return await generate_quiz_chunk(
                user, quiz_type, topic, chunk, topic_desc, direction)
        return _make

    grouped = await generate_all([chunk_maker(c) for c in chunks])
    questions = [q for group in grouped for q in group]

    # 1단계에는 avoid 힌트가 없었으므로 겹친 것만 골라 다시 만든다.
    # 같은 관점으로 다시 만들면 같은 문제가 또 나온다(실측: ROE 5문제에서 서로 다른
    # 관점인데도 유사도 0.979). 재시도는 세트 밖 관점으로 민다.
    async def retry_dup(i: int, avoid: list[str]) -> dict:
        # 재생성은 단건 경로를 쓴다. 걸린 문제 하나만 다시 만들면 되고,
        # 이때는 avoid 에 나머지 본문이 채워져 있어 단건으로도 충분히 갈린다.
        return await generate_quiz(
            user, quiz_type, topic, angle_for(offset + i + count), avoid,
            topic_desc, plan[i][1], direction)

    await resolve_duplicates(questions, retry_dup, user)

    # 검증 불합격 시 같은 조건으로 재생성하기 위해 생성 인자를 보관한다.
    # avoid 는 확정된 세트 전체 - 재생성본이 나머지와 겹치지 않게 한다.
    final_texts = [q["question_text"] for q in questions]
    gen_args = [
        (quiz_type, topic, plan[i][0],
         [t for j, t in enumerate(final_texts) if j != i],
         topic_desc, plan[i][1], direction)
        for i in range(count)
    ]

    await _revalidate_and_fix(questions, gen_args, user)
    # 검증·재생성이 끝난 확정본만 저장한다
    await bank.register(questions, user.user_id)
    return questions


async def _revalidate_and_fix(
    questions: list[dict], gen_args: list[tuple], user: UserContext,
) -> None:
    """정답/해설이 모순된 문제를 찾아 같은 조건으로 1회 재생성한다 (in-place 수정).

    세트 전체를 LLM 1회 호출로 검사하고, 불합격 건은 **동시에** 다시 만든다.
    (예전에는 불합격 건을 순차로 재생성해, 3건이 걸리면 3번을 직렬로 더 기다렸다.)
    재생성본은 다시 검증하지 않는다 - 검증 호출이 계속 늘어나는 것을 막기 위한 타협.
    """
    mode = (settings.QUIZ_VALIDATE_MODE or "blocking").strip().lower()
    if mode == "off":
        return
    if mode == "background":
        # 응답을 막지 않는다. 계측상 이 단계가 프롬프트 퀴즈 지연의 30% 였다.
        fire_and_forget(report_validation(list(questions), "quiz"), "사후 검증")
        return

    failed = await validate_questions(questions)

    if not failed:
        logger.info("문제 검증 통과 (전체 합격)")
        return

    logger.warning(f"검증 불합격 {len(failed)}건 - 재생성: {list(failed.values())}")

    async def retry_failed(idx: int) -> dict:
        quiz_type, topic, angle, avoid, topic_desc, answer_pos, direction = gen_args[idx]
        return await generate_quiz(
            user, quiz_type, topic, angle, avoid, topic_desc, answer_pos, direction)

    await regenerate_failed(questions, failed, retry_failed)


async def report_validation(questions: list[dict], label: str) -> None:
    """검사만 하고 결과를 로그로 남긴다 (background / 재생성 없음).

    응답은 이미 나갔으므로 고칠 수 없다. 목적은 "요즘도 걸리는 게 있는가"를
    계속 관찰하는 것이다. 며칠 로그를 보고 blocking 으로 되돌릴지, off 로
    완전히 뺄지 정하면 된다.
    """
    failed = await validate_questions(questions)
    if not failed:
        logger.info(f"[{label}] 사후 검증 통과 (전체 합격)")
        return
    for idx, reason in failed.items():
        logger.warning(
            f"[{label}] 사후 검증 불합격 {idx + 1}번: {reason} "
            f"| 문제: {questions[idx].get('question_text', '')[:60]}"
        )


# =========================================================
# 프롬프트 기반 문제 생성
# =========================================================

# 생성은 병렬이 됐지만 개수가 늘면 여전히 느려진다. 동시 실행 상한
# (QUIZ_GEN_CONCURRENCY=5)을 넘는 개수는 여러 파도로 나뉘고, 파도 하나가
# 라운드 하나이기 때문이다. count=10 이면 분석 1 + 생성 2파도 + 검증 1 = 4라운드.
#
# ⚠️ 파드 실측 기준 라운드당 약 22초라 count=10 은 약 88초다. RunPod 프록시가
# 100초에서 524 로 끊으므로 상한 요청은 위험 구간에 있다. 백엔드에서 사용자가
# 고를 수 있는 개수를 5 이하로 제한하는 것을 권장한다.
MAX_PROMPT_QUIZ_COUNT = 10

# 프롬프트에 개수 언급이 없을 때의 기본값. settings 에서 읽는다
# (발표 당일 재빌드 없이 템플릿 env 로 조절할 수 있어야 하므로).
def _default_prompt_quiz_count() -> int:
    return max(1, min(settings.PROMPT_QUIZ_DEFAULT_COUNT, MAX_PROMPT_QUIZ_COUNT))

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
        # 출제 방향까지 보여준다. 소재가 겹치는 카테고리를 가르는 결정적인 단서다.
        # (예: 중급 "보조지표 개념"과 고급 "보조지표 활용 판단"은 소재가 같고
        #  방향만 "의미 이해" / "투자 판단에 활용"으로 다르다)
        direction = f" [출제 방향: {cat.problem_direction}]" if cat.problem_direction else ""
        lines.append(f"{level}[{cat.category_code}] {cat.category_name}{desc}{direction}")
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
    chain = ANALYZE_PROMPT | _analyze_llm()
    catalog = _format_catalog(categories)

    data = None
    for attempt in range(3):
        try:
            response = await chain.ainvoke({
                "prompt": prompt,
                "catalog": catalog,
                "default_count": _default_prompt_quiz_count(),
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
    chain = MAP_TOPICS_PROMPT | _analyze_llm()
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

    # 같은 주제가 여러 항목에 걸쳐 나오므로 주제별로 출제 관점을 돌린다
    # (관점 없이 돌리면 "배당수익률 = 배당금/주가" 하나를 표현만 바꿔 반복함 - 실측).
    # 주제별로 과거 생성 이력만큼 관점을 밀어둔다 (재요청 시 중복 방지).
    # Qdrant 왕복이므로 주제 수만큼 동시에 조회한다.
    unique_topics = list({item["topic"] for item in plan})
    offsets = await asyncio.gather(
        *(bank.angle_offset(t, user.user_id) for t in unique_topics)
    )
    topic_slot: dict[str, int] = defaultdict(int, dict(zip(unique_topics, offsets)))

    # 생성 인자를 먼저 전부 확정한다. 관점은 주제별로, 정답 위치는 세트 전체
    # 순번으로 돈다 (주제별로 돌리면 주제가 전부 다를 때 모두 0번째라 정답이
    # 1번에 몰린다). 어느 쪽도 앞 문제의 결과에 의존하지 않으므로 병렬이 가능하다.
    specs: list[dict] = []
    for set_index, item in enumerate(plan):
        topic = item["topic"]
        angle = angle_for(topic_slot[topic])
        topic_slot[topic] += 1

        # 1단계 분석이 정해준 카테고리에서 주제 설명과 출제 방향을 꺼내 생성에 넘긴다.
        # 이걸 넘기지 않으면 카테고리는 저장용 라벨로만 쓰이고, 정작 문제 내용에는
        # 카테고리가 의도한 출제 방향이 반영되지 않는다.
        primary_detail = item["detail_codes"][0] if item["detail_codes"] else None
        specs.append({
            "quiz_type": item["question_type"],
            "topic": topic,
            "angle": angle,
            "topic_desc": get_detail_description(item["category_code"], primary_detail),
            "direction": get_problem_direction(item["category_code"]),
            "answer_pos": answer_pos_for(set_index),
            "category_code": item["category_code"],
            "detail_codes": item["detail_codes"],
            "primary_detail_code": primary_detail,
        })

    # 같은 (주제, 유형)끼리 묶어 한 번의 호출로 만든다. 중복은 같은 주제 안에서만
    # 생기므로 묶는 단위도 그것이다. 주제가 다르면 애초에 겹치지 않는다.
    groups: dict[tuple, list[int]] = {}
    for i, spec in enumerate(specs):
        groups.setdefault((spec["topic"], spec["quiz_type"]), []).append(i)

    jobs, job_indexes = [], []
    for (topic, quiz_type), idxs in groups.items():
        base = specs[idxs[0]]
        for chunk in chunk_slots(
            [{"angle": specs[i]["angle"], "answer_pos": specs[i]["answer_pos"]}
             for i in idxs],
            settings.QUIZ_BATCH_MAX,
        ):
            n = len(chunk)
            take, idxs = idxs[:n], idxs[n:]
            job_indexes.append(take)

            def maker(topic=topic, quiz_type=quiz_type, chunk=chunk, base=base):
                async def _make() -> list[dict]:
                    return await generate_quiz_chunk(
                        user, quiz_type, topic, chunk,
                        base["topic_desc"], base["direction"])
                return _make
            jobs.append(maker())

    grouped = await generate_all(jobs)

    # 원래 순서대로 되돌린다 (정답 위치·카테고리 매핑이 인덱스에 묶여 있다)
    questions: list[dict] = [None] * len(specs)
    for idxs, group in zip(job_indexes, grouped):
        for i, q in zip(idxs, group):
            questions[i] = q

    # 1단계에는 avoid 힌트가 없었으므로 겹친 것만 골라 다시 만든다.
    # 재시도는 세트 밖 관점으로 민다 - 같은 관점이면 같은 문제가 또 나온다.
    async def retry_dup(i: int, avoid: list[str]) -> dict:
        # 재생성은 단건 경로 (걸린 하나만, avoid 가 채워진 상태로)
        spec = specs[i]
        return await generate_quiz(
            user, spec["quiz_type"], spec["topic"],
            angle_for(topic_slot[spec["topic"]] + i), avoid,
            spec["topic_desc"], spec["answer_pos"], spec["direction"])

    await resolve_duplicates(questions, retry_dup, user)

    for question, spec in zip(questions, specs):
        question["category_code"] = spec["category_code"]
        question["detail_codes"] = spec["detail_codes"]
        question["primary_detail_code"] = spec["primary_detail_code"]

    final_texts = [q["question_text"] for q in questions]
    gen_args = [
        (spec["quiz_type"], spec["topic"], spec["angle"],
         [t for j, t in enumerate(final_texts) if j != i],
         spec["topic_desc"], spec["answer_pos"], spec["direction"])
        for i, spec in enumerate(specs)
    ]

    await _revalidate_and_fix(questions, gen_args, user)
    await bank.register(questions, user.user_id)
    return questions
