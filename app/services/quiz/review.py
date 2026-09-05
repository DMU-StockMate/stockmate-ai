"""오답 기반 문제 생성.

유저가 평소에 틀린 문제들을 받아서, 자주 틀린 주제 위주로 새 문제 5개를 만든다.

설계 메모
- 특정 문제 1개를 골라 변형하는 게 아니라, 누적 오답을 주제별로 묶어
  많이 틀린 주제에 문제를 더 많이 배분한다.
- 주제 판별은 detail_code(quiz_question_details.is_primary=TRUE) 우선.
  이 값이 없으면 topic 문자열, 그것도 없으면 문제 본문 앞부분으로 그룹핑한다.
- 응답은 PromptQuizQuestion과 같은 형태(category_code/detail_codes/primary_detail_code)라
  NestJS는 /quiz/generate/prompt 와 동일한 저장 로직을 재사용하면 된다.
- 로컬 Ollama가 단일 인스턴스라 generate_quiz_batch와 같은 이유로 순차 생성한다.
"""
import asyncio
from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate

from app.core.logger import setup_logger
from app.schemas.chat import UserContext
from app.schemas.quiz import WrongAnswerItem
from app.services.quiz import bank
from app.services.quiz.categories import QUIZ_CATEGORY_CATALOG
from app.services.quiz.generator import LEVEL_GUIDE, _map_topics_to_catalog
from app.services.quiz.concurrency import (
    generate_all,
    regenerate_failed,
    resolve_duplicates,
)
from app.services.quiz.quality import (
    ANGLE_BLOCK,
    GENERATION_ANGLES,
    MC_QUALITY_RULES,
    OX_QUALITY_RULES,
    ANSWER_POSITION_BLOCK,
    QUALITY_RULES,
    angle_for,
    answer_pos_for,
    check_mc_question,
    check_ox_question,
    extract_json,
    format_avoid_block,
    get_llm,
    validate_questions,
)

logger = setup_logger(__name__)


class NoWrongAnswerError(ValueError):
    """오답 데이터가 비어 있을 때 (라우터에서 422로 매핑)."""


# 고정 5문제 (팀 결정). 순차 생성이라 개수를 늘리면 응답 시간이 비례해 늘어난다.
REVIEW_QUIZ_COUNT = 5

# 한 주제당 프롬프트에 넣을 오답 예시 수.
# 너무 많이 넣으면 컨텍스트가 길어지고 LLM이 예시를 그대로 베끼는 경향이 생긴다.
MAX_SAMPLES_PER_TOPIC = 2

# 출제 관점은 quality.GENERATION_ANGLES 를 공용으로 쓴다
# (일반/프롬프트 기반 생성도 같은 관점 로테이션을 사용).
REVIEW_ANGLES = GENERATION_ANGLES


# 오답 기반 생성에만 필요한 규칙.
# 유형 공통 품질 규칙은 quality.QUALITY_RULES 에 있고 아래에서 합쳐 쓴다.
_REVIEW_ONLY_RULES = """- 주제: {topic}
  반드시 이 개념을 유지하세요. 다른 개념으로 바꾸거나 여러 개념을 조합하지 마세요.
- 난이도: {level} ({level_guide})
- 위 틀린 문제를 그대로 베끼지 마세요. 상황, 수치, 문장 구조를 모두 바꾸세요.
- 원본 문제가 가르치려던 결론을 뒤집지 마세요.
  원본이 "A라고 단정할 수 없다"를 가르쳤다면, 새 문제도 "A라고 단정할 수 없다" 쪽이어야 합니다.
- 해설에 "학생이", "학생은", "당신이 고른", "오답으로 선택한" 같은 표현을 절대 쓰지 마세요.
  이 문제는 나중에 다시 출제되므로, 문제와 해설만 읽어도 완결되는 일반적인 서술이어야 합니다.
  틀리기 쉬운 지점은 "~라고 생각하기 쉽지만" 같은 표현으로 자연스럽게 녹이세요.
- 위 "이미 출제된 문제"와 같은 내용을 묻지 마세요. 반드시 다른 측면을 물어야 합니다."""

_COMMON_RULES = _REVIEW_ONLY_RULES + "\n" + QUALITY_RULES


REVIEW_MC_PROMPT = ChatPromptTemplate.from_template("""
당신은 주식 투자 교육 전문가입니다.
아래는 학생이 실제로 틀린 문제입니다.
같은 개념을 다시 점검할 수 있는 새로운 4지선다 객관식 문제를 1개 생성하세요.

[학생이 틀린 문제]
{wrong_context}

조건:
""" + _COMMON_RULES + """
""" + MC_QUALITY_RULES + """
- 학생이 고른 오답에 담긴 오해를 오답 선택지 중 하나로 배치하세요.
""" + ANGLE_BLOCK + ANSWER_POSITION_BLOCK + """
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


REVIEW_OX_PROMPT = ChatPromptTemplate.from_template("""
당신은 주식 투자 교육 전문가입니다.
아래는 학생이 실제로 틀린 문제입니다.
같은 개념을 다시 점검할 수 있는 새로운 OX 문제를 1개 생성하세요.

[학생이 틀린 문제]
{wrong_context}

조건:
""" + _COMMON_RULES + """
""" + OX_QUALITY_RULES + """
""" + ANGLE_BLOCK + """
JSON 형식:
{{
  "question_text": "문제 내용 (평서문, 물음표로 끝나면 안 됨)",
  "answer": "O" 또는 "X",
  "explanation": "해설 내용 (2~3문장)",
  "topic": "{topic}"
}}
""")


# =========================================================
# 카탈로그 조회
# =========================================================

def _build_detail_index() -> dict[str, tuple[str, str]]:
    """detail_code → (category_code, detail_name) 역인덱스."""
    index: dict[str, tuple[str, str]] = {}
    for cat in QUIZ_CATEGORY_CATALOG:
        for d in cat.details:
            index[d.detail_code] = (cat.category_code, d.detail_name)
    return index


_DETAIL_INDEX = _build_detail_index()


# =========================================================
# 오답 그룹핑
# =========================================================

def _group_key(item: WrongAnswerItem) -> str:
    """오답을 묶을 주제 키. detail_code > topic > 문제 본문 앞부분 순으로 사용한다."""
    if item.detail_code and item.detail_code in _DETAIL_INDEX:
        return f"detail:{item.detail_code}"
    if item.topic and item.topic.strip():
        return f"topic:{item.topic.strip()}"
    return f"text:{item.question_text.strip()[:20]}"


def _topic_label(key: str, items: list[WrongAnswerItem]) -> str:
    """LLM 프롬프트에 넣을 주제명. detail_code면 카탈로그의 한글 이름을 쓴다."""
    if key.startswith("detail:"):
        return _DETAIL_INDEX[key.split(":", 1)[1]][1]
    if key.startswith("topic:"):
        return key.split(":", 1)[1]
    # 주제 정보가 아예 없는 경우 - 문제 본문을 주제 대신 넘긴다.
    return items[0].topic or "투자 기본 개념"


def group_wrong_answers(wrong_answers: list[WrongAnswerItem]) -> list[dict]:
    """오답을 주제별로 묶고 많이 틀린 순으로 정렬한다.

    반환: [{"key", "topic", "detail_code", "wrong_count", "samples"}, ...]
    """
    groups: dict[str, list[WrongAnswerItem]] = defaultdict(list)
    for item in wrong_answers:
        groups[_group_key(item)].append(item)

    result = []
    for key, items in groups.items():
        detail_code = key.split(":", 1)[1] if key.startswith("detail:") else None
        result.append({
            "key": key,
            "topic": _topic_label(key, items),
            "detail_code": detail_code,
            "wrong_count": len(items),
            "samples": items[:MAX_SAMPLES_PER_TOPIC],
            # 유형이 섞여 있으면 다수결. OX만 틀렸으면 OX로 복습시킨다.
            "question_type": _majority_type(items),
        })

    # 많이 틀린 주제 우선. 동점이면 주제명으로 안정 정렬.
    result.sort(key=lambda g: (-g["wrong_count"], g["topic"]))
    return result


def _majority_type(items: list[WrongAnswerItem]) -> str:
    ox = sum(1 for i in items if i.question_type == "OX")
    return "OX" if ox > len(items) / 2 else "MULTIPLE_CHOICE"


def allocate_slots(groups: list[dict], count: int = REVIEW_QUIZ_COUNT) -> list[dict]:
    """count개의 생성 슬롯을 오답 수에 비례해 주제에 배분한다 (최대잔여법).

    예) PER 3회 / RSI 1회 / 배당 1회, 5문제 → PER 3, RSI 1, 배당 1

    라운드로빈으로 돌리면 위 예시가 2:2:1이 되어 "많이 틀린 주제를 더 많이"라는
    의도가 깨지므로 비례 배분을 쓴다.
    주제가 슬롯보다 많으면 많이 틀린 주제부터 count개만 사용하고,
    비중이 아주 작은 주제는 슬롯을 못 받고 빠질 수 있다 (의도된 동작).
    """
    if not groups:
        return []

    groups = groups[:count]
    total = sum(g["wrong_count"] for g in groups)
    if total <= 0:  # 방어적 처리 - 정상 경로에서는 발생하지 않음
        return [groups[i % len(groups)] for i in range(count)]

    # [그룹, 정수 몫, 소수부]
    quotas: list[list] = []
    for g in groups:
        exact = count * g["wrong_count"] / total
        base = int(exact)
        quotas.append([g, base, exact - base])

    # 정수 몫만으로는 count에 모자라므로 소수부가 큰 순서로 남은 슬롯을 나눠준다
    assigned = sum(q[1] for q in quotas)
    for q in sorted(quotas, key=lambda x: -x[2])[:count - assigned]:
        q[1] += 1

    slots: list[dict] = []
    for g, n, _ in quotas:
        slots.extend([g] * n)
    return slots[:count]


# =========================================================
# 프롬프트 컨텍스트 구성
# =========================================================

def format_wrong_context(samples: list[WrongAnswerItem]) -> str:
    """오답 예시를 프롬프트에 넣을 텍스트로 변환한다.

    유저가 고른 선택지를 명시적으로 표시해야 LLM이 '무엇을 오해했는지'를
    잡아낼 수 있다. 이 표시가 없으면 그냥 비슷한 문제만 나온다.
    """
    blocks = []
    for idx, s in enumerate(samples, 1):
        lines = [f"({idx}) 문제: {s.question_text}"]

        if s.choices:
            lines.append("    선택지:")
            for c in s.choices:
                marks = []
                if c.is_correct:
                    marks.append("정답")
                if s.selected_choice_no is not None and c.choice_no == s.selected_choice_no:
                    marks.append("학생이 고른 답")
                suffix = f"  <- {', '.join(marks)}" if marks else ""
                lines.append(f"      {c.choice_no}. {c.text}{suffix}")

        if s.explanation:
            lines.append(f"    해설: {s.explanation}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


# =========================================================
# 문제 생성
# =========================================================

async def _generate_review_mc(
    user: UserContext, topic: str, wrong_context: str, avoid: list[str], angle: str,
    answer_pos: int | None = None,
) -> dict:
    level_guide = LEVEL_GUIDE.get(user.investment_level, LEVEL_GUIDE["미설정"])
    chain = REVIEW_MC_PROMPT | get_llm()

    for attempt in range(3):
        try:
            response = await chain.ainvoke({
                "topic": topic,
                "level": user.investment_level,
                "level_guide": level_guide,
                "wrong_context": wrong_context,
                "avoid_block": format_avoid_block(avoid),
                "angle": angle,
                # 지정하지 않으면 정답이 1번에 몰린다(실측 78.7%)
                "answer_pos": answer_pos or answer_pos_for(0),
            })
            data = extract_json(response.content)

            check_mc_question(data, learner_reference_check=True, topic=topic,
                              answer_pos=answer_pos)

            return {
                "question_type": "MULTIPLE_CHOICE",
                "question_text": data["question_text"],
                "choices": [
                    {
                        "choice_no": c["no"],
                        "text": c["text"],
                        "is_correct": c["no"] == data["correct_no"],
                    }
                    for c in data["choices"]
                ],
                "explanation": data["explanation"],
                "topic": data.get("topic", topic),
                "level": user.investment_level,
            }
        except Exception as e:
            if attempt == 2:
                raise ValueError(f"오답 기반 객관식 문제 생성 실패: {e}")
            continue


async def _generate_review_ox(
    user: UserContext, topic: str, wrong_context: str, avoid: list[str], angle: str,
    answer_pos: int | None = None,   # OX 는 선택지가 2개뿐이라 쓰지 않는다 (호출부 통일용)
) -> dict:
    level_guide = LEVEL_GUIDE.get(user.investment_level, LEVEL_GUIDE["미설정"])
    chain = REVIEW_OX_PROMPT | get_llm()

    for attempt in range(3):
        try:
            response = await chain.ainvoke({
                "topic": topic,
                "level": user.investment_level,
                "level_guide": level_guide,
                "wrong_context": wrong_context,
                "avoid_block": format_avoid_block(avoid),
                "angle": angle,
            })
            data = extract_json(response.content)

            check_ox_question(data, learner_reference_check=True, topic=topic)

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
                raise ValueError(f"오답 기반 OX 문제 생성 실패: {e}")
            continue


# =========================================================
# 카테고리 매핑
# =========================================================

async def _resolve_categories(groups: list[dict]) -> dict[str, dict]:
    """주제 그룹 → {group_key: {"category_code", "detail_codes"}}.

    detail_code를 받은 그룹은 카탈로그에서 바로 역추적하므로 LLM 호출이 필요 없다.
    detail_code가 없는 그룹만 모아 기존 _map_topics_to_catalog로 한 번에 매핑한다.
    매핑 실패는 치명적이지 않다 (category_code=null 허용 — 기존 프롬프트 생성과 동일).

    프롬프트 기반 생성과 달리 등급별 카탈로그로 좁히지 않고 전체를 대상으로 매핑한다.
    유저가 틀린 문제는 이미 출제된 문제이므로 자기 등급 밖 주제일 수 있기 때문이다.
    """
    resolved: dict[str, dict] = {}
    unmapped_topics: dict[str, list[str]] = defaultdict(list)

    for g in groups:
        if g["detail_code"]:
            category_code, _ = _DETAIL_INDEX[g["detail_code"]]
            resolved[g["key"]] = {
                "category_code": category_code,
                "detail_codes": [g["detail_code"]],
            }
        else:
            unmapped_topics[g["topic"]].append(g["key"])

    if unmapped_topics:
        logger.info(f"오답 주제 카테고리 매핑 필요: {list(unmapped_topics)}")
        mappings = await _map_topics_to_catalog(
            list(unmapped_topics), QUIZ_CATEGORY_CATALOG
        )
        for topic, keys in unmapped_topics.items():
            mapping = mappings.get(topic)
            for key in keys:
                resolved[key] = mapping or {"category_code": None, "detail_codes": []}

    return resolved


# =========================================================
# 파이프라인
# =========================================================

async def generate_quiz_from_wrong_answers(
    user: UserContext,
    wrong_answers: list[WrongAnswerItem],
) -> list[dict]:
    """오답 기반 문제 생성 파이프라인.

    1. 오답을 주제별로 묶고 많이 틀린 순으로 정렬
    2. 5개 슬롯을 주제에 배분 (많이 틀린 주제가 더 많이 가져감)
    3. 슬롯마다 해당 주제의 오답을 컨텍스트로 넣어 순차 생성
    4. 카테고리 코드 매핑해서 반환 (NestJS 저장용)
    """
    if not wrong_answers:
        raise NoWrongAnswerError("오답 데이터가 없습니다.")

    groups = group_wrong_answers(wrong_answers)
    slots = allocate_slots(groups, REVIEW_QUIZ_COUNT)
    categories = await _resolve_categories(groups)

    summary = ", ".join("{}x{}".format(g["topic"], g["wrong_count"]) for g in groups)
    logger.info(
        f"오답 기반 생성: 오답 {len(wrong_answers)}개 → 주제 {len(groups)}개 "
        f"[{summary}] → 문제 {REVIEW_QUIZ_COUNT}개"
    )

    # 주제별로 과거 생성 이력만큼 관점을 밀어둔다.
    # 오답 이력이 그대로면 그룹·배분·샘플이 모두 같아 1차와 똑같은 세트가 나오므로
    # 관점을 옮기지 않으면 재요청이 사실상 무의미해진다.
    # Qdrant 왕복이므로 그룹 수만큼 동시에 조회한다.
    keys = [g["key"] for g in groups]
    offsets = await asyncio.gather(
        *(bank.angle_offset(g["topic"], user.user_id) for g in groups)
    )
    slot_index: dict[str, int] = defaultdict(int, dict(zip(keys, offsets)))

    # 생성 인자를 먼저 전부 확정한다. 슬롯마다 오답 샘플·관점·정답 위치가
    # 인덱스만으로 정해지므로 앞 문제의 결과를 기다릴 이유가 없다.
    specs: list[dict] = []
    for set_index, group in enumerate(slots):
        key = group["key"]
        samples = group["samples"]

        # 같은 주제에 슬롯이 여러 개면 오답 샘플을 번갈아 사용한다.
        # 매번 같은 컨텍스트를 넣으면 LLM이 사실상 동일한 문제를 반복 생성한다(실측).
        if samples:
            sample = samples[slot_index[key] % len(samples)]
            wrong_context = format_wrong_context([sample])
        else:
            wrong_context = "(오답 상세 정보 없음 - 주제만 참고)"

        # 같은 주제의 n번째 슬롯 -> n번째 출제 관점 (결정론적 순환)
        angle = angle_for(slot_index[key])
        slot_index[key] += 1

        mapping = categories.get(key) or {}
        detail_codes = mapping.get("detail_codes") or []
        specs.append({
            "generate": (
                _generate_review_ox
                if group["question_type"] == "OX"
                else _generate_review_mc
            ),
            "topic": group["topic"],
            "key": key,
            "wrong_context": wrong_context,
            "angle": angle,
            # 정답 위치는 주제별이 아니라 **세트 전체 순번**으로 돌린다.
            # 주제별로 돌리면 주제가 5개일 때 모두 0번째 슬롯이라 전부 1번이 된다.
            "answer_pos": answer_pos_for(set_index),
            "category_code": mapping.get("category_code"),
            "detail_codes": detail_codes,
            "primary_detail_code": detail_codes[0] if detail_codes else None,
        })

    def make(i: int, avoid: list[str], angle: str = ""):
        spec = specs[i]

        async def _make() -> dict:
            return await spec["generate"](
                user, spec["topic"], spec["wrong_context"], avoid,
                angle or spec["angle"], spec["answer_pos"])

        return _make

    questions = await generate_all([make(i, []) for i in range(len(specs))])

    # 1단계에는 avoid 힌트가 없었으므로 겹친 것만 골라 다시 만든다.
    # 재시도는 세트 밖 관점으로 민다 - 같은 관점이면 같은 문제가 또 나온다.
    async def retry_dup(i: int, avoid: list[str]) -> dict:
        return await make(
            i, avoid, angle_for(slot_index[specs[i]["key"]] + REVIEW_QUIZ_COUNT))()

    await resolve_duplicates(questions, retry_dup, user)

    for question, spec in zip(questions, specs):
        question["category_code"] = spec["category_code"]
        question["detail_codes"] = spec["detail_codes"]
        question["primary_detail_code"] = spec["primary_detail_code"]

    # 검증에서 불합격 시 같은 조건으로 재생성하기 위해 생성 인자를 보관한다.
    # avoid 는 확정된 세트 전체 - 재생성본이 나머지와 겹치지 않게 한다.
    final_texts = [q["question_text"] for q in questions]
    gen_args = [
        (spec["generate"], spec["topic"], spec["wrong_context"],
         [t for j, t in enumerate(final_texts) if j != i],
         spec["angle"], spec["answer_pos"])
        for i, spec in enumerate(specs)
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
    재생성본은 다시 검증하지 않는다. 검증 호출이 계속 늘어나는 것을 막기 위한
    타협이며, 재생성으로도 안 고쳐지면 원본보다 나빠질 이유는 없으므로 채택한다.
    """
    failed = await validate_questions(questions)

    if not failed:
        logger.info("문제 검증 통과 (전체 합격)")
        return

    logger.warning(f"검증 불합격 {len(failed)}건 - 재생성: {list(failed.values())}")

    async def retry_failed(idx: int) -> dict:
        generate, topic, wrong_context, avoid, angle, answer_pos = gen_args[idx]
        return await generate(user, topic, wrong_context, avoid, angle, answer_pos)

    await regenerate_failed(questions, failed, retry_failed)
