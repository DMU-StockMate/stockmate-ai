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
import re
from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate

from app.core.logger import setup_logger
from app.schemas.chat import UserContext
from app.schemas.quiz import WrongAnswerItem
from app.services.quiz.categories import QUIZ_CATEGORY_CATALOG
from app.services.quiz.generator import (
    LEVEL_GUIDE,
    _extract_json,
    _get_llm,
    _map_topics_to_catalog,
)

logger = setup_logger(__name__)


class NoWrongAnswerError(ValueError):
    """오답 데이터가 비어 있을 때 (라우터에서 422로 매핑)."""


# 고정 5문제 (팀 결정). 순차 생성이라 개수를 늘리면 응답 시간이 비례해 늘어난다.
REVIEW_QUIZ_COUNT = 5

# 한 주제당 프롬프트에 넣을 오답 예시 수.
# 너무 많이 넣으면 컨텍스트가 길어지고 LLM이 예시를 그대로 베끼는 경향이 생긴다.
MAX_SAMPLES_PER_TOPIC = 2

# 이 길이 미만은 bigram 유사도가 불안정해서 완전 일치로만 중복 판정한다.
MIN_LEN_FOR_FUZZY_MATCH = 15

# 같은 주제에 여러 슬롯이 배정될 때 슬롯마다 강제할 출제 관점.
#
# avoid 목록("이미 낸 문제와 겹치지 마라")만으로는 중복이 계속 나왔다.
# 금지 지시는 LLM이 쉽게 무시하지만, 관점을 지정하면("이번엔 계산 방법을 물어라")
# 확실히 따른다. 슬롯 인덱스로 순환하므로 결정론적이다.
# 추상적인 명사구("다른 지표와 함께 봐야 하는 이유")로 뒀더니 LLM이 무시하고
# 계속 "흔한 오해" 유형만 만들었다. 명령문 + 필수 등장 요소를 박아야 따른다.
REVIEW_ANGLES = [
    "이 개념의 정의나 계산식 자체를 정확히 알고 있는지 묻는 문제를 만드세요. "
    "'무엇으로 나눈 값인가', '어떻게 계산하는가' 같은 형태가 좋습니다.",

    "이 개념을 해석할 때 흔히 저지르는 오해를 바로잡는 문제를 만드세요.",

    "이 개념 하나만 보고 판단하면 안 되는 이유를 다루는 문제를 만드세요. "
    "같은 업종의 평균과 비교해야 한다는 점, 또는 다른 지표(ROE, 부채비율, "
    "성장률 등)를 함께 봐야 한다는 점이 문제나 선택지에 자연스럽게 드러나게 하세요. "
    "억지스러운 가정이나 인위적인 수치 설정은 넣지 마세요.",

    "실제 투자 상황(매수할지, 보유할지, 매도할지)에서 이 개념을 적용해 "
    "판단하는 문제를 만드세요. 구체적인 상황 설정을 넣으세요.",

    "이 개념을 쓸 수 없거나 수치가 왜곡되는 예외 상황을 다루는 문제를 만드세요. "
    "적자 기업, 일회성 이익, 자본잠식 같은 구체적인 예외를 지문에 넣으세요.",
]


# 프롬프트 공통 규칙.
# 실측에서 나온 결함을 그대로 조건으로 박아둔 것이라 함부로 줄이지 말 것:
#  - 지문 전제와 정답이 모순 ("이익이 일정하다"고 해놓고 정답이 "이익이 증가")
#  - 원본이 가르치려던 결론을 뒤집음 ("PER 낮다고 저평가 아니다" -> "저평가로 봅니다")
#  - 해설에 "학생이 고른" 같은 표현이 남아 문제은행 텍스트로 부적합
_COMMON_RULES = """- 주제: {topic}
  반드시 이 개념을 유지하세요. 다른 개념으로 바꾸거나 여러 개념을 조합하지 마세요.
- 난이도: {level} ({level_guide})
- 위 틀린 문제를 그대로 베끼지 마세요. 상황, 수치, 문장 구조를 모두 바꾸세요.
- 문제 지문의 전제와 정답이 서로 모순되면 절대 안 됩니다.
  (나쁜 예: "이익이 일정하게 유지된다고 가정할 때" 라고 전제해놓고
   정답이 "이익이 증가했기 때문"인 문제 — 이런 문제는 만들지 마세요)
- 원본 문제가 가르치려던 결론을 뒤집지 마세요.
  원본이 "A라고 단정할 수 없다"를 가르쳤다면, 새 문제도 "A라고 단정할 수 없다" 쪽이어야 합니다.
- 해설에 "학생이", "학생은", "당신이 고른", "오답으로 선택한" 같은 표현을 절대 쓰지 마세요.
  이 문제는 나중에 다시 출제되므로, 문제와 해설만 읽어도 완결되는 일반적인 서술이어야 합니다.
  틀리기 쉬운 지점은 "~라고 생각하기 쉽지만" 같은 표현으로 자연스럽게 녹이세요.
- 문제는 반드시 한 문장으로 짧고 명확하게 작성하세요.
- 사용자 수준에 맞는 쉬운 용어만 사용하세요.
- 모든 텍스트는 자연스러운 한국어 문장으로 작성하고, 비문이나 오타가 없게 하세요.
- 실제로 존재하는 개념·지표·용어만 사용하고, 존재하지 않는 용어를 지어내지 마세요.
  특히 지표의 정의를 정확히 쓰세요.
  (PER = 주가 / 주당순이익(EPS). "순이익률"이 아닙니다.
   배당수익률 = 연간 배당금 / 주가. 분모는 주가입니다.)
- 반드시 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요."""


# 출제 관점 블록은 프롬프트 '맨 뒤'(JSON 형식 직전)에 둔다.
# 중간에 두었더니 앞의 오답 컨텍스트에 밀려 무시당했다(실측).
_ANGLE_BLOCK = """
=========================================
가장 중요한 지시 — 이번 문제의 출제 방향
=========================================
{angle}

- 위 지시를 반드시 따르세요. 이 방향에서 벗어난 문제는 만들지 마세요.
- 같은 주제의 다른 방향 문제들은 이미 출제되었습니다.
  위에 적힌 방향 하나만 다루세요.
"""


REVIEW_MC_PROMPT = ChatPromptTemplate.from_template("""
당신은 주식 투자 교육 전문가입니다.
아래는 학생이 실제로 틀린 문제입니다.
같은 개념을 다시 점검할 수 있는 새로운 4지선다 객관식 문제를 1개 생성하세요.

[학생이 틀린 문제]
{wrong_context}

[이번 세트에서 이미 출제된 문제]
{avoid_block}

조건:
""" + _COMMON_RULES + """
- 위 "이미 출제된 문제"와 같은 내용을 묻지 마세요.
  같은 주제라도 반드시 다른 측면(정의, 계산, 한계, 해석 오류, 실제 적용 등)을 물어야 합니다.
  단순히 문장만 바꿔서 같은 것을 묻는 것은 금지입니다.
- 학생이 고른 오답에 담긴 오해를 오답 선택지 중 하나로 배치하세요.
- 복잡한 수치나 여러 개념을 동시에 묻는 문제는 절대 만들지 마세요.
- **오답 선택지 3개는 모두 "사실이 아닌 내용"이어야 합니다.**
  그럴듯해 보이지만 실제로는 맞는 말을 오답 자리에 넣으면 안 됩니다.
  (나쁜 예: 정답이 "PER은 순이익에도 영향을 받는다"인데
   오답 자리에 "주가 하락이 반드시 실적 악화를 뜻하지는 않는다"를 넣는 경우
   — 이 오답도 사실이라 정답이 두 개가 되어버립니다)
  각 오답을 쓴 뒤 "이 문장은 틀린 말인가?"를 스스로 확인하세요.
- 문제가 묻는 방향과 해설이 설명하는 방향이 반드시 일치해야 합니다.
  ("PER이 낮아진 이유"를 물었으면 해설도 낮아지는 경우를 설명해야 하며,
   높아지는 경우를 설명하면 안 됩니다)
- 정답 번호(correct_no)와 해설이 서로 모순되지 않아야 합니다.
""" + _ANGLE_BLOCK + """
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


REVIEW_OX_PROMPT = ChatPromptTemplate.from_template("""
당신은 주식 투자 교육 전문가입니다.
아래는 학생이 실제로 틀린 문제입니다.
같은 개념을 다시 점검할 수 있는 새로운 OX 문제를 1개 생성하세요.

[학생이 틀린 문제]
{wrong_context}

[이번 세트에서 이미 출제된 문제]
{avoid_block}

조건:
""" + _COMMON_RULES + """
- 위 "이미 출제된 문제"와 같은 내용을 묻지 마세요. 반드시 다른 측면을 물어야 합니다.
- question_text는 반드시 참/거짓을 판단할 수 있는 **평서문**이어야 합니다.
  "~인가?", "~일까요?", "~무엇입니까?" 같은 의문문은 절대 쓰지 마세요.
  (좋은 예: "배당수익률이 높으면 항상 우량한 기업이다.")
- **문장은 반드시 긍정문으로 서술하세요.** 부정 표현이 두 번 들어간 문장은 절대 금지입니다.
  (나쁜 예: "배당수익률이 자동으로 높아지는 것이 아니다" — 이런 이중부정은
   O/X 판단이 헷갈려 정답을 틀리게 만듭니다.
   좋은 예: "주가가 오르면 배당수익률은 낮아진다")
- answer를 정하기 전에, 작성한 문장이 사실이면 "O", 사실이 아니면 "X"인지
  한 번 더 확인하세요. 해설의 내용과 answer가 반드시 일치해야 합니다.
""" + _ANGLE_BLOCK + """
JSON 형식:
{{
  "question_text": "문제 내용 (평서문, 물음표로 끝나면 안 됨)",
  "answer": "O" 또는 "X",
  "explanation": "해설 내용 (2~3문장)",
  "topic": "{topic}"
}}
""")


VALIDATE_PROMPT = ChatPromptTemplate.from_template("""
당신은 주식 투자 교육 문제를 검수하는 전문가입니다.
아래 문제들 각각에 대해, 표시된 정답이 실제로 맞는지 검사하세요.

[검사할 문제들]
{questions_block}

검사 항목 (각 문제마다 아래를 순서대로 확인하세요):

1. 표시된 정답이 사실에 비추어 실제로 맞는가?

2. **오답으로 표시된 선택지 중에 "사실인 문장"이 섞여 있지 않은가?**
   오답 자리에 맞는 말이 들어가 있으면 정답이 두 개가 되므로 불합격입니다.
   오답 선택지를 하나씩 읽으면서 "이 문장은 틀린 말인가?"를 확인하세요.

3. **문제가 묻는 방향과 해설이 설명하는 방향이 일치하는가?**
   문제는 "PER이 낮아진 이유"를 묻는데 해설이 "PER이 상승한다"를 설명하면
   방향이 반대이므로 불합격입니다.

4. 해설의 내용이 표시된 정답과 일치하는가?
   (해설은 A라고 설명하는데 정답은 B로 되어 있으면 불합격)

5. 문제 지문의 전제와 정답이 서로 모순되지 않는가?

6. 지표의 정의가 정확한가?
   (PER = 주가 / 주당순이익(EPS), 배당수익률 = 연간 배당금 / 주가)

7. OX 문제라면, 이중부정 등으로 참/거짓 판단이 헷갈리게 되어 있지 않은가?

8. 문제 문장이 비문이거나 무엇을 묻는지 불분명하지 않은가?

판정 기준:
- 위 항목에 하나라도 걸리면 "ok": false
- 사실 관계가 애매한 경우에만 통과시키세요.
  2번(오답에 사실이 섞임)과 3번(문제와 해설의 방향 불일치)은
  특히 놓치기 쉬우니 반드시 꼼꼼히 확인하세요.
- reason은 불합격일 때만 한 문장으로 간단히 작성하세요.

반드시 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.

JSON 형식:
{{
  "results": [
    {{"no": 1, "ok": true, "reason": ""}},
    {{"no": 2, "ok": false, "reason": "해설은 배당수익률이 낮아진다고 설명하는데 정답은 반대로 되어 있음"}}
  ]
}}
""")


def _format_questions_for_validation(questions: list[dict]) -> str:
    """검증 프롬프트에 넣을 문제 목록 텍스트."""
    blocks = []
    for i, q in enumerate(questions, 1):
        lines = [f"[{i}번] ({q['question_type']}) {q['question_text']}"]
        for c in q["choices"]:
            mark = "  <- 정답" if c["is_correct"] else ""
            lines.append(f"    {c['choice_no']}. {c['text']}{mark}")
        lines.append(f"    해설: {q['explanation']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


async def validate_questions(questions: list[dict]) -> dict[int, str]:
    """생성된 문제들의 정답/해설 일치를 한 번의 LLM 호출로 검사한다.

    반환: {문제 인덱스(0-based): 불합격 사유}. 통과한 문제는 포함되지 않는다.

    개별 검증은 호출 수가 문제 수만큼 늘어나 응답 시간이 두 배가 되므로
    배치로 처리한다(팀 결정). 검증 자체가 실패하면 빈 dict를 반환해
    생성 결과를 그대로 통과시킨다 — 검증은 부가 안전장치이지
    이것 때문에 기능 전체가 실패하면 안 된다.
    """
    if not questions:
        return {}

    chain = VALIDATE_PROMPT | _get_llm()
    try:
        response = await chain.ainvoke({
            "questions_block": _format_questions_for_validation(questions),
        })
        data = _extract_json(response.content)
    except Exception as e:
        logger.warning(f"문제 검증 실패 - 검증 없이 통과시킴: {e}")
        return {}

    failed: dict[int, str] = {}
    for item in data.get("results") or []:
        if not isinstance(item, dict) or item.get("ok", True):
            continue
        try:
            idx = int(item["no"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < len(questions):
            failed[idx] = (item.get("reason") or "정답/해설 불일치").strip()
    return failed


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


def format_avoid_block(texts: list[str]) -> str:
    """이미 생성된 문제 목록을 프롬프트에 넣을 텍스트로 변환한다.

    같은 주제에 슬롯이 여러 개 배정되면 매번 같은 오답 컨텍스트가 들어가서
    LLM이 사실상 동일한 문제를 반복 생성한다(실측 확인). 이미 만든 문제를
    보여주고 피하게 하는 것이 순차 생성에서 중복을 막는 가장 확실한 방법이다.
    """
    if not texts:
        return "(아직 없음)"
    return "\n".join(f"- {t}" for t in texts)


def _norm(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def is_near_duplicate(text: str, existing: list[str], threshold: float = 0.65) -> bool:
    """문자 bigram 자카드 유사도로 거의 같은 문장을 걸러낸다.

    프롬프트의 avoid 목록이 1차 방어선이고, 이건 그걸 뚫고 나온 표현만 다른
    복제를 잡는 백스톱이다. 의미는 같은데 표현이 완전히 다른 중복까지는
    잡지 못한다 (그건 임베딩 유사도가 필요한 영역).
    """
    a = _norm(text)
    if not a:
        return False

    # 짧은 문자열은 bigram 수가 적어 유사도가 요동친다
    # ("문제A" vs "문제B"가 0.67로 잡히는 식). 이 구간은 완전 일치만 중복으로 본다.
    if len(a) < MIN_LEN_FOR_FUZZY_MATCH:
        return any(_norm(o) == a for o in existing)

    a_grams = {a[i:i + 2] for i in range(len(a) - 1)}
    for other in existing:
        b = _norm(other)
        if not b:
            continue
        if len(b) < MIN_LEN_FOR_FUZZY_MATCH:
            if b == a:
                return True
            continue
        b_grams = {b[i:i + 2] for i in range(len(b) - 1)}
        union = a_grams | b_grams
        if union and len(a_grams & b_grams) / len(union) >= threshold:
            return True
    return False


# =========================================================
# 문제 생성
# =========================================================

# 생성된 문제는 문제은행에 저장되어 나중에 다시 출제된다.
# 해설이 "지금 이 학생"을 지칭하면 재출제 시 문맥이 깨지므로 걸러낸다.
_EXPLANATION_BANNED = ("학생이", "학생은", "학생께", "당신이 고른", "오답으로 선택한", "선택하신")

# 소문자 영단어 3글자 이상 = 한국어 문장에 영어가 새어 들어온 것.
# ("실적 alone 이 좋아도 valuation 의 숫자는..." 같은 실측 사례)
# PER / EPS / ROE / ETF 같은 정당한 대문자 약어는 걸리지 않는다.
_LATIN_LEAK = re.compile(r"[a-z]{3,}")


def _assert_korean(text: str, where: str) -> None:
    leak = _LATIN_LEAK.search(text)
    assert leak is None, f"{where}에 영어 단어 혼입: {leak.group()}"


def _assert_clean_explanation(explanation: str) -> None:
    for word in _EXPLANATION_BANNED:
        assert word not in explanation, f"해설에 학습자 지칭 표현 포함: {word}"
    _assert_korean(explanation, "해설")


async def _generate_review_mc(
    user: UserContext, topic: str, wrong_context: str, avoid: list[str], angle: str,
) -> dict:
    level_guide = LEVEL_GUIDE.get(user.investment_level, LEVEL_GUIDE["미설정"])
    chain = REVIEW_MC_PROMPT | _get_llm()

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
            data = _extract_json(response.content)

            assert "question_text" in data
            assert "choices" in data and len(data["choices"]) == 4
            assert "correct_no" in data and 1 <= data["correct_no"] <= 4
            assert "explanation" in data
            nos = sorted(c["no"] for c in data["choices"])
            assert nos == [1, 2, 3, 4], f"선택지 번호 이상: {nos}"
            _assert_korean(data["question_text"], "문제 지문")
            for c in data["choices"]:
                _assert_korean(c["text"], "선택지")
            _assert_clean_explanation(data["explanation"])

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
) -> dict:
    level_guide = LEVEL_GUIDE.get(user.investment_level, LEVEL_GUIDE["미설정"])
    chain = REVIEW_OX_PROMPT | _get_llm()

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
            data = _extract_json(response.content)

            assert "question_text" in data
            assert "answer" in data and data["answer"] in ["O", "X"]
            assert "explanation" in data
            # OX는 참/거짓을 판단할 평서문이어야 한다.
            # 의문문이면 O/X로 답할 대상이 아니므로 재시도한다 (실측에서 발생).
            assert not data["question_text"].strip().endswith("?"), "OX 문제가 의문문"
            _assert_korean(data["question_text"], "문제 지문")
            _assert_clean_explanation(data["explanation"])

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

    questions: list[dict] = []
    gen_args: list[tuple] = []
    all_texts: list[str] = []
    # 주제별 슬롯 소진 횟수 - 오답 샘플을 슬롯마다 돌려쓰기 위한 인덱스
    slot_index: dict[str, int] = defaultdict(int)
    # 주제별 이미 생성된 문제 - 프롬프트 avoid 목록으로 전달
    generated_by_topic: dict[str, list[str]] = defaultdict(list)

    for group in slots:
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
        angle = REVIEW_ANGLES[slot_index[key] % len(REVIEW_ANGLES)]
        slot_index[key] += 1

        generate = (
            _generate_review_ox
            if group["question_type"] == "OX"
            else _generate_review_mc
        )
        avoid = generated_by_topic[key]

        question = await generate(user, group["topic"], wrong_context, avoid, angle)
        # avoid 목록을 뚫고 나온 표현만 다른 복제는 1회 재생성으로 걸러낸다.
        # 그래도 겹치면 그대로 채택한다 (문제 개수는 항상 맞춰야 하므로).
        if is_near_duplicate(question["question_text"], all_texts):
            logger.info(f"중복 문제 감지 - 재생성: {question['question_text'][:40]}")
            question = await generate(user, group["topic"], wrong_context, avoid, angle)

        all_texts.append(question["question_text"])
        generated_by_topic[key].append(question["question_text"])

        mapping = categories.get(key) or {}
        detail_codes = mapping.get("detail_codes") or []
        question["category_code"] = mapping.get("category_code")
        question["detail_codes"] = detail_codes
        question["primary_detail_code"] = detail_codes[0] if detail_codes else None
        questions.append(question)
        # 검증에서 불합격 시 같은 조건으로 재생성하기 위해 생성 인자를 보관
        gen_args.append((generate, group["topic"], wrong_context, avoid, angle))

    await _revalidate_and_fix(questions, gen_args, user)
    return questions


async def _revalidate_and_fix(
    questions: list[dict], gen_args: list[tuple], user: UserContext,
) -> None:
    """정답/해설이 모순된 문제를 찾아 같은 조건으로 1회 재생성한다 (in-place 수정).

    재생성본은 다시 검증하지 않는다. 검증 호출이 계속 늘어나는 것을 막기 위한
    타협이며, 재생성으로도 안 고쳐지면 원본보다 나빠질 이유는 없으므로 채택한다.
    """
    failed = await validate_questions(questions)
    if not failed:
        logger.info("문제 검증 통과 (전체 합격)")
        return

    logger.warning(f"검증 불합격 {len(failed)}건 - 재생성: {list(failed.values())}")
    for idx, reason in failed.items():
        generate, topic, wrong_context, avoid, angle = gen_args[idx]
        try:
            fixed = await generate(user, topic, wrong_context, avoid, angle)
        except ValueError as e:
            # 재생성 실패 시 원본 유지 - 문제 개수는 항상 맞춰야 한다
            logger.warning(f"{idx + 1}번 재생성 실패 - 원본 유지: {e}")
            continue
        # 카테고리 매핑은 주제 기준이라 원본 것을 그대로 승계한다
        for field in ("category_code", "detail_codes", "primary_detail_code"):
            fixed[field] = questions[idx][field]
        questions[idx] = fixed
