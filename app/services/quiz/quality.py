"""퀴즈 문제 품질 공통 모듈.

`/quiz/generate`, `/quiz/generate/prompt`, `/quiz/generate/review` 세 경로가
같은 LLM으로 문제를 만들므로 품질 문제도 동일하게 발생한다.
오답 기반 생성에서 실측으로 잡아낸 결함과 그 방어 장치를 여기 모아
모든 생성 경로가 함께 쓰도록 한다.

실측으로 확인된 결함들 (Qwen3.5:9b, temperature 0.7):
- 해설에 영어 혼입: "실적 alone 이 좋아도 valuation 의 숫자는..."
- OX 이중부정으로 정답이 뒤집힘: "자동으로 높아지는 것이 아니다" -> 정답 X (실제로는 O)
- 오답 자리에 사실인 문장: 정답이 두 개가 되어버림
- 문제와 해설의 방향 불일치: "PER이 낮아진 이유"를 묻는데 해설은 "PER이 상승한다"
- 지표 정의 오류: "주가 대비 순이익률인 PER" (PER은 순이익률이 아님)
- 문제 지문의 전제와 정답이 모순

여기 있는 검증은 두 층으로 나뉜다.
1) 결정론적 검사 (assert_*) — 생성 직후 즉시, LLM 호출 없음
2) LLM 배치 검증 (validate_questions) — 세트 완성 후 1회 호출

LLM 헬퍼(_get_llm/extract_json)도 여기 둔다. generator와 review가 모두
이 모듈을 참조하므로, generator에 두면 순환 import가 된다.
"""
import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger(__name__)


# =========================================================
# LLM 헬퍼
# =========================================================

def get_llm() -> ChatOllama:
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


def extract_json(text: str) -> dict:
    """LLM 응답에서 JSON 추출.

    코드펜스만 제거하던 방식은 LLM이 JSON 앞뒤에 설명 문장을 붙이면 곧바로 실패해
    재시도를 소진하고 500이 나기 쉬웠다. 먼저 통짜 파싱을 시도하고, 실패하면
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


# =========================================================
# 프롬프트 조각 (모든 생성 프롬프트에 삽입)
# =========================================================

# 문제 유형과 무관한 공통 품질 규칙.
# 각 항목은 실측 결함에 대응하므로 함부로 줄이지 말 것.
QUALITY_RULES = """- 문제는 반드시 한 문장으로 짧고 명확하게 작성하세요.
- 사용자 수준에 맞는 쉬운 용어만 사용하세요.
- 모든 텍스트는 자연스러운 한국어로 작성하세요.
  영어 단어를 그대로 섞어 쓰지 마세요 (PER, EPS, ROE 같은 지표 약어는 허용).
  (나쁜 예: "실적 alone 이 좋아도 valuation 의 숫자는 낮아집니다")
- 문제 지문의 전제와 정답이 서로 모순되면 절대 안 됩니다.
  (나쁜 예: "이익이 일정하게 유지된다고 가정할 때"라고 전제해놓고
   정답이 "이익이 증가했기 때문"인 문제)
- 문제가 묻는 방향과 해설이 설명하는 방향이 반드시 일치해야 합니다.
  ("PER이 낮아진 이유"를 물었으면 해설도 낮아지는 경우를 설명해야 하며,
   높아지는 경우를 설명하면 안 됩니다)
- 실제로 존재하는 개념·지표·용어만 사용하고, 존재하지 않는 용어를 지어내지 마세요.
  특히 지표의 정의를 정확히 쓰세요.
  (PER = 주가 / 주당순이익(EPS). "순이익률"이 아닙니다.
   배당수익률 = 연간 배당금 / 주가. 분모는 주가입니다.)
- 반드시 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요."""


# 객관식 전용 규칙
MC_QUALITY_RULES = """- 복잡한 수치나 여러 개념을 동시에 묻는 문제는 절대 만들지 마세요.
- **오답 선택지 3개는 모두 "사실이 아닌 내용"이어야 합니다.**
  그럴듯해 보이지만 실제로는 맞는 말을 오답 자리에 넣으면 안 됩니다.
  (나쁜 예: 정답이 "PER은 순이익에도 영향을 받는다"인데
   오답 자리에 "주가 하락이 반드시 실적 악화를 뜻하지는 않는다"를 넣는 경우
   — 이 오답도 사실이라 정답이 두 개가 되어버립니다)
  각 오답을 쓴 뒤 "이 문장은 틀린 말인가?"를 스스로 확인하세요.
- **오답 선택지의 결론이 정답과 같으면 안 됩니다.**
  근거만 다르고 결론이 같으면 정답이 두 개가 됩니다.
  (나쁜 예: 정답이 "PER 값이 커진다"인데 오답이 "주가가 좋아서 숫자가 커집니다"
   — 근거는 틀렸지만 결론이 같아 둘 다 정답으로 읽힙니다)
- 정답 번호(correct_no)와 해설이 서로 모순되지 않아야 합니다."""


# OX 전용 규칙
OX_QUALITY_RULES = """- question_text는 반드시 참/거짓을 판단할 수 있는 **평서문**이어야 합니다.
  "~인가?", "~일까요?", "~무엇입니까?" 같은 의문문은 절대 쓰지 마세요.
  (좋은 예: "배당수익률이 높으면 항상 우량한 기업이다.")
- **문장은 반드시 긍정문으로 서술하세요.** 부정 표현이 두 번 들어간 문장은 절대 금지입니다.
  (나쁜 예: "배당수익률이 자동으로 높아지는 것이 아니다" — 이런 이중부정은
   O/X 판단이 헷갈려 정답을 틀리게 만듭니다.
   좋은 예: "주가가 오르면 배당수익률은 낮아진다")
- answer를 정하기 전에, 작성한 문장이 사실이면 "O", 사실이 아니면 "X"인지
  한 번 더 확인하세요. 해설의 내용과 answer가 반드시 일치해야 합니다."""


# =========================================================
# 출제 관점 로테이션
# =========================================================

# 같은 주제로 여러 문제를 만들 때 문제마다 강제할 출제 방향.
#
# "이미 낸 문제와 겹치지 마라"는 금지 지시만으로는 중복이 계속 나왔다(실측).
# LLM은 금지는 쉽게 무시하지만 "이번엔 계산식을 물어라" 같은 지시는 따른다.
# 추상적인 명사구로 뒀을 때도 무시당해서, 명령문 + 필수 등장 요소로 적는다.
# 슬롯 인덱스로 순환하므로 결정론적이다.
GENERATION_ANGLES = [
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


# 관점 블록은 프롬프트 '맨 뒤'(JSON 형식 직전)에 둔다.
# 중간에 두었더니 앞의 컨텍스트에 밀려 무시당했다(실측).
ANGLE_BLOCK = """
=========================================
가장 중요한 지시 — 이번 문제의 출제 방향
=========================================
{angle}

- 위 지시를 반드시 따르세요. 이 방향에서 벗어난 문제는 만들지 마세요.
- 같은 주제의 다른 방향 문제들은 이미 출제되었습니다.
  위에 적힌 방향 하나만 다루세요.

[이번 세트에서 이미 출제된 문제]
{avoid_block}
- 위 문제들과 같은 내용을 묻지 마세요.
  표현만 바꿔서 같은 것을 묻는 것도 중복입니다.
"""


def angle_for(slot_index: int) -> str:
    """주제별 n번째 문제에 적용할 출제 관점."""
    return GENERATION_ANGLES[slot_index % len(GENERATION_ANGLES)]


def format_avoid_block(texts: list[str]) -> str:
    """이미 생성된 문제 목록을 프롬프트에 넣을 텍스트로 변환한다."""
    if not texts:
        return "(아직 없음)"
    return "\n".join(f"- {t}" for t in texts)


# =========================================================
# 결정론적 검사 (생성 직후, LLM 호출 없음)
# =========================================================

# 소문자 영단어 3글자 이상 = 한국어 문장에 영어가 새어 들어온 것.
# PER / EPS / ROE / ETF 같은 정당한 대문자 약어는 걸리지 않는다.
_LATIN_LEAK = re.compile(r"[a-z]{3,}")

# 한자·일본어 가나 혼입. Qwen 계열에서 자주 나온다
# (실측: "주가 상승률이 항상 높은 公司股票").
# 한국 주식 교육 문맥에서 한자·가나가 정당하게 쓰일 일은 없으므로 전부 거부한다.
_CJK_LEAK = re.compile(r"[一-鿿぀-ヿ]")

# 문제은행에 저장되어 나중에 다시 출제되므로,
# 해설이 "지금 이 학생"을 지칭하면 재출제 시 문맥이 깨진다.
_LEARNER_REFERENCES = (
    "학생이", "학생은", "학생께", "당신이 고른", "오답으로 선택한", "선택하신",
)


def assert_korean(text: str, where: str = "텍스트") -> None:
    """한국어 문장에 영어·한자·가나가 섞였으면 AssertionError."""
    leak = _LATIN_LEAK.search(text)
    assert leak is None, f"{where}에 영어 단어 혼입: {leak.group()}"
    cjk = _CJK_LEAK.search(text)
    assert cjk is None, f"{where}에 한자/가나 혼입: {cjk.group()}"


def assert_no_learner_reference(explanation: str) -> None:
    """해설이 특정 학습자를 지칭하면 AssertionError."""
    for word in _LEARNER_REFERENCES:
        assert word not in explanation, f"해설에 학습자 지칭 표현 포함: {word}"


def assert_ox_declarative(question_text: str) -> None:
    """OX 문제가 의문문이면 AssertionError (O/X로 답할 대상이 아니므로)."""
    assert not question_text.strip().endswith("?"), "OX 문제가 의문문"


def check_mc_question(data: dict, learner_reference_check: bool = False) -> None:
    """객관식 생성 결과의 결정론적 검사. 실패 시 AssertionError -> 재시도."""
    assert "question_text" in data
    assert "choices" in data and len(data["choices"]) == 4
    assert "correct_no" in data and 1 <= data["correct_no"] <= 4
    assert "explanation" in data
    # 선택지 번호가 1~4로 중복 없이 존재해야 is_correct 매핑이 깨지지 않는다
    nos = sorted(c["no"] for c in data["choices"])
    assert nos == [1, 2, 3, 4], f"선택지 번호 이상: {nos}"

    assert_korean(data["question_text"], "문제 지문")
    for c in data["choices"]:
        assert_korean(c["text"], "선택지")
    assert_korean(data["explanation"], "해설")
    if learner_reference_check:
        assert_no_learner_reference(data["explanation"])


def check_ox_question(data: dict, learner_reference_check: bool = False) -> None:
    """OX 생성 결과의 결정론적 검사. 실패 시 AssertionError -> 재시도."""
    assert "question_text" in data
    assert "answer" in data and data["answer"] in ["O", "X"]
    assert "explanation" in data

    assert_ox_declarative(data["question_text"])
    assert_korean(data["question_text"], "문제 지문")
    assert_korean(data["explanation"], "해설")
    if learner_reference_check:
        assert_no_learner_reference(data["explanation"])


# =========================================================
# 근사 중복 판정
# =========================================================

# 이 길이 미만은 bigram 유사도가 불안정해서 완전 일치로만 중복 판정한다.
MIN_LEN_FOR_FUZZY_MATCH = 15


def _norm(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def is_near_duplicate(text: str, existing: list[str], threshold: float = 0.65) -> bool:
    """문자 bigram 자카드 유사도로 거의 같은 문장을 걸러낸다.

    표현만 다른 복제를 잡는 백스톱이다. 의미는 같은데 표현이 완전히 다른
    중복까지는 잡지 못한다 (그건 임베딩 유사도가 필요한 영역).
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
# LLM 배치 검증 (세트 완성 후 1회 호출)
# =========================================================

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
- 위 항목에 **명백히** 걸릴 때만 "ok": false 로 하세요.
  2번(오답에 사실이 섞임)과 3번(문제와 해설의 방향 불일치)은
  놓치기 쉬우니 특히 주의해서 보세요.
- **조금이라도 판단이 서지 않으면 "ok": true 로 통과시키세요.**
  멀쩡한 문제를 불합격 처리하면 불필요한 재생성이 발생합니다.
- reason은 불합격일 때만, **30자 이내 한 문장**으로 쓰세요.
  길게 설명하지 말고 무엇이 문제인지만 짧게 적으세요.
  판정을 번복하거나 "하지만", "다시 보니" 같은 표현을 쓰지 마세요.
  (나쁜 예: "...논란 여지가 있으나 다시 읽어보니 사실 모든 것이 맞다")
  (좋은 예: "오답 3번이 사실이라 정답이 두 개")

반드시 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.

JSON 형식:
{{
  "results": [
    {{"no": 1, "ok": true, "reason": ""}},
    {{"no": 2, "ok": false, "reason": "해설은 배당수익률이 낮아진다고 설명하는데 정답은 반대로 되어 있음"}}
  ]
}}
""")


def format_questions_for_validation(questions: list[dict]) -> str:
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

    chain = VALIDATE_PROMPT | get_llm()
    try:
        response = await chain.ainvoke({
            "questions_block": format_questions_for_validation(questions),
        })
        data = extract_json(response.content)
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
        if not (0 <= idx < len(questions)):
            continue

        reason = (item.get("reason") or "정답/해설 불일치").strip()
        if _reason_retracts(reason):
            logger.info(f"{idx + 1}번 불합격 사유가 스스로 번복 - 통과 처리: {reason[:50]}")
            continue
        failed[idx] = reason
    return failed


# 검증 LLM이 장황하게 쓰다가 "사실 문제 없다"로 결론을 뒤집는 경우가 있다(실측).
# ok=false 인데 사유가 자기 번복이면 그 판정은 신뢰할 수 없으므로 통과시킨다
# (멀쩡한 문제를 재생성하면 응답 시간만 늘어난다).
_RETRACTION_MARKERS = (
    "문제가 아님", "문제 없", "오류가 아님", "모든 것이 맞다", "모두 맞다",
    "사실 맞", "이상 없", "다시 읽어보니", "다시 보니",
)


def _reason_retracts(reason: str) -> bool:
    return any(marker in reason for marker in _RETRACTION_MARKERS)
