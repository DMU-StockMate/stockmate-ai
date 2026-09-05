"""오답 기반 문제 생성(app/services/quiz/review.py) 회귀 테스트.

LLM을 호출하지 않는다. langchain 체인을 스텁으로 갈아끼워서
"프롬프트에 무엇이 들어갔는지"와 "파이프라인이 어떻게 동작하는지"만 검증한다.
Ollama 없이도 돌아가므로 CI나 로컬에서 바로 실행 가능하다.

실행:
    uv run python scripts/test_quiz_review.py
"""
import asyncio
import json
import sys
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# =========================================================
# 스텁 (import 전에 설치해야 한다)
# =========================================================

CALLS: list[dict] = []       # 생성 호출에 들어간 프롬프트 인자
RESPONSES: list[str] = []    # 미리 넣어둔 LLM 응답 (순서대로 소비)
VALIDATE_RESULT: dict = {"results": []}
BANK_SIMILARITY = None   # 퀴즈 뱅크가 돌려줄 유사도 (None이면 결과 없음)
BANK_UPSERTS: list = []  # 뱅크에 저장된 호출 기록
BANK_TOPIC_COUNT = 0     # 관점 오프셋용 - 주제별 기존 문제 수


class _Resp:
    def __init__(self, content):
        self.content = content


# correct=None 이면 스텁이 "프롬프트가 요구한 정답 위치"를 따른다.
# 정답이 1번에 몰리는 문제를 막으려고 문제마다 위치를 지정하게 됐고,
# 대부분의 테스트는 위치가 관심사가 아니므로 기본을 순응으로 둔다.
def _mc(question, explanation="정상적인 해설 문장입니다.", correct=None):
    return json.dumps({
        "question_text": question,
        "choices": [{"no": i, "text": f"보기{i}"} for i in range(1, 5)],
        "correct_no": correct,
        "explanation": explanation,
        "topic": "PER",
    }, ensure_ascii=False)


def _ox(question, explanation="정상적인 해설 문장입니다.", answer="X"):
    return json.dumps({
        "question_text": question,
        "answer": answer,
        "explanation": explanation,
        "topic": "배당",
    }, ensure_ascii=False)


_FALLBACK_POOL = [
    "기업의 주당순이익을 구하는 올바른 계산식은 무엇인가",
    "업종마다 적정 밸류에이션 수준이 다른 배경으로 옳은 설명은",
    "적자 상태인 회사에서 이 지표를 활용하기 어려운 까닭은",
    "일회성 처분이익이 실적에 반영되었을 때 나타나는 현상은",
    "성장률이 높은 기업에서 관찰되는 특징으로 알맞은 것은",
    "재무제표의 어느 항목을 함께 확인해야 하는지 고르시오",
    "투자 판단 시 동종업계 비교가 필요한 이유로 옳은 것은",
]


def _obey_answer_pos(raw: str, prompt_vars: dict) -> str:
    """correct_no 가 null 인 응답을 지시받은 자리로 채운다 (지시를 따른 모델 흉내)."""
    if '"correct_no": null' not in raw:
        return raw
    return raw.replace('"correct_no": null',
                       f'"correct_no": {prompt_vars.get("answer_pos", 1)}')


def _install_stubs():
    lc_core = types.ModuleType("langchain_core")
    lc_prompts = types.ModuleType("langchain_core.prompts")

    class _Template:
        @classmethod
        def from_template(cls, template):
            obj = cls()
            obj.template = template
            # 검증 프롬프트와 생성 프롬프트를 구분한다
            obj.is_validator = "검수하는 전문가" in template
            return obj

        def __or__(self, other):
            return self

        async def ainvoke(self, kwargs):
            if self.is_validator:
                return _Resp(json.dumps(VALIDATE_RESULT, ensure_ascii=False))
            CALLS.append(kwargs)
            if RESPONSES:
                return _Resp(_obey_answer_pos(RESPONSES.pop(0), kwargs))
            return _Resp(_obey_answer_pos(
                _mc(_FALLBACK_POOL[(len(CALLS) - 1) % len(_FALLBACK_POOL)]), kwargs))

    lc_prompts.ChatPromptTemplate = _Template
    lc_core.prompts = lc_prompts
    sys.modules["langchain_core"] = lc_core
    sys.modules["langchain_core.prompts"] = lc_prompts

    lc_ollama = types.ModuleType("langchain_ollama")

    class _ChatOllama:
        def __init__(self, **kwargs):
            pass

    lc_ollama.ChatOllama = _ChatOllama
    sys.modules["langchain_ollama"] = lc_ollama

    # .env 없이도 import되게 settings를 대체한다
    config = types.ModuleType("app.core.config")

    class _Settings:
        # build_llm 이 LLM_BACKEND 를 보므로 스텁에도 있어야 한다.
        # 테스트는 ChatOllama 스텁을 쓰므로 ollama 경로로 고정한다.
        LLM_BACKEND = "ollama"
        LLM_BASE_URL = "http://127.0.0.1:8080/v1"
        LLM_API_KEY = "test"
        LLM_DISABLE_THINKING = True
        OLLAMA_BASE_URL = "http://localhost:11434"
        LLM_MODEL = "qwen3.5:9b"
        QUIZ_BANK_COLLECTION = "quiz_bank_test"
        QUIZ_BANK_ENABLED = True
        QUIZ_DUP_THRESHOLD = 0.90
        # 문제 생성이 순차 -> 병렬로 바뀌며 추가된 설정 (concurrency.py).
        QUIZ_GEN_CONCURRENCY = 5
        # 과거 문제와의 대조. 운영 기본값과 같게 꺼둔다.
        QUIZ_DUP_CHECK_PAST = False


    # Qdrant / 임베딩 스텁 - 퀴즈 뱅크가 실제 서버를 찾지 않게 한다
    vs = types.ModuleType("app.services.rag.vectorstore")
    class _Emb:
        def embed_query(self, t): return [0.0] * 1024
        def embed_documents(self, ts): return [[0.0] * 1024 for _ in ts]
    class _Hit:
        def __init__(self, score): self.score = score
    class _Client:
        def get_collections(self):
            return types.SimpleNamespace(collections=[])
        def create_collection(self, **kw): pass
        def create_payload_index(self, **kw): pass
        def query_points(self, **kw):
            points = [_Hit(BANK_SIMILARITY)] if BANK_SIMILARITY is not None else []
            return types.SimpleNamespace(points=points)
        def count(self, **kw): return types.SimpleNamespace(count=BANK_TOPIC_COUNT)
        def upsert(self, **kw): BANK_UPSERTS.append(kw)
    vs.get_embeddings = lambda: _Emb()
    vs.get_qdrant_client = lambda: _Client()
    sys.modules["app.services.rag.vectorstore"] = vs

    config.settings = _Settings()
    sys.modules["app.core.config"] = config


_install_stubs()

from app.schemas.chat import UserContext                     # noqa: E402
from app.schemas.quiz import WrongAnswerChoice, WrongAnswerItem  # noqa: E402
from app.services.quiz import quality as Q                    # noqa: E402
from app.services.quiz import review as R                     # noqa: E402


# =========================================================
# 헬퍼
# =========================================================

FAILURES: list[str] = []


def check(name: str, passed: bool, extra="") -> None:
    print(("PASS  " if passed else "FAIL  ") + name + (f"  {extra}" if extra else ""))
    if not passed:
        FAILURES.append(name)


def wrong(question, detail_code=None, topic=None, selected=1, qtype="MULTIPLE_CHOICE"):
    if qtype == "OX":
        choices = [WrongAnswerChoice(choice_no=1, text="O", is_correct=True),
                   WrongAnswerChoice(choice_no=2, text="X", is_correct=False)]
    else:
        choices = [WrongAnswerChoice(choice_no=i, text=f"보기{i}", is_correct=(i == 3))
                   for i in range(1, 5)]
    return WrongAnswerItem(
        question_text=question, question_type=qtype, detail_code=detail_code,
        topic=topic, explanation="원본 문제의 해설입니다.",
        selected_choice_no=selected, choices=choices,
    )


def reset(responses=None, validate=None):
    CALLS.clear()
    RESPONSES.clear()
    if responses:
        RESPONSES.extend(responses)
    global VALIDATE_RESULT
    VALIDATE_RESULT = validate if validate is not None else {"results": []}


USER = UserContext(user_id=1, investment_level="초급")


def run(wrongs):
    return asyncio.run(R.generate_quiz_from_wrong_answers(USER, wrongs))


# =========================================================
# 1. 그룹핑
# =========================================================

def test_grouping():
    print("\n--- 오답 그룹핑 ---")
    groups = R.group_wrong_answers([
        wrong("PER 문제1", "PER_BASIC"), wrong("PER 문제2", "PER_BASIC"),
        wrong("PER 문제3", "PER_BASIC"), wrong("RSI 문제1", "RSI_CONCEPT"),
        wrong("배당 문제1", "DIVIDEND"),
    ])
    check("주제 3개로 그룹핑", len(groups) == 3, [g["topic"] for g in groups])
    check("많이 틀린 순 정렬", groups[0]["wrong_count"] == 3)
    check("detail_code -> 한글 주제명", groups[0]["topic"] == "PER", groups[0]["topic"])
    check("샘플 상한 준수", len(groups[0]["samples"]) == R.MAX_SAMPLES_PER_TOPIC)

    check("topic 폴백",
          R.group_wrong_answers([wrong("q", topic="PER")])[0]["detail_code"] is None)
    check("미등록 detail_code -> topic 폴백",
          R.group_wrong_answers([wrong("q", "NOT_EXIST", topic="이상함")])[0]["topic"] == "이상함")
    check("주제 정보 없어도 크래시 없음",
          len(R.group_wrong_answers([wrong("주제 정보가 아예 없는 문제")])) == 1)

    ox = R.group_wrong_answers([wrong("a", "DIVIDEND", qtype="OX"),
                                wrong("b", "DIVIDEND", qtype="OX")])
    check("OX만 틀린 주제는 OX", ox[0]["question_type"] == "OX")
    mixed = R.group_wrong_answers([wrong("a", "DIVIDEND", qtype="OX"),
                                   wrong("b", "DIVIDEND"), wrong("c", "DIVIDEND")])
    check("유형 혼합이면 객관식", mixed[0]["question_type"] == "MULTIPLE_CHOICE")


# =========================================================
# 2. 슬롯 배분 (오답 수 비례)
# =========================================================

def test_allocation():
    print("\n--- 슬롯 배분 ---")

    def g(topic, n):
        return {"key": topic, "topic": topic, "detail_code": None,
                "wrong_count": n, "samples": [], "question_type": "MULTIPLE_CHOICE"}

    cases = [
        ("PER3/RSI1/배당1", [g("PER", 3), g("RSI", 1), g("배당", 1)], {"PER": 3, "RSI": 1, "배당": 1}),
        ("PER3/RSI2", [g("PER", 3), g("RSI", 2)], {"PER": 3, "RSI": 2}),
        ("단일 주제", [g("PER", 7)], {"PER": 5}),
        ("5주제 각1", [g(f"T{i}", 1) for i in range(5)], {f"T{i}": 1 for i in range(5)}),
    ]
    for name, groups, expected in cases:
        dist = dict(Counter(s["topic"] for s in R.allocate_slots(groups, 5)))
        check(f"배분 {name}", dist == expected, dist)

    check("주제 6개여도 5슬롯", len(R.allocate_slots([g(f"T{i}", 1) for i in range(6)], 5)) == 5)
    check("편중 시 상위 주제 독식",
          dict(Counter(s["topic"] for s in R.allocate_slots([g("PER", 20), g("RSI", 1)], 5))) == {"PER": 5})


# =========================================================
# 3. 프롬프트 컨텍스트
# =========================================================

def test_context():
    print("\n--- 프롬프트 컨텍스트 ---")
    ctx = R.format_wrong_context([wrong("PER 문제", "PER_BASIC", selected=1)])
    check("정답 표시", "정답" in ctx)
    check("학생 선택 표시", "학생이 고른 답" in ctx)
    check("해설 포함", "해설" in ctx)

    no_sel = R.format_wrong_context([wrong("PER 문제", "PER_BASIC", selected=None)])
    check("selected 없어도 동작", "학생이 고른 답" not in no_sel and "정답" in no_sel)

    check("avoid 빈 목록", R.format_avoid_block([]) == "(아직 없음)")
    check("avoid 항목 표기", R.format_avoid_block(["문제A", "문제B"]).count("- ") == 2)

    check("역인덱스 69개", len(R._DETAIL_INDEX) == 69, len(R._DETAIL_INDEX))
    check("PER_BASIC -> BEGINNER_FINANCIAL_BASIC",
          R._DETAIL_INDEX["PER_BASIC"][0] == "BEGINNER_FINANCIAL_BASIC")


# =========================================================
# 4. 근사 중복 판정
# =========================================================

def test_dedup():
    print("\n--- 근사 중복 판정 ---")
    check("동일 문장", Q.is_near_duplicate("PER이 낮으면 저평가라고 볼 수 있는가",
                                          ["PER이 낮으면 저평가라고 볼 수 있는가"]))
    check("다른 문장 통과", not Q.is_near_duplicate("배당수익률의 정의는 무엇인가",
                                                ["PER 계산식은 주가 나누기 EPS이다"]))
    check("빈 목록 통과", not Q.is_near_duplicate("아무 문장", []))
    # 짧은 문자열은 bigram이 불안정해 완전 일치만 중복으로 본다
    check("짧은 문자열 오탐 없음", not Q.is_near_duplicate("문제A", ["문제B"]))
    check("짧은 문자열 완전일치는 감지", Q.is_near_duplicate("문제A", ["문제A"]))


# =========================================================
# 5. 파이프라인
# =========================================================

def test_pipeline():
    print("\n--- 파이프라인 ---")

    # 샘플 로테이션 + avoid 계약
    reset()
    run([wrong("원본1: EPS 상승시 PER 변화", "PER_BASIC"),
         wrong("원본2: 업종 평균과의 비교", "PER_BASIC")])
    ctxs = [c["wrong_context"] for c in CALLS]
    check("샘플 번갈아 사용",
          "원본1" in ctxs[0] and "원본2" in ctxs[1] and "원본1" in ctxs[2])
    check("한 호출에 샘플 1개만", all(c.count("문제:") == 1 for c in ctxs))
    avoids = [c["avoid_block"] for c in CALLS]
    # 생성이 순차 -> 병렬로 바뀌면서 avoid 계약이 달라졌다 (concurrency.py 참고).
    # 동시에 만들면 서로를 볼 수 없으므로 1단계에는 avoid 힌트가 없고,
    # 실제로 겹친 문제에 한해 2단계(resolve_duplicates)에서 채워 다시 만든다.
    check("1단계는 avoid 없이 동시 생성",
          all(a == "(아직 없음)" for a in avoids), avoids)

    # 출제 관점 순환
    reset()
    run([wrong("원본", "PER_BASIC")])
    check("단일 주제 5슬롯 관점 5개", [c["angle"] for c in CALLS] == R.REVIEW_ANGLES)

    reset()
    run([wrong("PER1", "PER_BASIC"), wrong("PER2", "PER_BASIC"), wrong("PER3", "PER_BASIC"),
         wrong("배당1", "DIVIDEND"), wrong("배당2", "DIVIDEND")])
    per = [c["angle"] for c in CALLS if c["topic"] == "PER"]
    div = [c["angle"] for c in CALLS if c["topic"] == "배당"]
    check("주제별 관점 독립 순환",
          per == R.REVIEW_ANGLES[:3] and div == R.REVIEW_ANGLES[:2], f"PER {len(per)} / 배당 {len(div)}")

    # 중복 감지 후 재생성
    reset([_mc("PER이 낮으면 저평가라고 단정할 수 있는가"),
           _mc("PER이 낮으면 저평가라고 단정할 수 있는가"),
           _mc("PER 을 계산할 때 분모로 쓰는 값은 무엇인가"),
           _mc("업종 평균과 비교해야 하는 이유로 옳은 것은"),
           _mc("적자 기업에서 이 지표를 쓰기 어려운 까닭은"),
           _mc("일회성 이익이 반영되면 나타나는 현상으로 옳은 것은")])
    qs = run([wrong("원본", "PER_BASIC")])
    check("중복 감지 -> 재생성 호출", len(CALLS) == 6, len(CALLS))
    check("중복 문제 제외됨", qs[1]["question_text"] != qs[0]["question_text"])

    # 해설 가드 (학습자 지칭)
    reset([_mc("PER 해석 시 주의할 점은 무엇인가", "학생이 고른 오답은 틀렸습니다"),
           _mc("PER 해석 시 주의할 점은 무엇인가", "일반적인 해설입니다")])
    qs = run([wrong("원본", "PER_BASIC")])
    check("학습자 지칭 해설 거부", "학생이" not in qs[0]["explanation"])

    # OX 의문문 거부
    reset([_ox("배당수익률이 높으면 좋은가?"),
           _ox("배당수익률이 높으면 항상 우량한 기업이다."),
           _ox("배당은 기업 이익에서 지급된다."),
           _ox("배당락일에는 주가가 조정된다."),
           _ox("배당성향은 순이익 대비 배당금 비율이다."),
           _ox("분기 배당을 실시하는 기업도 있다.")])
    qs = run([wrong("배당 원본", "DIVIDEND", qtype="OX")])
    check("OX 의문문 거부", not qs[0]["question_text"].endswith("?"))
    check("OX 유형 유지", all(q["question_type"] == "OX" for q in qs))

    # 영어 혼입 거부 (실측: "실적 alone 이 좋아도 valuation 의 숫자는...")
    reset([_mc("PER 계산식으로 옳은 것은 무엇인가", "실적 alone 이 좋아도 valuation 은 낮아집니다"),
           _mc("PER 계산식으로 옳은 것은 무엇인가", "주가를 EPS 로 나눈 값입니다.")])
    qs = run([wrong("원본", "PER_BASIC")])
    check("해설 영어 혼입 거부", "alone" not in qs[0]["explanation"])

    # 빈 오답
    try:
        run([])
        check("빈 오답 예외", False)
    except R.NoWrongAnswerError:
        check("빈 오답 -> NoWrongAnswerError", True)


def test_korean_guard():
    print("\n--- 한국어 혼입 가드 ---")
    cases = [
        ("실측 사례 (alone/valuation)", "실적 alone 이 좋아도 valuation 의 숫자는 낮아집니다.", False),
        ("소문자 영단어", "이 지표는 growth 관점에서 봐야 합니다.", False),
        ("PER/EPS 약어 허용", "PER 은 주가를 EPS 로 나눈 값입니다.", True),
        ("ROE/ETF/OX 약어 허용", "ROE 와 ETF, OX 문제는 모두 정상입니다.", True),
        ("한글만", "배당수익률은 연간 배당금을 주가로 나눈 값입니다.", True),
    ]
    for name, text, should_pass in cases:
        try:
            Q.assert_korean(text, "검사")
            passed = True
        except AssertionError:
            passed = False
        check(f"한국어 가드 - {name}", passed == should_pass)


# =========================================================
# 6. 정답/해설 검증
# =========================================================

def test_validation():
    print("\n--- 정답/해설 배치 검증 ---")

    reset(validate={"results": [{"no": i, "ok": True, "reason": ""} for i in range(1, 6)]})
    qs = run([wrong("원본", "PER_BASIC")])
    check("전원 합격 시 생성 5회만", len(CALLS) == 5, len(CALLS))
    check("문제 5개 반환", len(qs) == 5)

    reset(validate={"results": [{"no": 2, "ok": False, "reason": "해설과 정답 불일치"}]})
    qs = run([wrong("원본", "PER_BASIC")])
    check("불합격 1건 -> 생성 6회", len(CALLS) == 6, len(CALLS))
    check("재생성 시 동일 관점 유지", CALLS[5]["angle"] == CALLS[1]["angle"])
    check("문제 개수 5개 유지", len(qs) == 5)
    check("카테고리 매핑 승계", qs[1]["primary_detail_code"] == "PER_BASIC")

    reset(validate={"깨진": "응답"})
    check("검증 실패해도 5문제 반환", len(run([wrong("원본", "PER_BASIC")])) == 5)


if __name__ == "__main__":
    test_grouping()
    test_allocation()
    test_context()
    test_dedup()
    test_pipeline()
    test_korean_guard()
    test_validation()

    print("\n" + "=" * 55)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("ALL PASS")
