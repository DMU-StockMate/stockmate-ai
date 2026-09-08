"""일반/프롬프트 기반 문제 생성(app/services/quiz/generator.py) 회귀 테스트.

품질 장치(app/services/quiz/quality.py)를 오답 기반 생성에서 공통 모듈로 분리한 뒤,
`/quiz/generate` 와 `/quiz/generate/prompt` 에도 동일하게 적용됐는지 확인한다.

LLM은 호출하지 않는다. langchain 체인을 스텁으로 갈아끼워 프롬프트 내용과
파이프라인 동작만 검증하므로 Ollama 없이 실행 가능하다.

실행:
    uv run python scripts/test_quiz_generator.py
"""
import asyncio
import json
import re
import math
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# =========================================================
# 스텁
# =========================================================

CALLS: list[dict] = []        # {"kind": analyze|map|validate|generate, **프롬프트 인자}
RESPONSES: list[str] = []     # 생성 호출용 미리 준비된 응답
ANALYZE_RESULT: dict = {}
VALIDATE_RESULT: dict = {"results": []}
BANK_SIMILARITY = None   # 퀴즈 뱅크가 돌려줄 유사도 (None이면 결과 없음)
BANK_UPSERTS: list = []  # 뱅크에 저장된 호출 기록
BANK_TOPIC_COUNT = 0     # 관점 오프셋용 - 주제별 기존 문제 수
FALLBACK_SEQ = 0         # 폴백 문장 순번 - 덩어리가 나뉘어도 겹치지 않게 한다


class _Resp:
    def __init__(self, content):
        self.content = content


# correct=None 이면 스텁이 "프롬프트가 요구한 정답 위치"를 그대로 따른다.
# 정답 위치 검사가 생긴 뒤로, 대부분의 테스트는 위치가 관심사가 아니라
# 지시를 따르기만 하면 된다. 위치 불이행을 시험할 때만 숫자를 명시한다.
def _mc(question, explanation="정상적인 해설 문장입니다.", correct=None, choices=None):
    return json.dumps({
        "question_text": question,
        "choices": choices or [{"no": i, "text": f"보기{i}"} for i in range(1, 5)],
        "correct_no": correct,
        "explanation": explanation,
        "topic": "PER",
    }, ensure_ascii=False)


def _ox(question, explanation="정상적인 해설 문장입니다.", answer="X"):
    return json.dumps({
        "question_text": question, "answer": answer,
        "explanation": explanation, "topic": "배당",
    }, ensure_ascii=False)


_FALLBACK = [
    "기업의 주당순이익을 구하는 올바른 계산식은 무엇인가",
    "업종마다 적정 밸류에이션 수준이 다른 배경으로 옳은 설명은",
    "적자 상태인 회사에서 이 지표를 활용하기 어려운 까닭은",
    "일회성 처분이익이 실적에 반영되었을 때 나타나는 현상은",
    "성장률이 높은 기업에서 관찰되는 특징으로 알맞은 것은",
    "재무제표의 어느 항목을 함께 확인해야 하는지 고르시오",
]


def _obey_answer_pos(raw: str, prompt_vars: dict) -> str:
    """correct_no 가 null 인 응답을 "지시받은 자리"로 채운다.

    실제 모델이 지시를 따랐을 때의 응답을 흉내내는 것이다.
    """
    if '"correct_no": null' not in raw:
        return raw
    return raw.replace('"correct_no": null',
                       f'"correct_no": {prompt_vars.get("answer_pos", 1)}')


# 묶음 프롬프트("정확히 N개 생성")에 대한 스텁 응답.
#
# 실제 모델은 questions 배열로 N개를 한 번에 준다. 기존 테스트들이 RESPONSES 로
# 개별 문제를 넣어 검사하고 있으므로, 묶음에서도 RESPONSES 를 앞에서부터
# 꺼내 배열로 조립한다. 그래야 "지시 위반 시 재생성" 같은 기존 시나리오가
# 묶음 경로에서도 그대로 성립한다.
def _batch_response(tmpl, kwargs) -> str:
    count = int(kwargs.get("count", 1))
    # 슬롯 블록에서 지정된 정답 위치를 뽑아 각 문제에 적용한다
    positions = [int(m) for m in re.findall(r"정답은 반드시 (\d)번", kwargs.get("slot_block", ""))]
    items = []
    for i in range(count):
        if RESPONSES:
            raw = RESPONSES.pop(0)
        else:
            # 전역 순번을 쓴다. 호출 수로 계산하면 덩어리가 병렬로 돌 때
            # 같은 문장이 두 덩어리에 나와 스텁이 스스로 중복을 만든다.
            global FALLBACK_SEQ
            text = _FALLBACK[FALLBACK_SEQ % len(_FALLBACK)]
            FALLBACK_SEQ += 1
            raw = (_ox(text.rstrip("?는은") + " 라고 볼 수 있다.")
                   if tmpl.is_ox else _mc(text))
        item = json.loads(raw)
        if not tmpl.is_ox and item.get("correct_no") is None:
            item["correct_no"] = positions[i] if i < len(positions) else 1
        items.append(item)
    return json.dumps({"questions": items}, ensure_ascii=False)


def _install_stubs():
    lc_core = types.ModuleType("langchain_core")
    lc_prompts = types.ModuleType("langchain_core.prompts")

    class _Template:
        @classmethod
        def from_template(cls, template):
            obj = cls()
            obj.template = template
            if "검수하는 전문가" in template:
                obj.kind = "validate"
            elif "퀴즈 출제 계획을" in template:
                obj.kind = "analyze"
            elif "카테고리 목록에 매핑" in template:
                obj.kind = "map"
            else:
                obj.kind = "generate"
            # 폴백 응답을 프롬프트 유형에 맞게 내기 위한 표시
            obj.is_ox = ("OX 문제를 1개 생성" in template
                         or "OX 문제를 **정확히" in template)
            # 한 번에 여러 문제를 만드는 묶음 프롬프트인지
            obj.is_batch = "정확히 {count}개** 생성" in template
            return obj

        def __or__(self, other):
            return self

        def format(self, **kwargs):
            return self.template.format(**kwargs)

        async def ainvoke(self, kwargs):
            CALLS.append({"kind": self.kind, **kwargs})
            if self.kind == "validate":
                return _Resp(json.dumps(VALIDATE_RESULT, ensure_ascii=False))
            if self.kind == "analyze":
                return _Resp(json.dumps(ANALYZE_RESULT, ensure_ascii=False))
            if self.kind == "map":
                return _Resp(json.dumps({"mappings": []}, ensure_ascii=False))
            if self.is_batch:
                return _Resp(_batch_response(self, kwargs))
            if RESPONSES:
                return _Resp(_obey_answer_pos(RESPONSES.pop(0), kwargs))
            n = len([c for c in CALLS if c["kind"] == "generate"])
            text = _FALLBACK[(n - 1) % len(_FALLBACK)]
            if self.is_ox:
                # OX는 평서문이어야 하므로 폴백 문장을 평서문으로 바꿔 쓴다
                return _Resp(_ox(text.rstrip("?는은") + " 라고 볼 수 있다."))
            return _Resp(_obey_answer_pos(_mc(text), kwargs))

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

    config = types.ModuleType("app.core.config")

    class _Settings:
        # 백엔드 전환이 생기면서 build_llm 이 LLM_BACKEND 를 본다.
        # 스텁에 없으면 AttributeError 로 테스트가 죽으므로 함께 둔다.
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
        # 프롬프트 경로의 기본 문제 수 (운영 기본값과 동일).
        PROMPT_QUIZ_DEFAULT_COUNT = 3
        # 검증 모드. 테스트는 재생성 동작까지 보므로 blocking 으로 둔다
        # (운영 기본값 background 는 test_validate_mode 가 따로 본다).
        QUIZ_VALIDATE_MODE = "blocking"
        ANALYZE_REASONING_EFFORT = ""
        QUIZ_BATCH_MAX = 3
        QUIZ_BATCH_TOKENS_PER_ITEM = 900
        LLM_MAX_TOKENS = 4096


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

from app.core.config import settings as SETTINGS   # noqa: E402  (스텁 _Settings 인스턴스)
from app.schemas.chat import UserContext            # noqa: E402
from app.services.quiz import generator as G        # noqa: E402
from app.services.quiz import quality as Q          # noqa: E402

FAILURES: list[str] = []
USER = UserContext(user_id=1, investment_level="초급")


def check(name, passed, extra=""):
    print(("PASS  " if passed else "FAIL  ") + name + (f"  {extra}" if extra else ""))
    if not passed:
        FAILURES.append(name)


def reset(responses=None, analyze=None, validate=None):
    global FALLBACK_SEQ
    FALLBACK_SEQ = 0
    CALLS.clear()
    RESPONSES.clear()
    if responses:
        RESPONSES.extend(responses)
    global ANALYZE_RESULT, VALIDATE_RESULT
    ANALYZE_RESULT = analyze or {}
    VALIDATE_RESULT = validate if validate is not None else {"results": []}


def gen_calls():
    return [c for c in CALLS if c["kind"] == "generate"]


# 묶음 생성으로 바뀌면서 프롬프트 변수가 달라졌다.
#   단건: {"angle": ..., "avoid_block": ..., "answer_pos": ...}
#   묶음: {"slot_block": "[1번 문제]\n<관점>\n→ 정답은 반드시 N번 ...", "count": N}
# 기존 테스트들의 검증 의도(관점 회전·정답 위치 분산)를 그대로 살리기 위해
# 두 형태를 모두 읽는 헬퍼를 둔다.

def angles_of(call) -> list:
    """호출 하나가 지시한 출제 관점들 (단건이면 1개)."""
    sb = call.get("slot_block")
    if not sb:
        return [call["angle"]] if call.get("angle") else []
    return [b.strip() for b in
            re.findall(r"\[\d+번 문제\]\n(.*?)(?=\n→|\n\[|\Z)", sb, re.S)]


def positions_of(call) -> list:
    """호출 하나가 지시한 정답 위치들."""
    sb = call.get("slot_block")
    if not sb:
        return [call["answer_pos"]] if call.get("answer_pos") else []
    return [int(m) for m in re.findall(r"정답은 반드시 (\d)번", sb)]


def all_angles(calls=None) -> list:
    """생성 호출 전체가 지시한 관점을 순서대로 편다."""
    return [a for c in (calls if calls is not None else gen_calls())
            for a in angles_of(c)]


# =========================================================
# 1. 품질 규칙이 프롬프트에 실제로 들어갔는가
# =========================================================

def test_prompts_carry_quality_rules():
    print("\n--- 프롬프트에 품질 규칙 반영 ---")
    mc = G.MC_PROMPT.template
    ox = G.OX_PROMPT.template

    check("MC: 영어 혼입 금지", "영어 단어를 그대로 섞어" in mc)
    check("MC: 오답이 사실이면 안 됨", '오답 선택지 3개는 모두 "사실이 아닌 내용"' in mc)
    check("MC: 문제/해설 방향 일치", "묻는 방향과 해설이 설명하는 방향" in mc)
    check("MC: 지표 정의 명시", "PER = 주가 / 주당순이익" in mc)
    check("MC: 전제-정답 모순 금지", "전제와 정답이 서로 모순" in mc)

    check("OX: 이중부정 금지", "부정 표현이 두 번 들어간 문장" in ox)
    check("OX: 평서문 강제", "평서문" in ox)
    check("OX: 영어 혼입 금지", "영어 단어를 그대로 섞어" in ox)
    check("OX: 지표 정의 명시", "배당수익률 = 연간 배당금 / 주가" in ox)


# =========================================================
# 2. 결정론적 검사가 생성 경로에 걸리는가
# =========================================================

def test_deterministic_checks():
    print("\n--- 생성 시 결정론적 검사 ---")

    # 영어 혼입 -> 재시도 후 정상본 채택
    reset([_mc("PER 계산식으로 옳은 것은", "실적 alone 이 좋아도 valuation 은 낮아집니다"),
           _mc("PER 계산식으로 옳은 것은", "주가를 EPS 로 나눈 값입니다.")])
    q = asyncio.run(G.generate_quiz(USER, "MULTIPLE_CHOICE", "PER"))
    check("해설 영어 혼입 거부", "alone" not in q["explanation"])
    check("영어 혼입 시 재생성 발생", len(gen_calls()) == 2, len(gen_calls()))

    # 선택지에 영어가 섞여도 거부
    reset([_mc("PER 계산식으로 옳은 것은", choices=[
               {"no": 1, "text": "valuation 기준으로 계산"},
               {"no": 2, "text": "주가 나누기 주당순이익"},
               {"no": 3, "text": "매출액 기준"}, {"no": 4, "text": "자산 기준"}]),
           _mc("PER 계산식으로 옳은 것은")])
    q = asyncio.run(G.generate_quiz(USER, "MULTIPLE_CHOICE", "PER"))
    check("선택지 영어 혼입 거부",
          all("valuation" not in c["text"] for c in q["choices"]))

    # OX 의문문 -> 거부
    reset([_ox("배당수익률이 높으면 좋은가?"), _ox("배당수익률이 높으면 항상 우량하다.")])
    q = asyncio.run(G.generate_quiz(USER, "OX", "배당"))
    check("OX 의문문 거부", not q["question_text"].endswith("?"))

    # 선택지 번호 이상 -> 거부
    reset([_mc("PER 문제", choices=[{"no": 1, "text": "a"}, {"no": 1, "text": "b"},
                                   {"no": 3, "text": "c"}, {"no": 4, "text": "d"}]),
           _mc("정상적인 PER 계산 문제입니다")])
    q = asyncio.run(G.generate_quiz(USER, "MULTIPLE_CHOICE", "PER"))
    check("선택지 번호 중복 거부", q["question_text"] == "정상적인 PER 계산 문제입니다")


# =========================================================
# 3. /quiz/generate — 배치 검증 + 근사 중복
# =========================================================

def test_generate_batch():
    print("\n--- generate_quiz_batch (/quiz/generate) ---")

    reset(validate={"results": [{"no": i, "ok": True, "reason": ""} for i in range(1, 4)]})
    qs = asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=3))
    check("count 유지", len(qs) == 3, len(qs))
    check("검증 호출 1회", len([c for c in CALLS if c["kind"] == "validate"]) == 1)
    # 3문제를 **한 번의 호출**로 만든다 (QUIZ_BATCH_MAX=3).
    # 하나씩 병렬로 만들면 서로를 못 봐서 중복이 심하다는 측정 결과에 따른 것이다
    # (2026-09-08: 같은 주제 3문제 중복 발동률 77%). generator.py 주석 참고.
    check("3문제를 1회 호출로", len(gen_calls()) == 1, f"{len(gen_calls())}회")

    # 상한을 넘으면 여러 덩어리로 나눠 병렬 호출한다
    reset(validate={"results": []})
    qs = asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=5))
    check("5문제는 2덩어리(3+2)", len(gen_calls()) == 2, f"{len(gen_calls())}회")
    check("5문제 모두 반환", len(qs) == 5, len(qs))

    # 검증 불합격 -> 그 문제만 단건으로 재생성 (묶음 1회 + 재생성 1회)
    reset(validate={"results": [{"no": 2, "ok": False, "reason": "정답/해설 불일치"}]})
    qs = asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=3))
    check("불합격 시 재생성", len(gen_calls()) == 2, f"{len(gen_calls())}회")
    check("개수는 그대로", len(qs) == 3)

    # 근사 중복 -> 걸린 문제만 단건 재생성 (묶음 1회 + 재생성 1회)
    reset([_mc("PER이 낮으면 저평가라고 단정할 수 있는가"),
           _mc("PER이 낮으면 저평가라고 단정할 수 있는가"),
           _mc("PER 을 계산할 때 분모로 쓰는 값은 무엇인가"),
           _mc("업종 평균과 비교해야 하는 이유로 옳은 것은")])
    qs = asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))
    check("근사 중복 재생성", len(gen_calls()) == 2, f"{len(gen_calls())}회")
    check("중복 재생성 후에도 개수 유지", len(qs) == 2, len(qs))
    check("중복 제거됨", qs[0]["question_text"] != qs[1]["question_text"])

    # 검증이 깨져도 결과 반환
    reset(validate={"깨진": "응답"})
    check("검증 실패해도 결과 반환",
          len(asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))) == 2)


# =========================================================
# 4. /quiz/generate/prompt — 계획 -> 생성 -> 검증
# =========================================================

def test_generate_from_prompt():
    print("\n--- generate_quiz_from_prompt (/quiz/generate/prompt) ---")

    plan = {"relevant": True, "count": 2, "items": [
        {"topic": "PER", "question_type": "MULTIPLE_CHOICE",
         "category_code": "BEGINNER_FINANCIAL_BASIC", "detail_codes": ["PER_BASIC"]},
        {"topic": "배당", "question_type": "MULTIPLE_CHOICE",
         "category_code": "BEGINNER_STOCK_BASIC", "detail_codes": ["DIVIDEND"]},
    ]}

    reset(analyze=plan,
          validate={"results": [{"no": 1, "ok": True, "reason": ""},
                                {"no": 2, "ok": True, "reason": ""}]})
    qs = asyncio.run(G.generate_quiz_from_prompt("PER이랑 배당 문제 내줘", USER))
    check("계획대로 2문제", len(qs) == 2, len(qs))
    check("카테고리 매핑 유지", qs[0]["category_code"] == "BEGINNER_FINANCIAL_BASIC")
    check("primary_detail_code 설정", qs[0]["primary_detail_code"] == "PER_BASIC")
    check("검증 호출 1회", len([c for c in CALLS if c["kind"] == "validate"]) == 1)

    # 검증 불합격 시 재생성해도 카테고리 매핑은 승계
    reset(analyze=plan, validate={"results": [{"no": 1, "ok": False, "reason": "오답에 사실 섞임"}]})
    qs = asyncio.run(G.generate_quiz_from_prompt("PER이랑 배당 문제 내줘", USER))
    check("재생성 후 매핑 승계", qs[0]["category_code"] == "BEGINNER_FINANCIAL_BASIC")
    check("재생성 후 개수 유지", len(qs) == 2)

    # 주제 파악 실패 -> PromptTopicError
    reset(analyze={"relevant": False})
    try:
        asyncio.run(G.generate_quiz_from_prompt("오늘 점심 뭐 먹지", USER))
        check("무관한 프롬프트 예외", False)
    except G.PromptTopicError:
        check("무관한 프롬프트 -> PromptTopicError", True)


def test_count_override():
    """요청 count 가 프롬프트에서 읽은 개수보다 우선해야 한다.

    백엔드가 count=1 을 보냈는데 3개가 오는 문제를 실제로 겪었다.
    이 경로는 count 를 아예 받지 않고 있었다 (pydantic 이 여분 필드를 조용히 버림).
    """
    print("\n--- 요청 count 우선 (/quiz/generate/prompt) ---")

    plan = {"relevant": True, "count": 3, "items": [
        {"topic": "PER", "question_type": "MULTIPLE_CHOICE",
         "category_code": "BEGINNER_FINANCIAL_BASIC", "detail_codes": ["PER_BASIC"]},
        {"topic": "배당", "question_type": "MULTIPLE_CHOICE",
         "category_code": "BEGINNER_STOCK_BASIC", "detail_codes": ["DIVIDEND"]},
        {"topic": "PBR", "question_type": "MULTIPLE_CHOICE",
         "category_code": "BEGINNER_FINANCIAL_BASIC", "detail_codes": ["PBR_BASIC"]},
    ]}
    ok3 = {"results": [{"no": i, "ok": True, "reason": ""} for i in (1, 2, 3)]}

    reset(analyze=plan, validate=ok3)
    qs = asyncio.run(G.generate_quiz_from_prompt("PER 문제 만들어줘", USER))
    check("count 미지정이면 계획대로 3개", len(qs) == 3, len(qs))

    reset(analyze=plan, validate=ok3)
    qs = asyncio.run(G.generate_quiz_from_prompt("PER 문제 만들어줘", USER, count=1))
    check("count=1 이면 1개만", len(qs) == 1, len(qs))
    check("잘라도 매핑은 유지", qs[0]["category_code"] == "BEGINNER_FINANCIAL_BASIC")

    # 계획보다 많이 요청하면 기존 항목을 순환 복제해 채운다
    reset(analyze=plan,
          validate={"results": [{"no": i, "ok": True, "reason": ""} for i in range(1, 6)]})
    qs = asyncio.run(G.generate_quiz_from_prompt("PER 문제 만들어줘", USER, count=5))
    check("count=5 면 5개로 채움", len(qs) == 5, len(qs))

    # 상한 밖 값은 잘라낸다 (라우터의 pydantic 검증과 별개로 방어)
    check("resize_plan 상한", len(G.resize_plan(list(plan["items"]), 99)) == G.MAX_PROMPT_QUIZ_COUNT)
    check("resize_plan 하한", len(G.resize_plan(list(plan["items"]), 0)) == 1)


# =========================================================
# 4-b. 문제 유형 통일 / 카테고리 등급 우선
# =========================================================

def test_question_type_enforcement():
    print("\n--- 문제 유형 통일 ---")

    def plan_with(types):
        return {"relevant": True, "count": len(types), "items": [
            {"topic": "배당수익률", "question_type": t,
             "category_code": "BEGINNER_STOCK_BASIC", "detail_codes": ["DIVIDEND"]}
            for t in types
        ]}

    # 실측 버그: "OX 3개" 요청인데 계획에 객관식이 섞여 나옴
    reset(analyze=plan_with(["OX", "MULTIPLE_CHOICE", "OX"]))
    qs = asyncio.run(G.generate_quiz_from_prompt("배당수익률 OX 문제 3개 내줘", USER))
    check("OX 명시 시 전부 OX", all(q["question_type"] == "OX" for q in qs),
          [q["question_type"] for q in qs])

    # 유형 언급 없이 섞여 나오면 객관식으로 통일
    reset(analyze=plan_with(["OX", "MULTIPLE_CHOICE"]))
    qs = asyncio.run(G.generate_quiz_from_prompt("배당 문제 내줘", USER))
    check("언급 없으면 객관식 통일",
          all(q["question_type"] == "MULTIPLE_CHOICE" for q in qs),
          [q["question_type"] for q in qs])

    # "섞어서" 요청이면 혼합 유지
    reset(analyze=plan_with(["OX", "MULTIPLE_CHOICE"]))
    qs = asyncio.run(G.generate_quiz_from_prompt("배당 문제 섞어서 내줘", USER))
    check("섞어서 요청은 혼합 유지",
          len({q["question_type"] for q in qs}) == 2,
          [q["question_type"] for q in qs])

    # 순수 함수 단위 - 한국어 조사가 붙은 표현까지 인식해야 한다
    ox_prompts = ["OX로 3개 내줘", "OX만 내줘", "OX 문제", "O/X로 내줘",
                  "오엑스 문제 내줘", "OX퀴즈 3개"]
    for prompt in ox_prompts:
        items = [{"question_type": "MULTIPLE_CHOICE"} for _ in range(3)]
        G._enforce_question_type(prompt, items)
        check(f"OX 인식 - '{prompt}'", all(i["question_type"] == "OX" for i in items))

    # OX가 단어 일부인 경우는 오탐하면 안 된다
    for prompt in ["BOX 관련 문제", "OXFORD 대학 문제"]:
        items = [{"question_type": "MULTIPLE_CHOICE"} for _ in range(2)]
        G._enforce_question_type(prompt, items)
        check(f"오탐 없음 - '{prompt}'",
              all(i["question_type"] == "MULTIPLE_CHOICE" for i in items))


def test_remap_prefers_user_level():
    print("\n--- 재매핑 시 등급 우선 ---")
    # 1차 매핑 실패(category_code=null) -> 2차 재매핑 경로
    plan = {"relevant": True, "count": 1, "items": [
        {"topic": "배당수익률", "question_type": "MULTIPLE_CHOICE",
         "category_code": None, "detail_codes": []},
    ]}
    reset(analyze=plan)
    asyncio.run(G.generate_quiz_from_prompt("배당수익률 문제 내줘", USER))
    remap = [c for c in CALLS if c["kind"] == "map"]
    check("재매핑 호출됨", len(remap) == 1, len(remap))
    check("선호 등급이 유저 등급", remap and remap[0]["preferred_level"] == "초급",
          remap[0]["preferred_level"] if remap else None)
    check("재매핑 프롬프트에 등급 우선 지시",
          "등급을 먼저 고르세요" in G.MAP_TOPICS_PROMPT.template)

    # 입문 유저는 초급 카탈로그를 쓰므로 선호 등급도 초급
    reset(analyze=plan)
    asyncio.run(G.generate_quiz_from_prompt(
        "배당수익률 문제 내줘", UserContext(user_id=2, investment_level="입문")))
    remap = [c for c in CALLS if c["kind"] == "map"]
    check("입문 -> 초급으로 매핑", remap and remap[0]["preferred_level"] == "초급",
          remap[0]["preferred_level"] if remap else None)


# =========================================================
# 5. 출제 관점 로테이션 (중복 방지)
# =========================================================

def test_angle_rotation():
    print("\n--- 출제 관점 로테이션 ---")

    # 같은 topic으로 여러 개 -> 관점이 매번 달라야 한다
    reset()
    asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=4))
    angles = all_angles()
    check("배치 4개 관점 모두 다름", len(set(angles)) == 4, len(set(angles)))
    check("관점 순서가 정의 순서대로", angles == Q.GENERATION_ANGLES[:4])

    # 묶음 생성으로 바뀌면서 1단계에는 avoid_block 자체가 없다.
    # 한 번의 호출로 N개를 쓰게 하므로 모델이 서로를 보고 다르게 만든다
    # (그 근거는 generator.py 의 묶음 생성 주석 참고).
    check("1단계 호출에는 avoid 개념이 없음",
          all("avoid_block" not in c for c in gen_calls()),
          [list(c)[:3] for c in gen_calls()])

    # 프롬프트 기반: 주제별로 독립 순환
    plan = {"relevant": True, "count": 5, "items": [
        {"topic": t, "question_type": "MULTIPLE_CHOICE",
         "category_code": "BEGINNER_FINANCIAL_BASIC", "detail_codes": ["PER_BASIC"]}
        for t in ["PER", "배당", "PER", "배당", "PER"]
    ]}
    reset(analyze=plan)
    asyncio.run(G.generate_quiz_from_prompt("PER이랑 배당 문제 내줘", USER))
    per = all_angles([c for c in gen_calls() if c["topic"] == "PER"])
    div = all_angles([c for c in gen_calls() if c["topic"] == "배당"])
    check("PER 3개 관점 다름", len(set(per)) == 3, len(set(per)))
    check("배당 2개 관점 다름", len(set(div)) == 2, len(set(div)))
    check("주제별 독립 순환", per[0] == div[0] == Q.GENERATION_ANGLES[0])

    # 단건 생성은 관점 지정 없이도 동작
    reset()
    asyncio.run(G.generate_quiz(USER, "MULTIPLE_CHOICE", "PER"))
    check("단건 생성 기본 관점", angles_of(gen_calls()[0])[0] == Q.GENERATION_ANGLES[0])

    # 재요청 시 과거 이력만큼 관점을 밀어야 1차와 안 겹친다
    global BANK_TOPIC_COUNT
    reset()
    BANK_TOPIC_COUNT = 2
    try:
        asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))
        angles = all_angles()
        check("과거 이력만큼 관점 시작점 이동",
              angles == Q.GENERATION_ANGLES[2:4], angles == Q.GENERATION_ANGLES[2:4])
    finally:
        BANK_TOPIC_COUNT = 0

    # 관점 개수를 넘어가면 순환
    reset()
    BANK_TOPIC_COUNT = len(Q.GENERATION_ANGLES)
    try:
        asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=1))
        check("오프셋이 관점 수를 넘으면 순환",
              angles_of(gen_calls()[0])[0] == Q.GENERATION_ANGLES[0])
    finally:
        BANK_TOPIC_COUNT = 0


# =========================================================
# 6. 한국어 가드 (영어 + 한자/가나)
# =========================================================

def test_language_guard():
    print("\n--- 한국어 혼입 가드 ---")
    cases = [
        ("영어 소문자", "실적 alone 이 좋아도 valuation 은 낮아집니다.", False),
        ("한자 (실측)", "주가 상승률이 항상 높은 公司股票 입니다.", False),
        ("일본어 가나", "이 지표는 かぶ 라고도 부릅니다.", False),
        ("PER/EPS 약어 허용", "PER 은 주가를 EPS 로 나눈 값입니다.", True),
        ("ROE/ETF 약어 허용", "ROE 와 ETF 는 모두 정상입니다.", True),
        ("한글만", "배당수익률은 연간 배당금을 주가로 나눈 값입니다.", True),
    ]
    for name, text, should_pass in cases:
        try:
            Q.assert_korean(text, "검사")
            passed = True
        except AssertionError:
            passed = False
        check(f"가드 - {name}", passed == should_pass)

    # 생성 경로에서 한자 혼입이 거부되는지
    reset([_mc("PER 계산식으로 옳은 것은", "높은 公司股票 를 의미합니다"),
           _mc("PER 계산식으로 옳은 것은", "주가를 EPS 로 나눈 값입니다.")])
    q = asyncio.run(G.generate_quiz(USER, "MULTIPLE_CHOICE", "PER"))
    check("생성 시 한자 혼입 거부", "公司" not in q["explanation"])


# =========================================================
# 5-b. 주제 이탈 차단
# =========================================================

def test_topic_guard():
    print("\n--- 주제 이탈 차단 ---")

    # 실측: 주제가 "주식"인데 PER 문제가 나와 라벨이 틀어졌다 (이탈률 32%)
    cases = [
        ("주식", "", "기업의 주가가 크게 떨어졌을 때 PER 이 낮아지는 이유는?", False),
        ("주가", "", "PER 이 높은 주식은 무엇을 의미하는가?", False),
        ("주식", "", "주식을 보유한 주주가 가지는 권리로 옳은 것은?", True),
        ("PER", "", "PER 을 계산할 때 분모에 들어가는 값은?", True),
        ("배당", "", "배당수익률이 높으면 항상 좋은 기업인가?", True),
        ("거래량", "", "거래량이 급증했을 때 해석으로 옳은 것은?", True),
        ("", "", "주제가 없으면 판단하지 않는다", True),

        # --- 상위 개념 주제 (실측 오탐: 정상 출제를 이탈로 막아 주제를 통째로 날렸다) ---
        # 볼린저밴드는 보조지표의 한 종류다
        ("보조지표", "기술적 보조지표를 투자 근거로 활용",
         "볼린저밴드의 상단선에 주가가 닿았을 때 해석으로 옳은 것은?", True),
        ("보조지표", "기술적 보조지표를 투자 근거로 활용",
         "이동평균선이 우상향할 때의 의미로 옳은 것은?", True),
        # 상위 개념이라도 계열이 다르면 여전히 이탈이다
        ("보조지표", "기술적 보조지표를 투자 근거로 활용",
         "PER 이 낮은 기업의 특징으로 옳은 것은?", False),
        # 기업 비교는 지표 사용이 곧 주제다
        ("같은 업종 내 기업 비교", "동일 업종 기업의 지표와 성과를 비교",
         "같은 업종의 두 기업 중 PER 이 더 낮은 쪽을 어떻게 볼 수 있는가?", True),
        # 설명이 없으면 주제명만으로 판단한다 (review 경로)
        ("보조지표", "", "RSI 가 70을 넘었을 때 해석으로 옳은 것은?", True),

        # --- 주인공 판정 (실측: 채택분의 25% 가 주제를 사칭한 PER 문제였다) ---
        # PER 의 정의가 "주가/주당순이익"이라 PER 문제에는 "주가"가 반드시 나온다.
        # 주제어가 들어 있다는 이유만으로 통과시키면 안 된다.
        ("주가", "", "PER 이 같은 수준에서 주가가 오르면 PER 값은 어떻게 변하는가?", False),
        ("주가", "", "PER(주가수익비율) 수치를 계산할 때 분모로 쓰는 값은?", False),
        # 주제어가 더 자주 나오면 그 주제의 문제다
        ("주가", "", "주가가 오르내리는 이유와 주가를 결정하는 요소로 옳은 것은?", True),
        # 다른 개념을 도구로만 쓴 정상 문제는 통과해야 한다 (같은 횟수)
        ("안정형", "", "안정형 투자자가 PER 이 낮은 주식을 고를 때 유의할 점은?", True),
        # 주제 개념의 구성 요소는 몇 번 나와도 이탈이 아니다
        ("PER", "", "PER 은 주가를 주당순이익(EPS)으로 나눈 값이며 주당순이익이 줄면 커진다", True),
        # 약어 대신 정식 명칭으로 써도 잡아야 한다
        ("거래량", "", "주가수익비율이 높다는 것은 무엇을 뜻하는가?", False),
    ]
    for topic, desc, text, should_pass in cases:
        try:
            Q.assert_on_topic(text, topic, desc)
            passed = True
        except AssertionError:
            passed = False
        check(f"주제 '{topic or '(없음)'}' - {text[:26]}", passed == should_pass)

    # 생성 경로에서 이탈본이 거부되고 재생성되는지
    reset([_mc("기업의 주가가 떨어질 때 PER 이 낮아지는 이유는?"),
           _mc("주식을 보유한 주주가 가지는 권리로 옳은 것은?")])
    q = asyncio.run(G.generate_quiz(USER, "MULTIPLE_CHOICE", "주식"))
    check("이탈본 거부 후 재생성", "PER" not in q["question_text"], q["question_text"][:30])

    # 난이도 가이드에 지표명이 없어야 한다 (이것 때문에 이탈이 났었다)
    for level, guide in G.LEVEL_GUIDE.items():
        check(f"난이도 가이드에 지표명 없음 - {level}",
              not any(x in guide for x in ("PER", "PBR", "ROE", "EPS")))


# =========================================================
# 6-a. 검증 자기 번복 처리
# =========================================================

def test_validation_retraction():
    print("\n--- 검증 사유 자기 번복 ---")

    # 실측: ok=false 인데 사유가 "사실 모든 것이 맞다"로 끝남 -> 재생성이 낭비됨
    retracting = ("...논란 여지가 있는 것으로 보아 주의 필요하나, 실제 검증 시에는 "
                  "모든 항목이 맞다. 다시 읽어보니 문제가 없으므로 사실 모든 것이 맞다.")
    reset(validate={"results": [{"no": 1, "ok": False, "reason": retracting}]})
    asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))
    # 묶음 1회뿐 - 재생성이 없다는 뜻
    check("번복된 사유는 재생성 안 함", len(gen_calls()) == 1, f"{len(gen_calls())}회")

    # 명확한 사유는 그대로 재생성
    reset(validate={"results": [{"no": 1, "ok": False,
                                 "reason": "오답 3번이 사실이라 정답이 두 개"}]})
    asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))
    # 묶음 1회 + 불합격 1건 단건 재생성
    check("명확한 사유는 재생성", len(gen_calls()) == 2, f"{len(gen_calls())}회")

    # 판정 함수 단위
    for text, expect in [
        ("오답 3번이 사실이라 정답이 두 개", False),
        ("해설 방향이 문제와 반대", False),
        ("다시 읽어보니 사실 모든 것이 맞다", True),
        ("논란 여지가 있으나 문제가 아님", True),
        ("확인 결과 이상 없음", True),
    ]:
        check(f"번복 판정 - '{text[:20]}'", Q._reason_retracts(text) == expect)

    # 프롬프트에 번복 금지 지시가 들어갔는지
    check("검증 프롬프트에 번복 금지", "판정을 번복하거나" in Q.VALIDATE_PROMPT.template)
    check("검증 프롬프트에 확신 없으면 통과", "판단이 서지 않으면" in Q.VALIDATE_PROMPT.template)


# =========================================================
# 6-b. 퀴즈 뱅크 연동 (과거 요청과의 중복)
# =========================================================

def test_bank_integration():
    print("\n--- 퀴즈 뱅크 연동 ---")
    global BANK_SIMILARITY

    # 과거 문제와 유사 -> 재생성 유발.
    # 이 경로는 QUIZ_DUP_CHECK_PAST 로 제어되고 운영 기본값이 꺼짐이라
    # (이번 요청 안에서의 중복만 막는 정책) 여기서만 켠다.
    # 끄고 켤 때의 동작 자체는 test_dup_scope 가 따로 본다.
    reset()
    BANK_SIMILARITY = 0.95
    BANK_UPSERTS.clear()
    SETTINGS.QUIZ_DUP_CHECK_PAST = True
    try:
        asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))
        # 묶음 1회 + 2문제가 모두 과거와 중복이라 단건 재생성 2회
        check("과거 중복 감지 시 재생성", len(gen_calls()) == 3, f"{len(gen_calls())}회")
    finally:
        BANK_SIMILARITY = None
        SETTINGS.QUIZ_DUP_CHECK_PAST = False

    # 중복 없으면 재생성 없음
    reset()
    BANK_UPSERTS.clear()
    asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))
    check("중복 없으면 재생성 없음", len(gen_calls()) == 1, f"{len(gen_calls())}회 (묶음 1회)")
    check("확정본이 뱅크에 저장됨", len(BANK_UPSERTS) == 1, len(BANK_UPSERTS))
    check("저장 개수 = 문제 개수",
          len(BANK_UPSERTS[0]["points"]) == 2, len(BANK_UPSERTS[0]["points"]))


# =========================================================
# 7. 공통 모듈 일관성
# =========================================================

def test_shared_module():
    print("\n--- 공통 모듈 일관성 ---")
    from app.services.quiz import review as R

    check("generator/review가 같은 QUALITY_RULES 사용",
          Q.QUALITY_RULES in G.MC_PROMPT.template
          and Q.QUALITY_RULES in R.REVIEW_MC_PROMPT.template)
    check("generator/review가 같은 OX 규칙 사용",
          Q.OX_QUALITY_RULES in G.OX_PROMPT.template
          and Q.OX_QUALITY_RULES in R.REVIEW_OX_PROMPT.template)
    check("하위 호환 별칭 유지", G._get_llm is Q.get_llm and G._extract_json is Q.extract_json)
    check("검증 프롬프트 단일 정의", not hasattr(R, "VALIDATE_PROMPT"))
    check("관점 목록 공유", R.REVIEW_ANGLES is Q.GENERATION_ANGLES)
    check("관점 블록 공유",
          Q.ANGLE_BLOCK in G.MC_PROMPT.template
          and Q.ANGLE_BLOCK in R.REVIEW_MC_PROMPT.template)


# =========================================================
# 7-b. 정답 위치 분산
# =========================================================

def test_answer_position():
    """정답이 특정 번호에 몰리지 않아야 한다.

    실측 사고: 정답의 78.7%(9B) / 85.7%(gpt-5.4)가 1번이었다. 기대값은 25%.
    원인은 프롬프트 예시 JSON 의 "correct_no": 1 이었다.
    이 상태로 두면 사용자가 1번만 찍어도 대부분 맞고, 학습 데이터로 쓰면
    파인튜닝 모델이 그 편향을 그대로 배운다.
    """
    print("\n--- 정답 위치 분산 ---")

    # 4문제면 1~4번을 한 번씩 써야 한다
    check("위치 순환", [Q.answer_pos_for(i) for i in range(6)] == [1, 2, 3, 4, 1, 2],
          [Q.answer_pos_for(i) for i in range(6)])
    # 관점(5주기)과 서로소여야 조합이 빨리 반복되지 않는다
    check("관점 주기와 서로소",
          math.gcd(len(Q.ANSWER_POSITIONS), len(Q.GENERATION_ANGLES)) == 1)

    # 프롬프트에 예시 정답이 박혀 있으면 안 된다 (이것이 원인이었다)
    check("MC 프롬프트에 correct_no 고정값 없음",
          '"correct_no": 1' not in G.MC_PROMPT.template)
    check("MC 프롬프트가 지정 위치를 요구", "{answer_pos}" in G.MC_PROMPT.template)

    # 위치 지시문에 4 외의 개수를 적으면 모델이 그 수만큼 선택지를 만든다.
    # 실측: "나머지 세 자리에" 때문에 선택지 3개짜리가 탈락분의 18.2% 를 차지했다.
    for word in ("세 자리", "세 개", "3개", "3 개"):
        check(f"위치 지시문에 '{word}' 없음", word not in Q.ANSWER_POSITION_BLOCK)
    check("위치 지시문이 4개를 명시", "정확히 4개" in Q.ANSWER_POSITION_BLOCK)

    # 주제마다 순환 시작점이 달라야 1번 쏠림(실측 40.0%)이 안 생긴다
    starts = {Q.answer_pos_offset(t) for t in
              ("PER", "배당", "거래량", "주가", "시가총액", "부채비율", "주식", "공시")}
    check("주제별 시작점이 분산됨", len(starts) >= 3, sorted(starts))
    check("시작점이 실행마다 같음",
          Q.answer_pos_offset("PER") == Q.answer_pos_offset("PER"))

    # 지시한 자리에 안 놓으면 거부되는가
    for requested, actual, should_pass in [(3, 3, True), (3, 1, False), (None, 1, True)]:
        try:
            Q.assert_answer_position(actual, requested)
            passed = True
        except AssertionError:
            passed = False
        check(f"지시 {requested} / 실제 {actual}", passed == should_pass)

    # --- 지시를 어겼을 때 자리를 옮겨 고치는가 ---
    # 실측: 9B 는 정답을 4번에 놓으라는 지시를 거의 못 지켰다(negative 의 4번 비율 0%).
    # 매번 재생성시켰더니 5개 주제 중 2개가 재시도를 소진하고 통째로 실패했다.
    data = {
        "question_text": "문제", "correct_no": 1,
        "choices": [{"no": 1, "text": "정답 문장"}, {"no": 2, "text": "오답2"},
                    {"no": 3, "text": "오답3"}, {"no": 4, "text": "오답4"}],
        "explanation": "정답은 1번입니다. 2번과 4번은 사실이 아닙니다.",
    }
    Q.enforce_answer_position(data, 4)
    check("정답 번호 갱신", data["correct_no"] == 4, data["correct_no"])
    texts = {c["no"]: c["text"] for c in data["choices"]}
    check("정답 문장이 4번으로 이동", texts[4] == "정답 문장", texts[4])
    check("4번에 있던 오답은 1번으로", texts[1] == "오답4", texts[1])
    check("건드리지 않은 자리는 그대로", texts[2] == "오답2" and texts[3] == "오답3")
    # 해설이 번호를 언급하면 함께 고쳐야 한다 (해설의 18% 가 번호를 언급한다)
    check("해설 번호도 교체", data["explanation"] == "정답은 4번입니다. 2번과 1번은 사실이 아닙니다.",
          data["explanation"])

    # 수치 선택지는 순서에 뜻이 있을 수 있어 바꾸지 않는다
    numeric = {
        "question_text": "문제", "correct_no": 1,
        "choices": [{"no": i, "text": f"{i * 10}%"} for i in range(1, 5)],
        "explanation": "해설",
    }
    Q.enforce_answer_position(numeric, 4)
    check("수치 선택지는 그대로 두고 재생성에 맡김", numeric["correct_no"] == 1,
          numeric["correct_no"])

    # 이미 맞으면 아무것도 바꾸지 않는다
    same = {"question_text": "문제", "correct_no": 2,
            "choices": [{"no": i, "text": f"보기{i}"} for i in range(1, 5)],
            "explanation": "정답은 2번입니다."}
    Q.enforce_answer_position(same, 2)
    check("이미 맞으면 무변경", same["explanation"] == "정답은 2번입니다.")

    # 생성 경로에서도 4번 지시를 만족시키는가 (모델이 1번을 줘도)
    reset([_mc("4번 지시인데 1번을 준 응답", correct=1)])
    q = asyncio.run(G.generate_mc_question(USER, "PER", answer_pos=4))
    pos = next(c["choice_no"] for c in q["choices"] if c["is_correct"])
    check("생성 경로에서 자리 교정", pos == 4, pos)

    # 생성 경로 전체에서 4문제의 정답 위치가 전부 다른가
    # (correct=None -> 스텁이 프롬프트가 요구한 자리를 그대로 따른다)
    reset([_mc(f"문제{i}") for i in range(1, 5)])
    qs = asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=4))
    positions = [
        next(c["choice_no"] for c in q["choices"] if c["is_correct"]) for q in qs
    ]
    check("4문제 정답 위치가 모두 다름", sorted(positions) == [1, 2, 3, 4], positions)

    # 지시를 어긴 응답은 재생성되는가.
    # 두 번째 슬롯이 요구하는 자리를 구해, 그와 다른 값을 일부러 돌려준다.
    wanted = Q.answer_pos_for(1 + Q.answer_pos_offset("PER"))
    defiant = 1 if wanted != 1 else 2
    reset([_mc("첫 시도"),                          # 요구대로 - 통과
           _mc("두번째 - 지시 위반", correct=defiant),  # 다른 자리 -> 거부
           _mc("두번째 - 재생성")])                    # 요구대로 -> 통과
    qs = asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))
    pos2 = next(c["choice_no"] for c in qs[1]["choices"] if c["is_correct"])
    check("지시 위반 시 재생성", pos2 == wanted, f"{pos2} (요구 {wanted})")


# =========================================================
# 14. 병렬 생성 후 중복 해소
# =========================================================

def test_parallel_dedupe():
    """동시 생성이라 1단계에는 avoid 가 없다. 겹친 것만 2단계에서 되갚는지 본다.

    중복은 과거 문제 대조(BANK_SIMILARITY)로 유도한다. 그 경로는 기본이 꺼져
    있으므로(QUIZ_DUP_CHECK_PAST=False) 이 테스트에서만 켠다.
    """
    print("\n--- 병렬 생성 후 중복 해소 ---")
    global BANK_SIMILARITY

    reset()
    BANK_SIMILARITY = 0.99  # 과거 문제와 전부 중복 판정시킨다
    SETTINGS.QUIZ_DUP_CHECK_PAST = True
    try:
        asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=3))
    finally:
        BANK_SIMILARITY = None
        SETTINGS.QUIZ_DUP_CHECK_PAST = False

    calls = gen_calls()
    check("중복분이 재생성됨", len(calls) > 3, f"{len(calls)}회 (1단계 3 + 재생성)")

    # 1단계는 묶음 1회, 그 뒤가 전부 단건 재생성이다
    first_pass, retries = calls[:1], calls[1:]
    check("1단계는 묶음 1회", len(first_pass) == 1 and "slot_block" in first_pass[0],
          list(first_pass[0])[:4])
    # 거부된 자기 자신을 avoid 에 넣지 않으면 모델이 방금 만든 것을 그대로
    # 다시 내놓는다 (실측 유사도 1.000). 재생성 호출에는 반드시 채워져야 한다.
    check("재생성에는 avoid 가 채워짐",
          bool(retries) and all(c.get("avoid_block", "(아직 없음)") != "(아직 없음)"
                                for c in retries),
          [c.get("avoid_block") for c in retries])
    # 같은 관점으로 다시 만들면 같은 문제가 또 나온다(실측 유사도 0.979) - 관점을 민다.
    # 관점은 5주기라 count 만큼 밀면 결국 순환한다. 세트 전체와 겹치지 않게 할
    # 수는 없고, 요구되는 것은 "그 문제가 방금 쓴 관점"을 다시 쓰지 않는 것이다.
    batch_angles = angles_of(first_pass[0])
    check("재생성은 자기 슬롯의 1단계 관점과 다름",
          all(angles_of(r)[0] != batch_angles[i]
              for i, r in enumerate(retries) if i < len(batch_angles)),
          [(batch_angles[i][:12], angles_of(r)[0][:12])
           for i, r in enumerate(retries) if i < len(batch_angles)])

    # 중복이 없으면 재생성이 한 번도 일어나지 않아야 한다 (불필요한 LLM 호출 금지)
    reset()
    asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=3))
    check("중복 없으면 재생성 없음", len(gen_calls()) == 1, f"{len(gen_calls())}회 (묶음 1회)")


# =========================================================
# 15. 중복 검사 범위 (QUIZ_DUP_CHECK_PAST)
# =========================================================

def test_dup_scope():
    """과거 문제 대조는 설정으로 끈다. 세트 내 중복 검사는 끄지 않는다."""
    print("\n--- 중복 검사 범위 ---")
    global BANK_SIMILARITY

    # 과거 문제와 전부 겹치는 상황을 만들어 둔다.
    # 꺼져 있으면 이 신호를 아예 보지 않아야 한다.
    BANK_SIMILARITY = 0.99
    try:
        reset()
        SETTINGS.QUIZ_DUP_CHECK_PAST = False
        asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=3))
        off_calls = len(gen_calls())
        check("꺼짐: 과거와 겹쳐도 재생성 없음", off_calls == 1, f"{off_calls}회 (묶음 1회)")

        reset()
        SETTINGS.QUIZ_DUP_CHECK_PAST = True
        asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=3))
        on_calls = len(gen_calls())
        check("켜짐: 과거와 겹치면 재생성", on_calls > 3, f"{on_calls}회")
    finally:
        BANK_SIMILARITY = None
        SETTINGS.QUIZ_DUP_CHECK_PAST = False

    # 세트 내 중복은 설정과 무관하게 항상 막아야 한다.
    # 같은 문장을 3번 내놓게 하면 문자 유사도 단계에서 걸린다.
    same = "PER이 낮다는 것에 대한 설명으로 옳은 것은 무엇인가"
    reset([_mc(same), _mc(same), _mc(same)])
    asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=3))
    texts = [q for q in gen_calls()]
    check("꺼져 있어도 세트 내 중복은 재생성", len(texts) > 1, f"{len(texts)}회")

# =========================================================
# 16. 검증 모드 (QUIZ_VALIDATE_MODE)
# =========================================================

def test_validate_mode():
    """blocking / background / off 가 각각 다르게 동작하는지 본다."""
    print("\n--- 검증 모드 ---")
    import app.services.quiz.concurrency as C

    def validate_calls():
        return len([c for c in CALLS if c["kind"] == "validate"])

    # 불합격 1건이 나오는 상황을 만들어 둔다
    bad = {"results": [{"no": 1, "ok": False, "reason": "정답과 해설이 모순"}]}

    # blocking: 검증하고 재생성까지 한다
    reset(validate=bad)
    SETTINGS.QUIZ_VALIDATE_MODE = "blocking"
    asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))
    check("blocking: 검증 호출됨", validate_calls() == 1, validate_calls())
    check("blocking: 불합격분 재생성", len(gen_calls()) == 2,
          f"{len(gen_calls())}회 (묶음 1 + 재생성 1)")

    # off: 검증 자체를 안 한다
    reset(validate=bad)
    SETTINGS.QUIZ_VALIDATE_MODE = "off"
    asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))
    check("off: 검증 호출 없음", validate_calls() == 0, validate_calls())
    check("off: 재생성 없음", len(gen_calls()) == 1, f"{len(gen_calls())}회 (묶음 1회)")

    # background: 응답을 막지 않는다 (재생성 없음). 검증은 뒤에서 돈다.
    async def run_background():
        reset(validate=bad)
        SETTINGS.QUIZ_VALIDATE_MODE = "background"
        qs = await G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2)
        # 응답 시점에는 재생성이 없어야 한다
        during = len(gen_calls())
        # 뒤에 뜬 태스크가 끝날 때까지 잠깐 양보한다
        for _ in range(50):
            if not C._background:
                break
            await asyncio.sleep(0.01)
        return qs, during

    qs, during = asyncio.run(run_background())
    SETTINGS.QUIZ_VALIDATE_MODE = "blocking"
    check("background: 문제 개수 유지", len(qs) == 2, len(qs))
    check("background: 응답 시점에 재생성 없음", during == 1, f"{during}회 (묶음 1회)")
    check("background: 검증은 뒤에서 실행됨", validate_calls() == 1, validate_calls())
    check("background: 태스크 참조가 정리됨", len(C._background) == 0, len(C._background))


if __name__ == "__main__":
    test_prompts_carry_quality_rules()
    test_deterministic_checks()
    test_generate_batch()
    test_generate_from_prompt()
    test_count_override()
    test_question_type_enforcement()
    test_remap_prefers_user_level()
    test_angle_rotation()
    test_language_guard()
    test_topic_guard()
    test_validation_retraction()
    test_bank_integration()
    test_shared_module()
    test_answer_position()
    test_parallel_dedupe()
    test_dup_scope()
    test_validate_mode()

    print("\n" + "=" * 55)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("ALL PASS")
