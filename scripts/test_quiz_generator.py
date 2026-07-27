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


class _Resp:
    def __init__(self, content):
        self.content = content


def _mc(question, explanation="정상적인 해설 문장입니다.", correct=2, choices=None):
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
            obj.is_ox = "OX 문제를 1개 생성" in template
            return obj

        def __or__(self, other):
            return self

        async def ainvoke(self, kwargs):
            CALLS.append({"kind": self.kind, **kwargs})
            if self.kind == "validate":
                return _Resp(json.dumps(VALIDATE_RESULT, ensure_ascii=False))
            if self.kind == "analyze":
                return _Resp(json.dumps(ANALYZE_RESULT, ensure_ascii=False))
            if self.kind == "map":
                return _Resp(json.dumps({"mappings": []}, ensure_ascii=False))
            if RESPONSES:
                return _Resp(RESPONSES.pop(0))
            n = len([c for c in CALLS if c["kind"] == "generate"])
            text = _FALLBACK[(n - 1) % len(_FALLBACK)]
            if self.is_ox:
                # OX는 평서문이어야 하므로 폴백 문장을 평서문으로 바꿔 쓴다
                return _Resp(_ox(text.rstrip("?는은") + " 라고 볼 수 있다."))
            return _Resp(_mc(text))

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
        OLLAMA_BASE_URL = "http://localhost:11434"
        LLM_MODEL = "qwen3.5:9b"
        QUIZ_BANK_COLLECTION = "quiz_bank_test"
        QUIZ_BANK_ENABLED = True
        QUIZ_DUP_THRESHOLD = 0.90


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
    CALLS.clear()
    RESPONSES.clear()
    if responses:
        RESPONSES.extend(responses)
    global ANALYZE_RESULT, VALIDATE_RESULT
    ANALYZE_RESULT = analyze or {}
    VALIDATE_RESULT = validate if validate is not None else {"results": []}


def gen_calls():
    return [c for c in CALLS if c["kind"] == "generate"]


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
    check("생성 3회", len(gen_calls()) == 3, len(gen_calls()))

    # 검증 불합격 -> 재생성
    reset(validate={"results": [{"no": 2, "ok": False, "reason": "정답/해설 불일치"}]})
    qs = asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=3))
    check("불합격 시 재생성", len(gen_calls()) == 4, len(gen_calls()))
    check("개수는 그대로", len(qs) == 3)

    # 근사 중복 -> 재생성
    reset([_mc("PER이 낮으면 저평가라고 단정할 수 있는가"),
           _mc("PER이 낮으면 저평가라고 단정할 수 있는가"),
           _mc("배당수익률 계산의 분모는 무엇인가"),
           _mc("영업이익률이 의미하는 바로 옳은 것은")])
    qs = asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))
    check("근사 중복 재생성", len(gen_calls()) == 3, len(gen_calls()))
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
    angles = [c["angle"] for c in gen_calls()]
    check("배치 4개 관점 모두 다름", len(set(angles)) == 4, len(set(angles)))
    check("관점 순서가 정의 순서대로", angles == Q.GENERATION_ANGLES[:4])

    avoids = [c["avoid_block"] for c in gen_calls()]
    check("첫 호출 avoid 비어있음", avoids[0] == "(아직 없음)", avoids[0])
    check("avoid 누적", avoids[3].count("- ") == 3, avoids[3].count("- "))

    # 프롬프트 기반: 주제별로 독립 순환
    plan = {"relevant": True, "count": 5, "items": [
        {"topic": t, "question_type": "MULTIPLE_CHOICE",
         "category_code": "BEGINNER_FINANCIAL_BASIC", "detail_codes": ["PER_BASIC"]}
        for t in ["PER", "배당", "PER", "배당", "PER"]
    ]}
    reset(analyze=plan)
    asyncio.run(G.generate_quiz_from_prompt("PER이랑 배당 문제 내줘", USER))
    per = [c["angle"] for c in gen_calls() if c["topic"] == "PER"]
    div = [c["angle"] for c in gen_calls() if c["topic"] == "배당"]
    check("PER 3개 관점 다름", len(set(per)) == 3, len(set(per)))
    check("배당 2개 관점 다름", len(set(div)) == 2, len(set(div)))
    check("주제별 독립 순환", per[0] == div[0] == Q.GENERATION_ANGLES[0])

    # 단건 생성은 관점 지정 없이도 동작
    reset()
    asyncio.run(G.generate_quiz(USER, "MULTIPLE_CHOICE", "PER"))
    check("단건 생성 기본 관점", gen_calls()[0]["angle"] == Q.GENERATION_ANGLES[0])

    # 재요청 시 과거 이력만큼 관점을 밀어야 1차와 안 겹친다
    global BANK_TOPIC_COUNT
    reset()
    BANK_TOPIC_COUNT = 2
    try:
        asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))
        angles = [c["angle"] for c in gen_calls()]
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
              gen_calls()[0]["angle"] == Q.GENERATION_ANGLES[0])
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
# 6-a. 검증 자기 번복 처리
# =========================================================

def test_validation_retraction():
    print("\n--- 검증 사유 자기 번복 ---")

    # 실측: ok=false 인데 사유가 "사실 모든 것이 맞다"로 끝남 -> 재생성이 낭비됨
    retracting = ("...논란 여지가 있는 것으로 보아 주의 필요하나, 실제 검증 시에는 "
                  "모든 항목이 맞다. 다시 읽어보니 문제가 없으므로 사실 모든 것이 맞다.")
    reset(validate={"results": [{"no": 1, "ok": False, "reason": retracting}]})
    asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))
    check("번복된 사유는 재생성 안 함", len(gen_calls()) == 2, len(gen_calls()))

    # 명확한 사유는 그대로 재생성
    reset(validate={"results": [{"no": 1, "ok": False,
                                 "reason": "오답 3번이 사실이라 정답이 두 개"}]})
    asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))
    check("명확한 사유는 재생성", len(gen_calls()) == 3, len(gen_calls()))

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

    # 과거 문제와 유사 -> 재생성 유발
    reset()
    BANK_SIMILARITY = 0.95
    BANK_UPSERTS.clear()
    try:
        asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))
        check("과거 중복 감지 시 재생성", len(gen_calls()) == 4, len(gen_calls()))
    finally:
        BANK_SIMILARITY = None

    # 중복 없으면 재생성 없음
    reset()
    BANK_UPSERTS.clear()
    asyncio.run(G.generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=2))
    check("중복 없으면 재생성 없음", len(gen_calls()) == 2, len(gen_calls()))
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


if __name__ == "__main__":
    test_prompts_carry_quality_rules()
    test_deterministic_checks()
    test_generate_batch()
    test_generate_from_prompt()
    test_question_type_enforcement()
    test_remap_prefers_user_level()
    test_angle_rotation()
    test_language_guard()
    test_validation_retraction()
    test_bank_integration()
    test_shared_module()

    print("\n" + "=" * 55)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("ALL PASS")
