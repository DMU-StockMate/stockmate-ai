"""퀴즈 뱅크(app/services/quiz/bank.py) 회귀 테스트.

요청 간 중복 방지가 목적인 모듈이라, 확인해야 할 것은 세 가지다.
1. 유사도 임계값 판정이 맞는가
2. 소유권 필터가 남의 개인 문제를 끌어오지 않는가
3. **Qdrant가 죽어도 문제 생성이 계속되는가 (fail-open)** — 가장 중요

Qdrant와 임베딩은 스텁으로 대체하므로 실제 서버 없이 실행된다.

실행:
    uv run python scripts/test_quiz_bank.py
"""
import asyncio
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# =========================================================
# 스텁
# =========================================================

SEARCH_SCORE = None       # 검색이 돌려줄 유사도 (None이면 결과 없음)
SEARCH_RAISES = False     # True면 검색이 예외를 던진다 (Qdrant 장애 시뮬레이션)
UPSERT_RAISES = False
SEARCH_CALLS: list = []
UPSERT_CALLS: list = []
EMBED_RAISES = False
TOPIC_COUNT = 0        # count()가 돌려줄 주제별 기존 문제 수
COUNT_CALLS: list = []


def _install_stubs():
    config = types.ModuleType("app.core.config")

    class _Settings:
        QUIZ_BANK_COLLECTION = "quiz_bank_test"
        QUIZ_BANK_ENABLED = True
        QUIZ_DUP_THRESHOLD = 0.90

    config.settings = _Settings()
    sys.modules["app.core.config"] = config

    vs = types.ModuleType("app.services.rag.vectorstore")

    class _Emb:
        def embed_query(self, text):
            if EMBED_RAISES:
                raise RuntimeError("임베딩 모델 로드 실패")
            return [0.1] * 1024

        def embed_documents(self, texts):
            if EMBED_RAISES:
                raise RuntimeError("임베딩 모델 로드 실패")
            return [[0.1] * 1024 for _ in texts]

    class _Hit:
        def __init__(self, score):
            self.score = score

    class _Client:
        def get_collections(self):
            return types.SimpleNamespace(collections=[])

        def create_collection(self, **kw):
            pass

        def create_payload_index(self, **kw):
            pass

        def query_points(self, **kw):
            SEARCH_CALLS.append(kw)
            if SEARCH_RAISES:
                raise ConnectionError("Qdrant 연결 실패")
            points = [_Hit(SEARCH_SCORE)] if SEARCH_SCORE is not None else []
            return types.SimpleNamespace(points=points)

        def count(self, **kw):
            COUNT_CALLS.append(kw)
            return types.SimpleNamespace(count=TOPIC_COUNT)

        def upsert(self, **kw):
            UPSERT_CALLS.append(kw)
            if UPSERT_RAISES:
                raise ConnectionError("Qdrant 연결 실패")

    vs.get_embeddings = lambda: _Emb()
    vs.get_qdrant_client = lambda: _Client()
    sys.modules["app.services.rag.vectorstore"] = vs


_install_stubs()

from app.services.quiz import bank as B  # noqa: E402

FAILURES: list[str] = []


def check(name, passed, extra=""):
    print(("PASS  " if passed else "FAIL  ") + name + (f"  {extra}" if extra else ""))
    if not passed:
        FAILURES.append(name)


def reset(score=None, search_raises=False, upsert_raises=False, embed_raises=False):
    global SEARCH_SCORE, SEARCH_RAISES, UPSERT_RAISES, EMBED_RAISES
    SEARCH_SCORE = score
    SEARCH_RAISES = search_raises
    UPSERT_RAISES = upsert_raises
    EMBED_RAISES = embed_raises
    SEARCH_CALLS.clear()
    UPSERT_CALLS.clear()
    COUNT_CALLS.clear()
    B._collection_ready = False


def q(text, qtype="MULTIPLE_CHOICE", detail="PER_BASIC"):
    return {"question_text": text, "question_type": qtype,
            "topic": "PER", "primary_detail_code": detail}


# =========================================================
# 1. 유사도 임계값 판정
# =========================================================

def test_threshold():
    print("\n--- 유사도 임계값 ---")

    reset(score=0.95)
    check("임계값 초과 -> 중복",
          asyncio.run(B.is_duplicate_of_past("PER 계산식은 무엇인가", 1)))

    reset(score=0.90)
    check("임계값 정확히 일치 -> 중복",
          asyncio.run(B.is_duplicate_of_past("PER 계산식은 무엇인가", 1)))

    reset(score=0.89)
    check("임계값 미만 -> 통과",
          not asyncio.run(B.is_duplicate_of_past("PER 계산식은 무엇인가", 1)))

    reset(score=None)
    check("기존 문제 없음 -> 통과",
          not asyncio.run(B.is_duplicate_of_past("PER 계산식은 무엇인가", 1)))


# =========================================================
# 2. 소유권 필터
# =========================================================

def test_owner_filter():
    print("\n--- 소유권 필터 ---")

    reset(score=0.5)
    asyncio.run(B.is_duplicate_of_past("문제", 7))
    cond = SEARCH_CALLS[0]["query_filter"].must[0]
    owners = cond.match.any
    check("공개 + 본인 문제만 대상", set(owners) == {B.PUBLIC_OWNER, 7}, owners)
    check("남의 개인 문제는 제외", 8 not in owners)

    reset(score=0.5)
    asyncio.run(B.is_duplicate_of_past("문제", None))
    owners = SEARCH_CALLS[0]["query_filter"].must[0].match.any
    check("user_id 없으면 공개만", owners == [B.PUBLIC_OWNER], owners)

    # 저장 시 소유자 기록
    reset()
    asyncio.run(B.register([q("문제1")], user_id=7))
    payload = UPSERT_CALLS[0]["points"][0].payload
    check("저장 시 owner 기록", payload["owner_user_id"] == 7, payload["owner_user_id"])
    check("메타데이터 보존",
          payload["detail_code"] == "PER_BASIC" and payload["question_type"] == "MULTIPLE_CHOICE")

    reset()
    asyncio.run(B.register([q("문제1")], user_id=None))
    check("user_id 없으면 공개로 저장",
          UPSERT_CALLS[0]["points"][0].payload["owner_user_id"] == B.PUBLIC_OWNER)


# =========================================================
# 3. fail-open (가장 중요)
# =========================================================

def test_fail_open():
    print("\n--- fail-open (Qdrant 장애 시 생성 계속) ---")

    reset(search_raises=True)
    check("검색 실패해도 중복 아님으로 통과",
          not asyncio.run(B.is_duplicate_of_past("문제", 1)))

    reset(embed_raises=True)
    check("임베딩 실패해도 통과",
          not asyncio.run(B.is_duplicate_of_past("문제", 1)))

    reset(upsert_raises=True)
    try:
        asyncio.run(B.register([q("문제1")], 1))
        check("저장 실패해도 예외 없음", True)
    except Exception as e:
        check("저장 실패해도 예외 없음", False, str(e))

    reset(embed_raises=True)
    try:
        asyncio.run(B.register([q("문제1")], 1))
        check("저장 시 임베딩 실패해도 예외 없음", True)
    except Exception as e:
        check("저장 시 임베딩 실패해도 예외 없음", False, str(e))


# =========================================================
# 4. 비활성화 스위치 / 빈 입력
# =========================================================

def test_disabled_and_empty():
    print("\n--- 비활성화 / 빈 입력 ---")

    from app.core.config import settings
    reset(score=0.99)
    settings.QUIZ_BANK_ENABLED = False
    try:
        check("비활성화 시 항상 통과",
              not asyncio.run(B.is_duplicate_of_past("문제", 1)))
        check("비활성화 시 검색 호출 안 함", len(SEARCH_CALLS) == 0, len(SEARCH_CALLS))
        asyncio.run(B.register([q("문제1")], 1))
        check("비활성화 시 저장 안 함", len(UPSERT_CALLS) == 0, len(UPSERT_CALLS))
    finally:
        settings.QUIZ_BANK_ENABLED = True

    reset()
    asyncio.run(B.register([], 1))
    check("빈 목록 저장 시 호출 없음", len(UPSERT_CALLS) == 0)


# =========================================================
# 4-b. 관점 오프셋
# =========================================================

def test_angle_offset():
    print("\n--- 관점 오프셋 ---")
    global TOPIC_COUNT

    reset()
    TOPIC_COUNT = 0
    check("이력 없으면 0", asyncio.run(B.angle_offset("PER", 1)) == 0)

    TOPIC_COUNT = 3
    reset()
    TOPIC_COUNT = 3
    check("3개 있으면 3", asyncio.run(B.angle_offset("PER", 1)) == 3)
    flt = COUNT_CALLS[0]["count_filter"]
    keys = [c.key for c in flt.must]
    check("소유자 + 주제로 필터", set(keys) == {"owner_user_id", "topic"}, keys)

    reset(search_raises=False)
    TOPIC_COUNT = 0
    check("topic 없으면 0", asyncio.run(B.angle_offset("", 1)) == 0)
    check("topic 없으면 조회 안 함", len(COUNT_CALLS) == 0)

    # fail-open
    from app.core.config import settings
    reset()
    settings.QUIZ_BANK_ENABLED = False
    try:
        check("비활성화 시 0", asyncio.run(B.angle_offset("PER", 1)) == 0)
    finally:
        settings.QUIZ_BANK_ENABLED = True
    TOPIC_COUNT = 0


# =========================================================
# 5. 스텁이 실제 qdrant-client API와 맞는가
# =========================================================

def test_stub_matches_real_client():
    """스텁이 실제로는 없는 메서드를 흉내내고 있지 않은지 검사한다.

    실측 사고: qdrant-client 1.18에서 client.search() 가 제거됐는데
    스텁은 search() 를 갖고 있어서 테스트는 통과하고 운영에서만 터졌다
    ('QdrantClient' object has no attribute 'search').
    스텁과 실물의 API 표면을 대조해 같은 유형의 사고를 막는다.
    """
    print("\n--- 스텁 / 실제 클라이언트 API 일치 ---")
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        check("qdrant-client 미설치 - 검사 생략", True)
        return

    for method in ("query_points", "upsert", "count", "get_collections",
                   "create_collection", "create_payload_index"):
        check(f"실제 클라이언트에 {method}() 존재", hasattr(QdrantClient, method))

    # bank.py가 제거된 구버전 API를 실제로 호출하고 있지 않은지.
    # 주석/독스트링에 언급하는 것은 허용하고 호출 형태만 본다.
    source = (ROOT / "app" / "services" / "quiz" / "bank.py").read_text(encoding="utf-8")
    check("제거된 client.search() 호출 없음",
          "get_qdrant_client().search(" not in source)
    check("query_points() 사용", "get_qdrant_client().query_points(" in source)


# =========================================================
# 6. 저장 동작
# =========================================================

def test_register():
    print("\n--- 저장 ---")

    reset()
    asyncio.run(B.register([q("문제1"), q("문제2"), q("문제3")], 1))
    points = UPSERT_CALLS[0]["points"]
    check("3개 한 번에 저장", len(points) == 3, len(points))
    check("포인트 ID가 서로 다름", len({p.id for p in points}) == 3)
    check("벡터 차원 1024", len(points[0].vector) == 1024, len(points[0].vector))
    check("본문 payload 저장", points[0].payload["question_text"] == "문제1")
    check("생성 시각 기록", "created_at" in points[0].payload)


if __name__ == "__main__":
    test_threshold()
    test_owner_filter()
    test_fail_open()
    test_disabled_and_empty()
    test_angle_offset()
    test_stub_matches_real_client()
    test_register()

    print("\n" + "=" * 55)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        sys.exit(1)
    print("ALL PASS")
