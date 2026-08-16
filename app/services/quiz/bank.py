"""생성된 퀴즈 문제 저장소 (Qdrant) — 요청 간 중복 방지.

세트 안에서의 중복은 quality.is_near_duplicate 가 문자 유사도로 막지만,
그건 한 번의 요청 안에서만 유효하다. 응답이 나가면 목록이 사라져서
같은 요청을 다시 하면 비슷한 문제가 또 나온다.

특히 출제 관점 로테이션이 매 요청 첫 번째 관점부터 시작하는 결정론적 순환이라
2차 요청이 1차와 같은 순서를 밟는다. 그래서 과거에 만든 문제를 기억해야 한다.

문자 유사도로는 "주가가 오르지 않고 배당금이 늘면" 과
"주가가 그대로인데 배당금이 증가하면" 을 구분하지 못한다(실측).
의미 중복을 잡으려면 임베딩 유사도가 필요하다.

설계 원칙
- **fail-open**: Qdrant가 죽어도 문제 생성은 계속돼야 한다. 중복 회피는 부가 기능이지
  이것 때문에 기능 전체가 실패하면 안 된다. 모든 예외는 로그만 남기고 통과시킨다.
- **RAG 컬렉션과 분리**: 뉴스/공시 청크와 섞으면 검색 품질이 떨어진다
  (프로젝트 원칙: 데이터 성격이 다르면 컬렉션을 나눈다).
- **논블로킹**: 임베딩(CPU 50~200ms)과 Qdrant 호출은 동기 API라
  asyncio.to_thread 로 감싸 이벤트 루프를 막지 않는다.

비용은 문제당 임베딩 1회 + 벡터 검색 1회로, LLM 호출(5~15초) 대비 무시할 수준이다.
"""
import asyncio
import uuid
from datetime import datetime, timezone

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.core.config import settings
from app.core.logger import setup_logger
from app.services.rag.vectorstore import get_embeddings, get_qdrant_client

logger = setup_logger(__name__)

# 공개 문제(특정 유저 소유가 아님)를 나타내는 owner_user_id 값.
# null 조건은 Qdrant 필터에서 다루기 번거로워 0을 센티넬로 쓴다.
PUBLIC_OWNER = 0

_collection_ready = False


def _ensure_collection() -> None:
    """컬렉션과 payload 인덱스를 준비한다 (최초 1회)."""
    global _collection_ready
    if _collection_ready:
        return

    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if settings.QUIZ_BANK_COLLECTION not in existing:
        client.create_collection(
            collection_name=settings.QUIZ_BANK_COLLECTION,
            # bge-m3 = 1024차원, 정규화된 임베딩이라 COSINE
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        logger.info(f"퀴즈 뱅크 컬렉션 생성: {settings.QUIZ_BANK_COLLECTION}")

    # owner_user_id 필터를 자주 쓰므로 payload 인덱스를 만든다
    # (이미 있으면 예외가 나므로 무시한다)
    for field, schema in (("owner_user_id", PayloadSchemaType.INTEGER),
                          ("topic", PayloadSchemaType.KEYWORD)):
        try:
            client.create_payload_index(
                collection_name=settings.QUIZ_BANK_COLLECTION,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            pass

    _collection_ready = True


def _owner_filter(user_id: int | None) -> Filter:
    """공개 문제 + 본인 소유 개인 문제만 대상으로 하는 필터.

    남의 개인 문제까지 중복 검사에 넣으면, 그 사람의 문제 때문에
    내 문제가 폐기되는 이상한 동작이 된다 (텍스트가 새어나가지는 않지만
    불필요한 재생성이 늘어난다).
    """
    owners = [PUBLIC_OWNER]
    if user_id:
        owners.append(user_id)
    return Filter(
        must=[FieldCondition(key="owner_user_id", match=MatchAny(any=owners))]
    )


def _search_similar(text: str, user_id: int | None) -> float:
    """가장 비슷한 기존 문제의 유사도를 반환한다 (동기, 스레드에서 실행).

    qdrant-client 1.18 에서 client.search() 가 제거되어 query_points() 를 쓴다
    (구버전 API를 쓰면 'QdrantClient' object has no attribute 'search' 로 실패).
    COSINE 거리라 score 가 클수록 유사하다.
    """
    _ensure_collection()
    vector = get_embeddings().embed_query(text)
    response = get_qdrant_client().query_points(
        collection_name=settings.QUIZ_BANK_COLLECTION,
        query=vector,
        query_filter=_owner_filter(user_id),
        limit=1,
        with_payload=False,
    )
    hits = response.points
    return hits[0].score if hits else 0.0


async def is_duplicate_of_past(text: str, user_id: int | None = None) -> bool:
    """과거에 만든 문제와 의미가 겹치는지 검사한다.

    Qdrant 장애나 미기동 시에는 False(중복 아님)를 반환해 생성을 계속한다.
    """
    if not settings.QUIZ_BANK_ENABLED:
        return False

    try:
        score = await asyncio.to_thread(_search_similar, text, user_id)
    except Exception as e:
        logger.warning(f"퀴즈 뱅크 검색 실패 - 중복 검사 생략: {e}")
        return False

    duplicate = score >= settings.QUIZ_DUP_THRESHOLD
    # 임계값을 넘든 안 넘든 항상 점수를 남긴다.
    # 넘을 때만 찍으면 "왜 중복이 안 걸렸는지" 알 수 없어 임계값 튜닝이 불가능하다.
    logger.info(
        f"뱅크 유사도 {score:.3f} (임계값 {settings.QUIZ_DUP_THRESHOLD}) "
        f"-> {'중복 폐기' if duplicate else '통과'}: {text[:35]}"
    )
    return duplicate


# 한 요청 안에서 같은 문장을 여러 번 임베딩하지 않도록 벡터를 재사용한다.
# 세트 중복 검사는 매 문항마다 이전 문항 전체와 비교하므로, 캐시가 없으면
# 임베딩 호출이 문항 수의 제곱으로 늘어난다 (bge-m3 는 CPU 라 1회 50~200ms).
_vec_cache: dict[str, list[float]] = {}
_VEC_CACHE_MAX = 256


def _vector_of(text: str) -> list[float]:
    vec = _vec_cache.get(text)
    if vec is None:
        vec = get_embeddings().embed_query(text)
        _vec_cache[text] = vec
        if len(_vec_cache) > _VEC_CACHE_MAX:
            _vec_cache.pop(next(iter(_vec_cache)))
    return vec


def _max_similarity_in_set(text: str, others: list[str]) -> float:
    """text 와 others 중 가장 비슷한 것의 코사인 유사도."""
    vec = _vector_of(text)
    best = 0.0
    for other in others:
        other_vec = _vector_of(other)
        # 임베딩이 normalize 되어 있어 내적이 곧 코사인 유사도다
        score = sum(a * b for a, b in zip(vec, other_vec))
        best = max(best, score)
    return best


async def max_similarity_in_set(text: str, others: list[str]) -> float:
    """text 가 others 중 가장 비슷한 것과 얼마나 겹치는지 (0.0 ~ 1.0).

    실패 시 0.0 을 반환해 생성을 막지 않는다.
    """
    if not others:
        return 0.0
    try:
        return await asyncio.to_thread(_max_similarity_in_set, text, others)
    except Exception as e:
        logger.warning(f"세트 내 유사도 계산 실패 - 생략: {e}")
        return 0.0


async def is_duplicate_in_set(text: str, others: list[str]) -> bool:
    """지금 만들고 있는 세트 안의 다른 문항과 의미가 겹치는지 검사한다.

    is_near_duplicate 는 문자 bigram 유사도라 표현이 다르면 못 잡는다.
    실측(qwen3.6-35b, 같은 주제 5문제):
      "ROE에 대한 설명으로 옳은 것은?" vs "ROE의 정의에 대한 설명으로 옳은 것은?"
      -> 문자 기준 통과, 임베딩 기준 0.979
      "주가가 상승했을 때 배당수익률의 변화는?" vs "주가가 하락했을 때 ...?"
      -> 임베딩 기준 0.854
    3개 주제 30쌍 중 5쌍이 임계값을 넘었는데 하나도 걸러지지 않았다.

    과거 문제 대조(is_duplicate_of_past)와 같은 임계값을 쓴다.
    같은 '중복'을 판정하는데 기준이 다르면 설명할 수 없다.
    """
    score = await max_similarity_in_set(text, others)
    duplicate = score >= settings.QUIZ_DUP_THRESHOLD
    logger.info(
        f"세트 내 유사도 {score:.3f} (임계값 {settings.QUIZ_DUP_THRESHOLD}) "
        f"-> {'중복 폐기' if duplicate else '통과'}: {text[:35]}"
    )
    return duplicate


def _count_for_topic(topic: str, user_id: int | None) -> int:
    """해당 주제로 이미 만든 문제 수 (동기, 스레드에서 실행)."""
    _ensure_collection()
    owner = _owner_filter(user_id)
    result = get_qdrant_client().count(
        collection_name=settings.QUIZ_BANK_COLLECTION,
        count_filter=Filter(
            must=owner.must + [FieldCondition(key="topic", match=MatchValue(value=topic))]
        ),
        exact=True,
    )
    return result.count


async def angle_offset(topic: str, user_id: int | None = None) -> int:
    """이 주제의 출제 관점을 어디서부터 시작할지 결정한다.

    관점 로테이션이 매 요청 0번부터 시작하면, 두 번째 요청도 똑같이
    정의 -> 오해 -> 비교 순서를 밟아 1차와 겹치는 문제가 나온다(실측:
    "PER 계산 시 분모는?" 과 "PER은 주가를 무엇으로 나누나?" 가 두 요청에 걸쳐 중복).

    이미 만든 문제 수만큼 관점을 밀어두면 2차 요청은 다음 관점부터 시작한다.
    임계값 튜닝으로는 풀 수 없는 종류의 중복이라 별도로 다룬다.

    실패 시 0을 반환한다(기존 동작과 동일).
    """
    if not settings.QUIZ_BANK_ENABLED or not topic:
        return 0
    try:
        return await asyncio.to_thread(_count_for_topic, topic, user_id)
    except Exception as e:
        logger.warning(f"관점 오프셋 조회 실패 - 0부터 시작: {e}")
        return 0


def _upsert(questions: list[dict], user_id: int | None) -> int:
    """문제들을 벡터로 저장한다 (동기, 스레드에서 실행)."""
    _ensure_collection()
    texts = [q["question_text"] for q in questions]
    vectors = get_embeddings().embed_documents(texts)

    now = datetime.now(timezone.utc).isoformat()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "owner_user_id": user_id or PUBLIC_OWNER,
                "question_text": q["question_text"],
                "question_type": q.get("question_type"),
                "topic": q.get("topic"),
                "detail_code": q.get("primary_detail_code"),
                "created_at": now,
            },
        )
        for q, vector in zip(questions, vectors)
    ]
    get_qdrant_client().upsert(
        collection_name=settings.QUIZ_BANK_COLLECTION, points=points,
    )
    return len(points)


async def register(questions: list[dict], user_id: int | None = None) -> None:
    """생성이 확정된 문제들을 저장한다.

    검증·재생성이 모두 끝난 뒤에 호출한다. 중간에 폐기된 문제까지 저장하면
    다음 요청에서 쓸데없이 재생성을 유발한다.

    NestJS가 실제로 DB에 저장했는지는 알 수 없으므로, 저장하지 않고 버린 문제도
    뱅크에는 남는다. 중복 회피 용도라 유령 데이터가 조금 있어도 무해하다고 보고
    별도 확정 왕복은 두지 않았다 (팀 결정).
    """
    if not settings.QUIZ_BANK_ENABLED or not questions:
        return

    try:
        count = await asyncio.to_thread(_upsert, questions, user_id)
        logger.info(f"퀴즈 뱅크 저장: {count}개 (owner={user_id or PUBLIC_OWNER})")
    except Exception as e:
        logger.warning(f"퀴즈 뱅크 저장 실패 - 무시하고 진행: {e}")
