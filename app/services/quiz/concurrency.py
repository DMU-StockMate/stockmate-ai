"""문제 생성 병렬 실행 + 중복 해소 공통 모듈.

## 왜 만들었나

세 생성 경로(`generate_quiz_batch` / `generate_quiz_from_prompt` /
`generate_quiz_from_wrong_answers`)가 문제를 하나씩 **순차로** 만들고 있었다.
근거는 코드 주석에 남아 있던 다음 문장이다.

    "로컬 Ollama가 단일 모델 인스턴스라 동시 요청을 병렬로 못 받는 경우가 많아
     순차 생성으로 처리한다 (팀 논의 결과 - 속도보단 안정성 우선)."

서빙이 vLLM 으로 바뀌면서 이 전제가 무너졌다. vLLM 은 연속 배칭(continuous
batching)이라 동시 요청이 오히려 정상 사용 형태다. 2026-08-30 리허설 실측:

    동시 20건 -> 20/20 성공, p50 = p95 = 15.0초, 집계 1,029 tok/s
    GPU KV cache 807,634 tokens = 16k 요청 기준 동시 49개 여유

5건 병렬은 부하 축에도 들지 않는다. 개발용 OpenRouter 도 당연히 병렬을 받는다.

## 순차 생성이 치르던 대가

`/quiz/generate/prompt` 는 LLM 을 최소 7번(분석 1 + 생성 5 + 검증 1) **직렬**로
호출한다. 호출당 20~26초라 합계가 2분을 넘었고, 두 군데서 동시에 끊겼다.

1. NestJS `HttpModule` 타임아웃 (런북 §8-3 이 180초를 요구하는 이유)
2. RunPod HTTP 프록시(Cloudflare). 런북 §8-2 는 이 제한을 "틀린 가설"로
   기록했지만, 근거가 `count=4` 의 **98.9초 성공**이었다. 100초를 넘겨서
   통과한 게 아니라 안 넘겨서 통과한 것이다. 100초 아래로 내리는 것은
   선택이 아니라 요구사항이다.

## 어떻게 바꾸나 — 2단계

1) **생성**: 전부 동시에 만든다. 출제 관점(angle)과 정답 위치(answer_pos)는
   인덱스만으로 결정되므로 앞 문제의 결과를 기다릴 이유가 애초에 없었다.
2) **중복 해소**: 나온 세트를 훑어 겹치는 것만 골라 다시 만든다. 이때
   `avoid` 목록이 채워지므로, 순차 생성이 주던 "이미 만든 문제를 피하라"는
   힌트가 그 자리에서 복원된다.

즉 1단계에서 avoid 힌트를 잃는 대신, **잃은 항목에 한해서만** 2단계에서
되갚는다. 전부가 아니라 걸린 것만 한 라운드 더 도는 구조다.

이 모듈에 `_is_duplicate` / `_less_duplicated` 를 모아둔 이유도 같다. 원래
generator.py 와 review.py 에 글자까지 똑같은 사본이 하나씩 있었는데, 한쪽만
고치면 조용히 갈라진다.
"""
import asyncio
from typing import Awaitable, Callable

from app.core.config import settings
from app.core.logger import setup_logger
from app.schemas.chat import UserContext
from app.services.quiz import bank
from app.services.quiz.quality import is_near_duplicate

logger = setup_logger(__name__)


# 문제 1개를 만드는 무인자 코루틴 팩토리.
# 호출부마다 생성 함수 시그니처가 달라(OX / 객관식 / 오답기반) 클로저로 감싸 넘긴다.
QuestionMaker = Callable[[], Awaitable[dict]]

# 중복으로 걸린 i번 문제를 avoid 목록을 받아 다시 만드는 함수.
DupRetry = Callable[[int, list[str]], Awaitable[dict]]

# 검증 불합격한 i번 문제를 같은 조건으로 다시 만드는 함수.
FailRetry = Callable[[int], Awaitable[dict]]


def _limit() -> int:
    return max(1, settings.QUIZ_GEN_CONCURRENCY)


async def generate_all(makers: list[QuestionMaker]) -> list[dict]:
    """문제들을 동시에 만든다. 결과 순서는 makers 순서 그대로다.

    하나라도 3회 재시도를 소진하면 ValueError 가 그대로 전파된다 —
    순차 생성일 때와 같은 동작이다(라우터가 500 으로 변환). 이때 나머지
    작업은 asyncio.gather 가 취소하므로 GPU 를 계속 태우지 않는다.

    동시 실행 수를 settings.QUIZ_GEN_CONCURRENCY 로 묶는다. vLLM 은 49개까지
    여유가 있지만 그건 **서버 전체** 기준이고, 동시 사용자 20명이 각자
    5개짜리 세트를 요청하면 100건이 된다. 요청 하나가 큐를 독점하지 않도록
    요청 단위로 상한을 둔다.
    """
    if not makers:
        return []
    if len(makers) == 1:
        # 세마포어와 gather 를 만들 이유가 없다 (count=1 이 가장 흔한 경로).
        return [await makers[0]()]

    sem = asyncio.Semaphore(_limit())

    async def run(maker: QuestionMaker) -> dict:
        async with sem:
            return await maker()

    loop = asyncio.get_running_loop()
    started = loop.time()
    results = await asyncio.gather(*(run(m) for m in makers))
    logger.info(
        f"문제 {len(makers)}개 병렬 생성 완료 "
        f"({loop.time() - started:.1f}s, 동시 상한 {_limit()})"
    )
    return results


async def is_duplicate(text: str, seen_texts: list[str], user: UserContext) -> bool:
    """세 단계로 본다: 세트 내 문자 유사도 -> 세트 내 의미 유사도 -> 과거 문제.

    비용이 싼 순서다. 문자 유사도는 0원, 세트 내 임베딩은 1회,
    과거 대조는 Qdrant 왕복까지 든다. 앞에서 걸리면 뒤는 건너뛴다.
    LLM 은 한 번도 부르지 않는다 - 그래서 이 단계는 순차로 둬도 싸다.

    3단계(과거 대조)는 settings.QUIZ_DUP_CHECK_PAST 로 끌 수 있고 기본이 꺼짐이다.
    끄면 이번 요청 안에서 만든 문제끼리만 겹치지 않게 한다.
    """
    if is_near_duplicate(text, seen_texts):
        logger.info(f"세트 내 중복 감지(문자) - 재생성: {text[:40]}")
        return True
    if await bank.is_duplicate_in_set(text, seen_texts):
        return True
    if not settings.QUIZ_DUP_CHECK_PAST:
        return False
    return await bank.is_duplicate_of_past(text, user.user_id)


async def less_duplicated(first: dict, retry: dict, seen_texts: list[str]) -> dict:
    """중복으로 걸린 원본과 재생성본 중 세트와 덜 겹치는 쪽을 고른다.

    재시도가 항상 나은 게 아니다. 실측(배당수익률 5문제): 0.846 으로 걸러서
    다시 만들었더니 0.920 이 나왔다. 재시도본을 무조건 채택하면 오히려 나빠진다.
    벡터는 캐시되어 있어 추가 임베딩 비용은 재생성본 1건뿐이다.
    """
    if not seen_texts:
        return retry
    first_score = await bank.max_similarity_in_set(first["question_text"], seen_texts)
    retry_score = await bank.max_similarity_in_set(retry["question_text"], seen_texts)
    if retry_score <= first_score:
        return retry
    logger.info(
        f"재시도가 더 겹쳐 원본 유지 (원본 {first_score:.3f} < 재시도 {retry_score:.3f})"
    )
    return first


async def resolve_duplicates(
    questions: list[dict],
    make_retry: DupRetry,
    user: UserContext,
) -> list[dict]:
    """병렬 생성 결과에서 중복만 골라 다시 만든다 (questions 를 in-place 수정).

    1) 앞에서부터 훑으며 이미 채택된 문제 / 과거 문제와 겹치는 인덱스를 모은다.
       LLM 을 쓰지 않으므로 순차로 돌아도 비용이 거의 없다.
    2) 걸린 인덱스만 **동시에** 재생성한다. avoid 에 "채택된 문제 전부 +
       거부된 자기 자신"을 넘긴다. 거부된 자신을 빼먹으면 모델이 방금 만든 것을
       그대로 다시 내놓는다 (실측 유사도 1.000).
    3) 원본과 재생성본 중 세트와 덜 겹치는 쪽을 고른다.

    재생성이 실패해도 원본을 유지한다 — 중복이더라도 문제 개수는 맞춰야 한다.
    """
    accepted: list[str] = []
    dup_indexes: list[int] = []

    for i, question in enumerate(questions):
        if await is_duplicate(question["question_text"], accepted, user):
            dup_indexes.append(i)
        else:
            accepted.append(question["question_text"])

    if not dup_indexes:
        return questions

    logger.info(f"중복 {len(dup_indexes)}건 재생성: {[i + 1 for i in dup_indexes]}번")

    sem = asyncio.Semaphore(_limit())

    async def retry(i: int) -> dict | None:
        async with sem:
            try:
                return await make_retry(i, accepted + [questions[i]["question_text"]])
            except ValueError as e:
                logger.warning(f"{i + 1}번 중복 재생성 실패 - 원본 유지: {e}")
                return None

    retried = await asyncio.gather(*(retry(i) for i in dup_indexes))

    for i, candidate in zip(dup_indexes, retried):
        if candidate is None:
            continue
        questions[i] = await less_duplicated(questions[i], candidate, accepted)

    return questions


async def regenerate_failed(
    questions: list[dict],
    failed: dict[int, str],
    make_retry: FailRetry,
    carry_fields: tuple[str, ...] = ("category_code", "detail_codes", "primary_detail_code"),
) -> None:
    """검증 불합격 문제들을 동시에 재생성한다 (questions 를 in-place 수정).

    재생성본은 다시 검증하지 않는다 (검증 호출이 계속 늘어나는 것을 막기 위한
    타협). 재생성에 실패하면 원본을 그대로 둔다 - 문제 개수는 항상 맞춰야 한다.

    `carry_fields` 는 카테고리 매핑처럼 **생성 이후에 붙은** 필드다. 재생성본에는
    없으므로 원본 것을 승계한다.
    """
    if not failed:
        return

    indexes = list(failed.keys())
    sem = asyncio.Semaphore(_limit())

    async def fix(i: int) -> dict | None:
        async with sem:
            try:
                return await make_retry(i)
            except ValueError as e:
                logger.warning(f"{i + 1}번 재생성 실패 - 원본 유지: {e}")
                return None

    fixed_list = await asyncio.gather(*(fix(i) for i in indexes))

    for i, fixed in zip(indexes, fixed_list):
        if fixed is None:
            continue
        for field in carry_fields:
            if field in questions[i]:
                fixed[field] = questions[i][field]
        questions[i] = fixed
