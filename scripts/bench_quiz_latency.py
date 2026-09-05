"""퀴즈 생성 경로의 벽시계 지연 측정.

## 왜 필요한가

세 생성 경로가 LLM 을 여러 번 호출하는데, 그 호출들이 직렬인지 병렬인지에 따라
응답 시간이 배수로 갈린다. 그런데 지금까지 이 프로젝트에는 **엔드포인트 단위
지연을 재는 도구가 없었다.** `loadtest_llm.py` 는 vLLM 을 직접 때리므로 앱
파이프라인(분석 -> 생성 -> 중복 검사 -> 검증)의 누적 지연을 보지 못하고,
`compare_models.py` 는 품질 비교용이라 통과/불통과만 본다.

그 결과 "count=1 은 52초" 같은 숫자만 남고, `/quiz/generate/prompt` 가 실제로
몇 초인지는 베타 백엔드 연동에서 타임아웃이 나고 나서야 드러났다.

## 무엇을 재는가

라우터를 거치지 않고 서비스 함수를 직접 호출한다. HTTP/프록시 왕복을 빼고
**LLM 호출 누적 지연만** 남기기 위해서다. 프록시까지 포함한 실측이 필요하면
파드를 띄우고 `--http <BASE_URL>` 로 실제 엔드포인트를 때린다.

## 쓰는 법

    # 로컬 개발 구성 그대로 (.env = OpenRouter)
    uv run python scripts/bench_quiz_latency.py

    # 특정 경로만, 반복 측정
    uv run python scripts/bench_quiz_latency.py --only prompt --rounds 3

    # 파드에 뜬 실제 서버를 HTTP 로 (프록시 지연 포함)
    uv run python scripts/bench_quiz_latency.py --http https://<POD_ID>-8080.proxy.runpod.net

병렬화 전후를 비교하려면 `git stash` 로 코드를 되돌리고 같은 명령을 돌린다.

주의: `QUIZ_BANK_ENABLED=true` 면 Qdrant 가 떠 있어야 한다(`docker start qdrant`).
없으면 `--no-bank` 로 끄고 잰다 - 중복 검사 왕복이 빠지므로 수치가 조금 낙관적이다.
"""
import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# PowerShell 파이프는 CP949 로 디코드해 한글을 깨뜨린다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _parse_args():
    p = argparse.ArgumentParser(description="퀴즈 생성 지연 측정")
    p.add_argument("--only", choices=["batch", "prompt", "review", "all"], default="all")
    p.add_argument("--rounds", type=int, default=1, help="경로별 반복 횟수")
    p.add_argument("--count", type=int, default=5, help="batch 경로의 문제 수")
    p.add_argument("--http", default="", help="지정하면 HTTP 엔드포인트를 때린다")
    p.add_argument("--api-key", default=os.environ.get("AI_API_KEY", ""))
    p.add_argument("--no-bank", action="store_true", help="퀴즈 뱅크(Qdrant) 끄기")
    return p.parse_args()


ARGS = _parse_args()
if ARGS.no_bank:
    # pydantic-settings 는 셸 환경변수가 .env 보다 우선한다. import 전에 세팅해야 한다.
    os.environ["QUIZ_BANK_ENABLED"] = "false"

from app.core.config import settings          # noqa: E402
from app.schemas.chat import UserContext      # noqa: E402
from app.schemas.quiz import WrongAnswerItem  # noqa: E402

USER = UserContext(user_id=99, investment_level="초급")

PROMPT_TEXT = "PER이랑 분산투자에 대한 문제 5개 내줘"

WRONG_ANSWERS = [
    WrongAnswerItem(
        question_text="PER이 낮으면 항상 저평가된 주식이다.",
        question_type="OX", topic="PER", detail_code="PER_BASIC",
        selected_choice_no=1,
        choices=[{"choice_no": 1, "text": "O"}, {"choice_no": 2, "text": "X"}],
    ),
    WrongAnswerItem(
        question_text="분산투자의 주된 목적은 무엇인가?",
        question_type="MULTIPLE_CHOICE", topic="분산투자",
        detail_code="DIVERSIFICATION", selected_choice_no=2,
        choices=[
            {"choice_no": 1, "text": "위험을 줄이기 위해"},
            {"choice_no": 2, "text": "수수료를 아끼기 위해"},
            {"choice_no": 3, "text": "세금을 줄이기 위해"},
            {"choice_no": 4, "text": "거래량을 늘리기 위해"},
        ],
    ),
]


async def _run_local(kind: str) -> int:
    from app.services.quiz.generator import generate_quiz_batch, generate_quiz_from_prompt
    from app.services.quiz.review import generate_quiz_from_wrong_answers

    if kind == "batch":
        qs = await generate_quiz_batch(USER, "MULTIPLE_CHOICE", "PER", count=ARGS.count)
    elif kind == "prompt":
        qs = await generate_quiz_from_prompt(PROMPT_TEXT, USER)
    else:
        qs = await generate_quiz_from_wrong_answers(USER, WRONG_ANSWERS)
    return len(qs)


async def _run_http(kind: str) -> int:
    import httpx

    base = ARGS.http.rstrip("/")
    headers = {"X-API-Key": ARGS.api_key} if ARGS.api_key else {}
    user = USER.model_dump()

    # 카테고리 카탈로그는 요청 본문이 아니라 AI 서버에 내장돼 있다
    # (app/services/quiz/categories.py - DB 초기 데이터의 미러).
    if kind == "batch":
        path, body = "/quiz/generate", {
            "user": user, "quiz_type": "MULTIPLE_CHOICE",
            "topic": "PER", "count": ARGS.count,
        }
    elif kind == "prompt":
        path, body = "/quiz/generate/prompt", {"user": user, "prompt": PROMPT_TEXT}
    else:
        path, body = "/quiz/generate/review", {
            "user": user,
            "wrong_answers": [w.model_dump() for w in WRONG_ANSWERS],
        }

    # 측정이 목적이므로 타임아웃을 넉넉히 - 여기서 끊기면 무엇을 쟀는지 알 수 없다.
    async with httpx.AsyncClient(timeout=600) as client:
        r = await client.post(base + path, json=body, headers=headers)
        r.raise_for_status()
        return len(r.json().get("questions") or [])


async def measure(kind: str) -> None:
    label = {
        "batch": f"/quiz/generate (count={ARGS.count})",
        "prompt": "/quiz/generate/prompt",
        "review": "/quiz/generate/review",
    }[kind]

    runner = _run_http if ARGS.http else _run_local
    times: list[float] = []

    for i in range(ARGS.rounds):
        started = time.perf_counter()
        try:
            n = await runner(kind)
        except Exception as e:
            print(f"  {i + 1}회차  실패: {type(e).__name__}: {e}")
            continue
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        print(f"  {i + 1}회차  {elapsed:6.1f}초  (문제 {n}개)")

    if not times:
        print(f"{label:36s}  전부 실패")
        return

    avg = statistics.mean(times)
    # RunPod HTTP 프록시(Cloudflare)의 100초 선을 넘는지가 실무상 가장 중요하다.
    verdict = "OK" if max(times) < 100 else "!! 100초 초과 - 프록시에서 끊길 수 있음"
    print(f"{label:36s}  평균 {avg:6.1f}초  최대 {max(times):6.1f}초  {verdict}\n")


async def main() -> None:
    where = ARGS.http or f"로컬 함수 직접 호출 ({settings.LLM_BASE_URL})"
    print(f"대상      : {where}")
    print(f"모델      : {settings.LLM_MODEL}  (reasoning_effort={settings.LLM_REASONING_EFFORT or '미지정'})")
    print(f"동시 상한 : {getattr(settings, 'QUIZ_GEN_CONCURRENCY', '없음 (순차 생성)')}")
    print(f"퀴즈 뱅크 : {'켬' if settings.QUIZ_BANK_ENABLED else '끔'}\n")

    kinds = ["batch", "prompt", "review"] if ARGS.only == "all" else [ARGS.only]
    for kind in kinds:
        await measure(kind)


if __name__ == "__main__":
    asyncio.run(main())
