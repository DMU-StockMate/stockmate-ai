"""앱 엔드포인트 동시 부하 측정 (NestJS 백엔드 호출을 흉내낸다).

## 왜 필요한가

`loadtest_llm.py` 는 vLLM 을 직접 때린다. 앱 파이프라인(분석 → 병렬 생성 →
중복 검사 → 검증)을 거치지 않으므로 사용자가 실제로 겪는 지연을 못 본다.
`bench_quiz_latency.py` 는 앱을 재지만 요청을 하나씩 보낸다.

**둘 다 "동시 사용자 20명"을 재지 못한다.** 그런데 2026-09-05 병렬화로
동시성 프로파일이 바뀌었다.

    이전: 사용자 20명 × 요청당 1건 in-flight = 동시 20건   <- 리허설이 검증한 것
    이후: 사용자 20명 × 요청당 최대 5건     = 동시 최대 100건

KV 캐시 여유가 16k 기준 49개이므로 100건이면 vLLM 이 큐잉한다. GPU 총 작업량은
같아서 처리량은 안 변하지만, 리허설의 "동시 20명 p95 41.5초"는 더 이상 이
구성의 측정치가 아니다. 이 스크립트가 그 자리를 메운다.

## 쓰는 법

    uv run python scripts/loadtest_api.py `
      --base-url https://<POD_ID>-8080.proxy.runpod.net `
      --api-key "<AI_API_KEY>" --users 20

    # 프롬프트 기반(문제 5개, 가장 무거운 경로)으로
    uv run python scripts/loadtest_api.py --base-url ... --api-key ... --endpoint prompt

## 합격선 (베타 기준)

- 성공 20/20, HTTP 5xx·타임아웃 0
- **최대 지연 100초 미만** — RunPod HTTP 프록시(Cloudflare) 한계가 아직
  검증되지 않았다. 런북 8-2 는 "제한 없음"으로 기록했지만 근거가 count=4 의
  98.9초 성공이라, 100 초를 넘겨서 통과한 게 아니라 안 넘겨서 통과한 것이다.
- NestJS 는 타임아웃을 180초 이상으로 잡을 것 (런북 8-3).
"""
import argparse
import asyncio
import statistics
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _args():
    p = argparse.ArgumentParser(description="앱 엔드포인트 동시 부하 측정")
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", default="")
    p.add_argument("--users", type=int, default=20, help="동시 사용자 수")
    p.add_argument("--rounds", type=int, default=1, help="라운드 수 (prefix caching 확인용)")
    p.add_argument("--endpoint", choices=["generate", "prompt", "review"], default="generate")
    p.add_argument("--count", type=int, default=1, help="generate 경로의 문제 수")
    p.add_argument("--timeout", type=float, default=300.0)
    return p.parse_args()


A = _args()

# 사용자마다 다른 주제를 줘서 뱅크/프롬프트 캐시가 결과를 왜곡하지 않게 한다.
TOPICS = [
    "PER", "PBR", "ROE", "EPS", "배당수익률", "시가총액", "코스피", "코스닥",
    "ETF", "공매도", "분산투자", "손절매", "익절매", "포트폴리오", "리스크 관리",
    "재무제표", "영업이익", "당기순이익", "부채비율", "유상증자",
]


def _payload(i: int) -> tuple[str, dict]:
    topic = TOPICS[i % len(TOPICS)]
    user = {"user_id": 1000 + i, "investment_level": "초급"}

    if A.endpoint == "generate":
        return "/quiz/generate", {
            "user": user, "quiz_type": "MULTIPLE_CHOICE",
            "topic": topic, "count": A.count,
        }
    if A.endpoint == "prompt":
        return "/quiz/generate/prompt", {
            "user": user, "prompt": f"{topic}에 대한 문제 5개 내줘",
        }
    return "/quiz/generate/review", {
        "user": user,
        "wrong_answers": [{
            "question_text": f"{topic}에 대한 설명으로 옳은 것은?",
            "question_type": "MULTIPLE_CHOICE", "topic": topic,
            "selected_choice_no": 2,
            "choices": [
                {"choice_no": 1, "text": "올바른 설명"},
                {"choice_no": 2, "text": "틀린 설명"},
                {"choice_no": 3, "text": "관계없는 설명"},
                {"choice_no": 4, "text": "반대 설명"},
            ],
        }],
    }


async def _one(client, i: int) -> dict:
    path, body = _payload(i)
    headers = {"X-API-Key": A.api_key} if A.api_key else {}
    started = time.perf_counter()
    try:
        r = await client.post(A.base_url.rstrip("/") + path, json=body, headers=headers)
        elapsed = time.perf_counter() - started
        if r.status_code != 200:
            # 본문을 잘라 남긴다 - 500 의 detail 이 원인 파악의 전부다
            return {"ok": False, "elapsed": elapsed, "status": r.status_code,
                    "error": r.text[:160]}
        n = len((r.json() or {}).get("questions") or [])
        return {"ok": True, "elapsed": elapsed, "status": 200, "questions": n}
    except Exception as e:
        return {"ok": False, "elapsed": time.perf_counter() - started,
                "status": 0, "error": f"{type(e).__name__}: {e}"[:160]}


async def main() -> None:
    import httpx

    print(f"대상      : {A.base_url}")
    print(f"엔드포인트: {A.endpoint}" + (f" (count={A.count})" if A.endpoint == "generate" else ""))
    print(f"동시 사용자: {A.users}   라운드: {A.rounds}\n")

    limits = httpx.Limits(max_connections=A.users + 5, max_keepalive_connections=A.users + 5)
    async with httpx.AsyncClient(timeout=A.timeout, limits=limits) as client:
        for rnd in range(1, A.rounds + 1):
            started = time.perf_counter()
            results = await asyncio.gather(*(_one(client, i) for i in range(A.users)))
            wall = time.perf_counter() - started

            ok = [r for r in results if r["ok"]]
            bad = [r for r in results if not r["ok"]]
            times = sorted(r["elapsed"] for r in ok)

            print(f"[라운드 {rnd}] 성공 {len(ok)}/{A.users}   전체 {wall:.1f}초")
            if times:
                p50 = statistics.median(times)
                p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
                verdict = "OK" if times[-1] < 100 else "!! 100초 초과 - 프록시에서 끊길 수 있음"
                print(f"          p50 {p50:.1f}초  p95 {p95:.1f}초  최대 {times[-1]:.1f}초  {verdict}")
            for r in bad:
                print(f"          실패 status={r['status']}  {r.get('error')}")
            print()

    if A.rounds > 1:
        print("라운드 2가 라운드 1보다 빠르면 prefix caching 이 도는 것이다.")


if __name__ == "__main__":
    asyncio.run(main())
