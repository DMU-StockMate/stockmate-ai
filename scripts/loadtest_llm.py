"""vLLM 서버 동시 부하 테스트 (베타 예상 규모 20명).

RunPod H100 리허설용. FastAPI 를 거치지 않고 vLLM 의 OpenAI 호환
엔드포인트를 직접 때려서 서버 자체의 배칭 성능을 잰다.

사용:
    uv run python scripts/loadtest_llm.py \
        --base-url https://<POD_ID>-8000.proxy.runpod.net/v1 \
        --model Qwen/Qwen3.8-27B-FP8 \
        --api-key <서버에 준 --api-key> \
        --concurrency 20 --rounds 2

주의: RunPod HTTP 프록시는 Cloudflare 100초 타임아웃이 걸린다.
      --timeout 을 100 이상으로 줘도 프록시가 먼저 끊는다.
      100초를 넘기면 TCP 포트 직결로 다시 재볼 것.
"""
import argparse
import asyncio
import statistics
import time

import httpx

PROMPTS = [
    "PER과 PBR의 차이를 초급 학습자에게 설명해줘.",
    "ROE가 높은 기업이 항상 좋은 투자처인지 판단 근거와 함께 설명해줘.",
    "영업이익과 당기순이익의 차이를 예시 숫자와 함께 설명해줘.",
    "시가총액이 무엇인지, 어떻게 계산하는지 설명해줘.",
    "부채비율이 200%인 기업을 어떻게 해석해야 하는지 설명해줘.",
]


async def one_call(client, args, idx, results):
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "너는 한국어로만 답하는 주식 학습 도우미다."},
            {"role": "user", "content": PROMPTS[idx % len(PROMPTS)]},
        ],
        "temperature": 0.7,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"reasoning_effort": args.reasoning_effort},
    }
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    t0 = time.perf_counter()
    try:
        r = await client.post("/chat/completions", json=body, headers=headers)
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            results.append((idx, dt, None, f"HTTP {r.status_code}: {r.text[:200]}"))
            return
        data = r.json()
        out_tok = data.get("usage", {}).get("completion_tokens", 0)
        results.append((idx, dt, out_tok, None))
    except Exception as e:  # noqa: BLE001
        results.append((idx, time.perf_counter() - t0, None, f"{type(e).__name__}: {e}"))


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True, help=".../v1 까지")
    p.add_argument("--model", default="Qwen/Qwen3.8-27B-FP8")
    p.add_argument("--api-key", default="")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "xhigh"])
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args()

    limits = httpx.Limits(max_connections=args.concurrency + 5)
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"), timeout=args.timeout, limits=limits
    ) as client:
        # 워밍업 1회 (첫 요청은 그래프 컴파일/캐시 준비로 느리다)
        warm: list = []
        await one_call(client, args, 0, warm)
        print(f"워밍업: {warm[0][1]:.1f}s  err={warm[0][3]}")

        for rnd in range(1, args.rounds + 1):
            results: list = []
            t0 = time.perf_counter()
            await asyncio.gather(
                *(one_call(client, args, i, results) for i in range(args.concurrency))
            )
            wall = time.perf_counter() - t0

            ok = [r for r in results if r[3] is None]
            bad = [r for r in results if r[3] is not None]
            lat = sorted(r[1] for r in ok)
            toks = sum(r[2] or 0 for r in ok)

            print(f"\n--- 라운드 {rnd} / 동시 {args.concurrency} ---")
            print(f"성공 {len(ok)}/{len(results)}   전체 소요 {wall:.1f}s")
            if lat:
                print(
                    f"지연  p50 {statistics.median(lat):.1f}s"
                    f"  p95 {lat[int(len(lat) * 0.95) - 1]:.1f}s"
                    f"  max {lat[-1]:.1f}s"
                )
                print(f"출력 토큰 {toks}  →  집계 처리량 {toks / wall:.1f} tok/s")
            for idx, dt, _, err in bad[:5]:
                print(f"  실패 #{idx} ({dt:.1f}s): {err}")


if __name__ == "__main__":
    asyncio.run(main())
