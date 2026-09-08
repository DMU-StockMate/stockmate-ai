"""하트비트 응답이 실제 네트워크 경로에서 통하는지 확인한다.

`app/core/keepalive.py` 의 전제는 "응답 바이트가 흐르기 시작하면 Cloudflare 가
안 끊는다"는 것이다. 이건 코드 테스트로는 확인할 수 없다 - **중간 프록시가
application/json 을 버퍼링해 버리면 하트비트가 도착하지 않아 소용이 없다.**
그래서 실제 URL 로 재는 도구가 따로 필요하다.

보는 것:

  헤더 도착      프록시가 응답 시작을 언제 통과시켰나
  박동 타임라인  공백이 실시간으로 흘러오나, 아니면 끝에 몰려서 오나  ← 핵심
  본문 도착      전체 소요
  파싱           선행 공백이 붙은 채로 표준 파서가 그대로 읽나

**버퍼링 판정**: 박동이 흩어져 도착하면 통과, 마지막에 한꺼번에 오면 버퍼링이다.
버퍼링이면 100초 벽이 그대로 남으므로 SSE 전환(2순위)으로 가야 한다.

    uv run python scripts/check_keepalive.py --base-url http://127.0.0.1:8000 --api-key X
    uv run python scripts/check_keepalive.py \
        --base-url https://<POD_ID>-8080.proxy.runpod.net --api-key X --endpoint prompt
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import httpx  # noqa: E402

USER = {"user_id": 990001, "investment_level": "초급"}

BODIES = {
    "prompt": lambda a: ("/quiz/generate/prompt", {
        "prompt": a.prompt,
        "user": USER,
    }),
    "generate": lambda a: ("/quiz/generate", {
        "quiz_type": "MULTIPLE_CHOICE",
        "topic": "PER",
        "count": a.count,
        "user": USER,
    }),
    "review": lambda a: ("/quiz/generate/review", {
        "user": USER,
        "wrong_answers": [
            {
                "question_text": "PER 이 낮으면 항상 저평가된 주식이다.",
                "question_type": "OX",
                "detail_code": "PER_BASIC",
                "choices": [
                    {"choice_no": 1, "text": "O", "is_correct": False},
                    {"choice_no": 2, "text": "X", "is_correct": True},
                ],
                "selected_choice_no": 1,
            },
            {
                "question_text": "ROE 는 무엇을 나타내는 지표인가?",
                "question_type": "MULTIPLE_CHOICE",
                "detail_code": "ROE_BASIC",
                "choices": [
                    {"choice_no": 1, "text": "자기자본이익률", "is_correct": True},
                    {"choice_no": 2, "text": "총자산회전율", "is_correct": False},
                    {"choice_no": 3, "text": "부채비율", "is_correct": False},
                    {"choice_no": 4, "text": "배당성향", "is_correct": False},
                ],
                "selected_choice_no": 3,
            },
        ],
    }),
}


async def run(args):
    path, body = BODIES[args.endpoint](args)
    url = args.base_url.rstrip("/") + path
    headers = {"X-API-Key": args.api_key, "Content-Type": "application/json"}

    print(f"POST {url}")
    print(f"타임아웃 {args.timeout}초 / 엔드포인트 {args.endpoint}\n")

    t0 = time.monotonic()
    beats: list[tuple[float, int]] = []      # (도착 시각, 바이트 수)
    payload = b""
    header_at = None

    timeout = httpx.Timeout(args.timeout, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=body, headers=headers) as r:
            header_at = time.monotonic() - t0
            print(f"[{header_at:6.1f}s] 헤더 도착  status={r.status_code} "
                  f"content-type={r.headers.get('content-type')} "
                  f"transfer-encoding={r.headers.get('transfer-encoding', '-')}")
            if r.status_code != 200:
                print(await r.aread())
                return 1
            async for chunk in r.aiter_raw():
                at = time.monotonic() - t0
                if chunk.strip() == b"":
                    beats.append((at, len(chunk)))
                    print(f"[{at:6.1f}s] 박동 {len(chunk)}바이트")
                else:
                    payload += chunk
                    if len(payload) == len(chunk):
                        print(f"[{at:6.1f}s] 본문 시작")

    total = time.monotonic() - t0
    print(f"[{total:6.1f}s] 완료\n")

    # ── 판정 ──
    whitespace = sum(n for _, n in beats)
    print(f"박동 {len(beats)}회 / 공백 {whitespace}바이트 / 본문 {len(payload)}바이트")

    ok = True
    if not beats:
        print("판정: 하트비트 없음 - 응답이 대기 시간 안에 끝났거나 스위치가 꺼져 있다.")
    else:
        spread = beats[-1][0] - beats[0][0]
        gaps = [round(beats[i][0] - beats[i - 1][0], 1) for i in range(1, len(beats))]
        print(f"박동 간격: {gaps}")
        # 박동이 실시간으로 흩어져 왔는지. 전부 0.5초 안에 몰려 왔으면 버퍼링이다.
        if len(beats) >= 2 and spread < 0.5:
            print("판정: ❌ 버퍼링 - 박동이 한꺼번에 도착했다. 100초 벽이 그대로 남는다.")
            ok = False
        else:
            print(f"판정: ✅ 무버퍼링 - 박동이 {spread:.1f}초에 걸쳐 분산 도착했다.")

    try:
        parsed = json.loads(payload.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"판정: ❌ 본문 파싱 실패: {e}")
        print(payload[:400])
        return 1

    if parsed.get("keepalive_error"):
        print("파싱: ✅ 본문은 읽혔지만 **생성 실패 응답**이다 "
              "(하트비트가 시작된 뒤 실패해 상태 코드를 못 바꾼 경우)")
        print(f"  error_type: {parsed.get('error_type')}")
        print(f"  detail    : {parsed.get('detail')}")
        return 1

    n = len(parsed.get("questions", []))
    print(f"파싱: ✅ 선행 공백이 붙은 채로 그대로 읽힘, 문제 {n}개")
    if n == 0:
        print("경고: questions 가 비었다.")
        ok = False

    if args.show:
        for i, q in enumerate(parsed.get("questions", []), 1):
            print(f"\n[{i}] ({q.get('topic')}) {q.get('question_text')}")
            for c in q.get("choices", []):
                mark = "O" if c.get("is_correct") else " "
                print(f"  {mark} {c.get('choice_no')}. {c.get('text')}")
            print(f"    해설: {q.get('explanation')}")

    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--endpoint", default="prompt", choices=sorted(BODIES))
    p.add_argument("--prompt", default="PER 과 PBR 에 대한 문제를 5개 만들어줘")
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--show", action="store_true", help="생성된 문제 내용도 출력")
    args = p.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
