"""`/chat/stream` 을 실제 HTTP 로 측정한다 (SSE).

`bench_chat_stream.py` 는 라우터를 건너뛰고 체인을 직접 돌린다 - LLM 스트림의
성질만 보려는 의도였다. 그래서 **프록시를 거친 뒤에도 같은지는 못 본다.**
런북 §8-2 의 SSE 검증은 일회성 PowerShell 계측이라 반복할 수 없었다.

여기서 보는 것:

  첫 이벤트까지 / 첫 '내용 있는' token 까지 / 완료까지
  빈 content 비율        서버가 사고 구간을 걸러내고 있는지 (백엔드 전달사항)
  : keepalive 주석 수     10초 이상 침묵이 있었는지
  버퍼링                 이벤트가 분산 도착하는지, 끝에 몰려 오는지

    uv run python scripts/check_chat_stream.py \
        --base-url https://<POD_ID>-8080.proxy.runpod.net --api-key X
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


async def run(args):
    url = args.base_url.rstrip("/") + "/chat/stream"
    body = {
        "question": args.question,
        "user": {"user_id": 990003, "investment_level": "초급"},
        "history": [],
    }
    headers = {"X-API-Key": args.api_key, "Content-Type": "application/json"}
    print(f"POST {url}\n질문: {args.question}\n")

    t0 = time.monotonic()
    first_event = first_content = None
    events = {}            # event 이름별 개수
    tokens = empty = 0
    keepalives = 0
    text = []
    arrivals = []          # 내용 있는 토큰의 도착 시각

    timeout = httpx.Timeout(args.timeout, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, json=body, headers=headers) as r:
            print(f"[{time.monotonic() - t0:6.2f}s] 헤더  status={r.status_code} "
                  f"content-type={r.headers.get('content-type')} "
                  f"CF-RAY={r.headers.get('cf-ray', '-')}")
            if r.status_code != 200:
                print(await r.aread())
                return 1
            cur_event = None
            async for line in r.aiter_lines():
                now = time.monotonic() - t0
                if first_event is None:
                    first_event = now
                if line.startswith(":"):
                    keepalives += 1
                    print(f"[{now:6.2f}s] 주석 {line!r}")
                    continue
                if line.startswith("event:"):
                    cur_event = line.split(":", 1)[1].strip()
                    events[cur_event] = events.get(cur_event, 0) + 1
                    continue
                if not line.startswith("data:"):
                    continue
                raw = line.split(":", 1)[1].strip()
                try:
                    data = json.loads(raw)
                except Exception:  # noqa: BLE001
                    continue
                if cur_event == "token" or "content" in data:
                    tokens += 1
                    content = data.get("content", "")
                    if content == "":
                        empty += 1
                    else:
                        text.append(content)
                        arrivals.append(now)
                        if first_content is None:
                            first_content = now
                            print(f"[{now:6.2f}s] 첫 내용 토큰: {content!r}")

    total = time.monotonic() - t0
    print(f"[{total:6.2f}s] 완료\n")

    print(f"이벤트: {events}")
    print(f"token {tokens}개 중 빈 content {empty}개 ({empty / tokens * 100:.1f}%)"
          if tokens else "token 이벤트 없음")
    print(f": keepalive 주석 {keepalives}회")
    print(f"첫 이벤트 {first_event:.2f}s / 첫 내용 토큰 "
          f"{first_content if first_content is None else round(first_content, 2)}s / 완료 {total:.2f}s")
    print(f"조립 길이 {len(''.join(text))}자")

    ok = True
    if arrivals and len(arrivals) > 2:
        spread = arrivals[-1] - arrivals[0]
        if spread < 0.5:
            print("판정: ❌ 버퍼링 - 토큰이 한꺼번에 도착했다. 스트리밍이 무의미하다.")
            ok = False
        else:
            print(f"판정: ✅ 무버퍼링 - 토큰이 {spread:.1f}초에 걸쳐 분산 도착했다.")

    ratio = (empty / tokens * 100) if tokens else 0
    if ratio > 10:
        print(f"판정: ⚠️ 빈 토큰이 {ratio:.1f}% - 서버 필터가 안 먹었거나 "
              "사고 강도가 높다(CHAT_REASONING_EFFORT 확인).")
        ok = False
    else:
        print(f"판정: ✅ 빈 토큰 {ratio:.1f}% - 필터 동작 중.")

    if args.show:
        print("\n" + "".join(text)[:1200])
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--question", default="PER과 PBR의 차이를 초보자에게 설명해줘")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
