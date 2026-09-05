"""채팅 스트리밍 지연·빈 토큰 측정.

## 왜 필요한가

`/chat/stream` 은 NestJS 가 붙는 주 엔드포인트인데, 두 가지가 문제로 보고됐다.

1. **응답이 오래 걸린다** (답은 온다)
2. **초반에 빈 문자열 토큰이 몇 개 온다**

둘 다 재는 도구가 없었다. `bench_quiz_latency.py` 는 퀴즈 경로 전용이고,
런북 §8-2 의 SSE 검증은 일회성 PowerShell 계측이라 반복할 수 없다.

## 무엇을 재는가

라우터를 거치지 않고 `chain.py` 의 스트리밍 함수를 직접 돌린다.
HTTP/프록시를 빼고 **LLM 스트림 자체의 성질만** 남기기 위해서다.

- 첫 청크까지 / **첫 '내용 있는' 청크까지** / 전체
- 빈 청크 개수와 비율  ← 사고 토큰 구간의 크기를 그대로 보여준다
- RAG 경로는 적재(뉴스/공시/재무)와 검색 시간을 따로 나눠 잰다

## 쓰는 법

    uv run python scripts/bench_chat_stream.py                  # 일반 질문
    uv run python scripts/bench_chat_stream.py --mode rag       # 종목 질문(적재 포함)

    # 사고 강도를 바꿔가며 비교 (셸 환경변수가 .env 보다 우선)
    $env:LLM_REASONING_EFFORT="none"; uv run python scripts/bench_chat_stream.py
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _args():
    p = argparse.ArgumentParser(description="채팅 스트리밍 측정")
    p.add_argument("--mode", choices=["general", "rag"], default="general")
    p.add_argument("--question", default="")
    p.add_argument("--rounds", type=int, default=1)
    return p.parse_args()


A = _args()

from app.core.config import settings              # noqa: E402
from app.schemas.chat import UserContext          # noqa: E402

DEFAULT_Q = {
    "general": "PER과 PBR의 차이를 초보자에게 설명해줘",
    "rag": "삼성전자 요즘 어때?",
}


async def _run_once() -> dict:
    from app.services.rag.chain import stream_general_chain, stream_rag_chain
    from app.services.rag.ticker_extractor import extract_tickers

    question = A.question or DEFAULT_Q[A.mode]
    started = time.perf_counter()
    ingest_sec = 0.0

    if A.mode == "rag":
        from app.routers.chat import _ingest_all_sources
        from app.services.external.dart import load_corp_codes

        # extract_tickers 는 DART 종목 코드 맵을 참조하는데, 그 맵은 앱 lifespan
        # 에서 채워진다. 라우터를 안 거치는 이 스크립트는 직접 불러줘야 한다
        # (안 부르면 종목을 못 찾아 조용히 general 경로로 새어나간다).
        await load_corp_codes()
        tickers = extract_tickers(question)
        if not tickers:
            print(f"  종목을 못 찾음: {question!r} - general 로 돈다")
            chunks = stream_general_chain(question, [], "초급")
        else:
            t0 = time.perf_counter()
            await _ingest_all_sources(tickers)
            ingest_sec = time.perf_counter() - t0
            print(f"  종목 {tickers} / 적재 {ingest_sec:.1f}초")
            chunks = stream_rag_chain(question, tickers, [], "초급")
    else:
        chunks = stream_general_chain(question, [], "초급")

    first_chunk = None
    first_content = None
    empty = 0
    filled = 0
    text_len = 0

    async for chunk in chunks:
        now = time.perf_counter() - started
        if first_chunk is None:
            first_chunk = now
        if chunk:
            if first_content is None:
                first_content = now
            filled += 1
            text_len += len(chunk)
        else:
            empty += 1

    return {
        "total": time.perf_counter() - started,
        "ingest": ingest_sec,
        "first_chunk": first_chunk or 0.0,
        "first_content": first_content or 0.0,
        "empty": empty,
        "filled": filled,
        "chars": text_len,
    }


async def main() -> None:
    print(f"모드      : {A.mode}")
    print(f"모델      : {settings.LLM_MODEL} @ {settings.LLM_BASE_URL}")
    # 채팅 경로는 CHAT_REASONING_EFFORT 를 쓴다. 전역값을 찍으면 오해한다.
    effort = (settings.CHAT_REASONING_EFFORT or settings.LLM_REASONING_EFFORT).strip()
    print(f"사고 강도 : {effort or '미지정'} (채팅 경로 기준)\n")

    for i in range(1, A.rounds + 1):
        r = await _run_once()
        total_chunks = r["empty"] + r["filled"]
        ratio = (r["empty"] / total_chunks * 100) if total_chunks else 0.0
        print(f"  {i}회차  전체 {r['total']:.1f}초"
              + (f" (적재 {r['ingest']:.1f}초)" if r["ingest"] else ""))
        print(f"         첫 청크 {r['first_chunk']:.1f}초 / "
              f"첫 내용 {r['first_content']:.1f}초"
              f"  ← 사용자가 글자를 처음 보는 시점")
        print(f"         청크 {total_chunks}개 중 빈 것 {r['empty']}개 ({ratio:.0f}%), "
              f"본문 {r['chars']}자\n")


if __name__ == "__main__":
    asyncio.run(main())
