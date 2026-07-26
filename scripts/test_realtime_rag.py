"""RAG 최신성/실시간 정확성 테스트 스크립트.

실행 중인 StockMate AI API(/chat/evaluate)와 원본 API(KIS/DART/Naver)를 동시에 호출해
"모델이 지금 이 순간의 최신 사건/실시간 수치를 제대로 보고 답하는가"를 대조 검증한다.

전제: 서버가 떠 있어야 함
    uv run uvicorn app.main:app --reload   (기본 http://localhost:8000)

사용 예:
    uv run python scripts/test_realtime_rag.py
    uv run python scripts/test_realtime_rag.py -q "SK하이닉스 최근 공시 알려줘"
    uv run python scripts/test_realtime_rag.py -q "삼성전자 실적 어때" --level 중급
    uv run python scripts/test_realtime_rag.py --negative -q "신풍제약 최근 뉴스 있어?"

판정 표기:
    [PASS] 결정적 검증 통과   [FAIL] 결정적 검증 실패
    [OK]/[WARN] 소프트 체크 (LLM 자유서술이라 참고용, 실패해도 곧바로 버그는 아님)
"""
import argparse
import asyncio
import os
import sys

import httpx

# 프로젝트 루트를 path에 추가 (어느 cwd에서 실행해도 app 패키지 import 되도록)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from app.services.external.dart import load_corp_codes, get_disclosures
from app.services.external.naver_news import search_news
from app.services.rag.chain import _SOURCE_SEARCH_CONFIG

# 참고: 실시간 시세(KIS)는 원본을 독립 호출하지 않는다. KIS는 토큰 재발급을 1분에 1회로
# 제한하는데, 서버가 이미 토큰을 캐시해 쓰고 있어 스크립트가 또 발급하면 403이 난다.
# 그래서 시세는 서버 응답(stock_data)을 신뢰값으로 사용한다(서버는 시세를 매 호출 실시간 조회).

GREEN, RED, YEL, DIM, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"


def _tag(ok: bool, hard: bool = True) -> str:
    if hard:
        return f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"
    return f"{GREEN}[OK]{RESET}" if ok else f"{YEL}[WARN]{RESET}"


def _fmt_date(v) -> str:
    try:
        s = str(int(v))
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}" if len(s) == 8 else str(v)
    except (ValueError, TypeError):
        return str(v)


import re


def check_number_grounding(answer: str, context: str) -> tuple[list, list]:
    """답변에 등장한 숫자가 LLM에 넣은 컨텍스트에 실제로 있는지 확인해 근거 없는 숫자를 잡는다.

    숫자 할루시네이션(컨텍스트에 없는 금액·지표를 지어내는 것)이 가장 위험한 오류라
    이를 소프트 신호로 탐지한다. 콤마/공백 제거 후 숫자 코어로 느슨하게 매칭하며
    (예: 답변 "2978억" ↔ 컨텍스트 "2,978.0억원"), 한 자리 수는 노이즈라 제외한다.
    반환: (답변 내 숫자 목록, 컨텍스트에서 못 찾은 숫자 목록)
    """
    ans_nums = re.findall(r"\d[\d,\.]*\d|\d", answer)
    ctx_norm = context.replace(",", "").replace(" ", "")
    grounded, ungrounded = [], []
    for n in ans_nums:
        core = n.replace(",", "").rstrip(".")
        if len(core.replace(".", "")) < 2:  # 한 자리 숫자(월/일 등)는 스킵
            continue
        if core in ctx_norm or n in context:
            grounded.append(n)
        else:
            ungrounded.append(n)
    return grounded, ungrounded


async def call_evaluate(base_url: str, question: str, level: str) -> dict:
    body = {"question": question, "user": {"user_id": 0, "investment_level": level}}
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(f"{base_url}/chat/evaluate", json=body)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}", "traceback": r.text[:2000]}
        return r.json()


def _cutoff_int(days: int) -> int:
    return int((datetime.now() - timedelta(days=days)).strftime("%Y%m%d"))


def print_retrieved(retrieved: list[dict]) -> None:
    print(f"\n{DIM}── 검색 후보 (score·날짜·채택여부) ──{RESET}")
    if not retrieved:
        print("  (검색된 문서 없음)")
        return
    order = {"naver_news": 0, "dart": 1, "dart_financials": 2}
    for r in sorted(retrieved, key=lambda x: (order.get(x["source"], 9), -(x["published_at"] or 0))):
        mark = "★채택" if r["taken"] else ("·통과" if r["passed_score"] else "✕컷")
        head = r["content"].replace("\n", " ")[:50]
        print(f"  {r['source']:15} {_fmt_date(r['published_at'])}  score={r['score']:.3f}  {mark}  {DIM}{head}{RESET}")


async def run_positive(base_url: str, question: str, level: str) -> int:
    print(f"\n{'='*70}\n질문: {question}  (수준: {level})\n{'='*70}")
    res = await call_evaluate(base_url, question, level)

    if res.get("error"):
        print(f"{RED}서버 에러: {res['error']} ({res.get('error_type', '')}){RESET}")
        print(f"{DIM}{res.get('traceback', '')[:1800]}{RESET}")
        return 1

    tickers = res.get("tickers", [])
    print(f"추출 종목: {tickers or '(없음)'}   mode: {res.get('mode')}   적재: {res.get('ingested')}")
    print(f"\n{DIM}── 답변 ──{RESET}\n{res.get('answer', '')[:600]}")

    retrieved = res.get("retrieved", [])
    print_retrieved(retrieved)

    if res.get("mode") != "rag" or not tickers:
        print(f"\n{YEL}종목이 추출되지 않아 일반(비RAG) 답변입니다. 최신성 검증 생략.{RESET}")
        return 0

    ticker = tickers[0]
    # 원본 뉴스/공시를 직접 호출해 '지금 이 순간'의 정답을 만든다 (KIS는 위 주석대로 제외)
    gt_news, gt_disc = await asyncio.gather(
        search_news(ticker, display=5),
        get_disclosures(ticker, days=90),
    )

    fails = 0
    print(f"\n{DIM}── 검증 ──{RESET}")

    # 1) [HARD] 검색된 문서가 전부 설정된 기간(cutoff) 안에 있는가 (오래된 문서 혼입 방지)
    bound_ok = True
    for r in retrieved:
        cfg = _SOURCE_SEARCH_CONFIG.get(r["source"])
        if not cfg or r["published_at"] is None:
            continue
        if int(r["published_at"]) < _cutoff_int(cfg["cutoff_days"]):
            bound_ok = False
            print(f"    · 기간 초과 문서: {r['source']} {_fmt_date(r['published_at'])}")
    print(f"  {_tag(bound_ok)} 검색 문서 최신성(기간 내) 보장")
    fails += 0 if bound_ok else 1

    # 2) [HARD] 채택된 문서는 모두 score 임계값을 통과했는가
    taken = [r for r in retrieved if r["taken"]]
    score_ok = all(r["passed_score"] for r in taken)
    print(f"  {_tag(score_ok)} 채택 문서 score 임계값 통과 (채택 {len(taken)}건)")
    fails += 0 if score_ok else 1

    # 3) [SOFT] 원본 최신 뉴스 헤드라인이 검색 뉴스에 반영됐는가
    news_records = [r for r in retrieved if r["source"] == "naver_news"]
    if gt_news:
        newest = gt_news[0]["title"]
        covered = any(newest[:15] and newest[:15] in r["content"] for r in news_records)
        print(f"  {_tag(covered, hard=False)} 원본 최신 뉴스 반영  {DIM}(원본 최신: {newest[:40]}){RESET}")

    # 4) [SOFT] 원본 최신 공시가 검색 공시에 반영됐는가
    disc_records = [r for r in retrieved if r["source"] == "dart"]
    if gt_disc:
        newest_d = gt_disc[0]["title"]
        covered_d = any(newest_d[:12] and newest_d[:12] in r["content"] for r in disc_records)
        print(f"  {_tag(covered_d, hard=False)} 원본 최신 공시 반영  {DIM}(원본 최신: {newest_d[:40]}){RESET}")

    # 5) [SOFT] 실시간 시세 — 서버 응답값(실시간 조회) 기준
    api_stock = res.get("stock_data", {}).get(ticker)
    if api_stock and api_stock.get("current_price") is not None:
        price = api_stock["current_price"]
        print(f"  {DIM}서버 실시간 시세: 현재가 {price:,}원 "
              f"({api_stock.get('change_rate')}%) | PER {api_stock.get('per')} | PBR {api_stock.get('pbr')}{RESET}")
        # 6) [SOFT] 답변 본문에 현재가가 실제로 언급됐는가
        price_str = f"{price:,}"
        in_answer = price_str in res.get("answer", "")
        print(f"  {_tag(in_answer, hard=False)} 답변에 현재가 언급  {DIM}({price_str}원){RESET}")
    else:
        print(f"  {YEL}[WARN]{RESET} 서버 stock_data 비어있음 (KIS 키/모의투자 도메인/장시간 확인)")

    # 7) [SOFT] 답변 숫자의 컨텍스트 근거 (숫자 할루시네이션 탐지)
    grounded, ungrounded = check_number_grounding(res.get("answer", ""), res.get("context_sent_to_llm", ""))
    print(f"  {_tag(not ungrounded, hard=False)} 답변 숫자 근거  {DIM}(근거있음 {len(grounded)} / 근거없음 {len(ungrounded)}){RESET}")
    if ungrounded:
        print(f"    {YEL}근거 못 찾은 숫자: {', '.join(ungrounded[:8])}{RESET}  {DIM}(지어냈을 가능성 — 답변과 대조 확인){RESET}")

    return fails


async def run_negative(base_url: str, question: str, level: str) -> int:
    """뉴스/공시가 거의 없는 종목 → 자료 없다고 정직하게 말해야 하고, 지어내면 안 된다."""
    print(f"\n{'='*70}\n[음성 테스트] {question}\n{'='*70}")
    res = await call_evaluate(base_url, question, level)
    if res.get("error"):
        print(f"{RED}서버 에러: {res['error']} ({res.get('error_type', '')}){RESET}")
        print(f"{DIM}{res.get('traceback', '')[:1800]}{RESET}")
        return 1
    retrieved = res.get("retrieved", [])
    taken = [r for r in retrieved if r["taken"]]
    answer = res.get("answer", "")
    print(f"추출 종목: {res.get('tickers')}   채택 문서: {len(taken)}건")
    print(f"\n{DIM}── 답변 ──{RESET}\n{answer[:400]}")
    print_retrieved(retrieved)

    print(f"\n{DIM}── 검증 ──{RESET}")
    honest_phrases = ["찾을 수 없", "자료가 없", "정보가 없", "확인되지 않", "없습니다"]
    honest = (len(taken) == 0 and any(p in answer for p in honest_phrases))
    print(f"  {_tag(honest, hard=False)} 자료 없을 때 정직하게 응답(할루시네이션 아님)")
    if len(taken) > 0:
        print(f"    {DIM}※ 채택 문서가 {len(taken)}건 있으니 '자료 있는 종목'입니다. "
              f"진짜 음성 테스트는 뉴스/공시가 거의 없는 소형주로 하세요.{RESET}")

    # 답변 숫자가 실제 검색 근거에 있는지 (할루시네이션 탐지)
    grounded, ungrounded = check_number_grounding(answer, res.get("context_sent_to_llm", ""))
    print(f"  {_tag(not ungrounded, hard=False)} 답변 숫자 근거  {DIM}(근거있음 {len(grounded)} / 근거없음 {len(ungrounded)}){RESET}")
    if ungrounded:
        print(f"    {YEL}근거 못 찾은 숫자: {', '.join(ungrounded[:8])}{RESET}  {DIM}(지어냈을 가능성){RESET}")
    return 0


async def main() -> None:
    ap = argparse.ArgumentParser(description="StockMate RAG 최신성/실시간 정확성 테스트")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("-q", "--question", default="삼성전자 오늘 주가랑 최근 이슈 알려줘")
    ap.add_argument("--level", default="초급")
    ap.add_argument("--negative", action="store_true", help="음성 테스트(자료 없는 종목) 모드")
    args = ap.parse_args()

    await load_corp_codes()  # 원본 DART 조회에 필요한 종목코드 로드

    if args.negative:
        await run_negative(args.base_url, args.question, args.level)
        return

    fails = await run_positive(args.base_url, args.question, args.level)
    print(f"\n{'='*70}")
    if fails == 0:
        print(f"{GREEN}결정적(HARD) 검증 전부 통과.{RESET} 소프트 체크는 위 표기 참고.")
    else:
        print(f"{RED}결정적(HARD) 검증 {fails}건 실패.{RESET} 위 [FAIL] 항목 확인.")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
