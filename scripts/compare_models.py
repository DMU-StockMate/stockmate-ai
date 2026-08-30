"""모델 교체 판단용 A/B 비교 수집기.

같은 입력을 서로 다른 LLM 백엔드에 통과시켜 결과를 JSON 으로 남긴다.
채점은 하지 않는다. 수집만 하고 판단은 사람이 한다.

두 모델을 동시에 띄울 VRAM 이 없으므로 한 번에 하나씩 두 번 실행한다.

  1) 현행 기준선 - .env: LLM_BACKEND=ollama, LLM_MODEL=qwen3.5:9b
        uv run python scripts/compare_models.py --tag 9b

  2) 후보 - 먼저 llama-server 를 띄운다
        llama-server.exe -m <gguf> -ngl 999 --n-cpu-moe 34 --flash-attn on -c 16384 --port 8080
     .env: LLM_BACKEND=openai / LLM_BASE_URL=http://127.0.0.1:8080/v1 / LLM_MODEL=qwen3.6-35b-a3b
        uv run python scripts/compare_models.py --tag 35b

결과: compare_out/compare_<tag>.json

측정 영역 (--only 로 하나만 돌릴 수 있다)
  terms - 한국어 금융 용어 정확도. LLM 만 있으면 되고 자동 채점된다.
          Qdrant 가 꺼져 있어도 이것만 먼저 돌아간다.
  quiz  - 실제 생성 파이프라인(generate_quiz). 방어 장치가 몇 번 걸렀는지,
          최종 산출물에 결함이 남았는지. Qdrant 필요.
  rag   - run_rag_chain 고정 질의. Qdrant + 수집된 문서 필요.

공정성을 위해 실행 중에는 퀴즈 뱅크와 학습데이터 수집을 끈다.
뱅크를 켜두면 두 번째 실행이 첫 번째가 쌓아놓은 문제와 중복 판정되어 불리해진다.
"""
import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.core.config import settings  # noqa: E402

# 비교 공정성 - 상태를 남기는 기능을 끈다
settings.QUIZ_BANK_ENABLED = False
# 학습데이터 수집 필드는 archive/finetune-9b 브랜치에만 있다.
# develop 에는 없으므로 존재할 때만 끈다.
if hasattr(settings, "DATASET_COLLECT_ENABLED"):
    settings.DATASET_COLLECT_ENABLED = False

from app.core.llm import build_llm            # noqa: E402
from app.schemas.chat import UserContext      # noqa: E402
from app.services.quiz import quality         # noqa: E402

USER = UserContext(user_id=1, investment_level="초급")

# 추론 모드를 켜면 llama-server(--jinja)가 사고 과정을 본문에 섞어 보내기도 한다.
# 그대로 채점하면 assert_korean 이 영어 혼입으로 오탐하고 키워드 판정도 흐려진다.
_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    return _THINK.sub("", text).strip()

# =========================================================
# 1. 한국어 금융 용어 정확도 (자동 채점)
# =========================================================
# expect : 이 중 하나라도 들어가야 하는 키워드 그룹들. 그룹마다 최소 1개 필요.
# forbid : 들어가면 오답인 표현. quality.py 에 기록된 실측 오류에서 가져왔다.

TERM_CASES = [
    {"id": "per_def",
     "q": "주식 투자 용어를 묻는 거야. PER이 무엇인지 한 문장으로 정의하고 계산식도 알려줘.",
     "expect": [["주가", "주식 가격", "주식의 가격", "현재가", "시가총액"],
                ["주당순이익", "EPS", "순이익"]],
     "forbid": ["순이익률인 PER", "주가 대비 순이익률"]},
    {"id": "pbr_def",
     "q": "주식 투자 용어를 묻는 거야. PBR(주가순자산비율)이 무엇인지 한 문장으로 정의하고 계산식도 알려줘.",
     "expect": [["주가", "주식 가격", "주식의 가격", "현재가", "시가총액"],
                ["주당순자산", "BPS", "순자산", "자본"]],
     "forbid": ["렌더링", "Rendering"]},
    {"id": "roe_def",
     "q": "주식 투자 용어를 묻는 거야. ROE가 무엇인지 한 문장으로 정의하고 계산식도 알려줘.",
     "expect": [["자기자본", "자본"], ["순이익", "당기순이익"]],
     "forbid": []},
    {"id": "eps_def",
     "q": "주식 투자 용어를 묻는 거야. EPS가 무엇인지 한 문장으로 정의해줘.",
     "expect": [["순이익", "이익"], ["주식", "주당"]],
     "forbid": []},
    {"id": "debt_ratio",
     "q": "재무제표 용어를 묻는 거야. 부채비율은 어떻게 계산해? 한 문장으로.",
     "expect": [["부채"], ["자기자본", "자본"]],
     "forbid": []},
    {"id": "per_low_meaning",
     "q": "PER이 낮으면 무조건 저평가라고 볼 수 있어? 두세 문장으로 답해줘.",
     "expect": [["아니", "않", "없습니다", "없다", "단정", "무조건"],
                ["업종", "산업", "성장", "비교", "전망"]],
     "forbid": []},
    {"id": "rights_offering",
     "q": "유상증자와 무상증자의 차이를 두 문장으로 설명해줘.",
     "expect": [["자금", "납입", "돈", "대금", "유입"],
                ["잉여금", "무상", "대가", "자본금 전입", "무상으로"]],
     "forbid": []},
    {"id": "op_vs_net",
     "q": "영업이익과 당기순이익의 차이를 두 문장으로 설명해줘.",
     "expect": [["영업"], ["영업외", "이자", "법인세", "세금", "세후"]],
     "forbid": []},
    {"id": "market_cap",
     "q": "시가총액은 어떻게 계산해? 한 문장으로.",
     "expect": [["주가", "현재가", "주식의 가격", "현재 가격", "주식 가격"],
                ["발행", "주식 수", "주식수", "주식 총수", "상장주식", "총 주식"]],
     "forbid": []},
    {"id": "dividend_yield",
     "q": "배당수익률은 어떻게 계산해? 한 문장으로.",
     "expect": [["배당"], ["주가", "현재가", "주식 가격", "주식의 가격"]],
     "forbid": []},
    {"id": "per_vs_pbr",
     "q": "PER과 PBR의 차이를 두 문장으로 설명해줘.",
     "expect": [["이익", "순이익", "수익"], ["자산", "순자산", "장부"]],
     "forbid": ["렌더링"]},
    {"id": "roe_vs_roa",
     "q": "ROE와 ROA의 차이를 두 문장으로 설명해줘.",
     "expect": [["자기자본", "자본"], ["총자산", "자산"]],
     "forbid": []},
]


def score_terms(text: str, case: dict) -> dict:
    """키워드 기반 자동 채점 + 한국어 순도 검사."""
    hits, misses = [], []
    for group in case["expect"]:
        if any(k in text for k in group):
            hits.append(group[0])
        else:
            misses.append("|".join(group))
    forbidden = [f for f in case["forbid"] if f in text]

    try:
        quality.assert_korean(text, where="용어설명")
        korean_ok, korean_err = True, None
    except Exception as e:
        korean_ok, korean_err = False, str(e)

    return {
        "hit": hits, "miss": misses, "forbidden_hit": forbidden,
        "pass": not misses and not forbidden,
        "korean_ok": korean_ok, "korean_err": korean_err,
    }


async def run_terms() -> list:
    llm = build_llm(temperature=0.3)
    out = []
    for i, case in enumerate(TERM_CASES, 1):
        print(f"  [terms {i}/{len(TERM_CASES)}] {case['id']}", flush=True)
        t0 = time.perf_counter()
        try:
            resp = await llm.ainvoke(case["q"])
            text = strip_think(getattr(resp, "content", str(resp)))
            err = None
        except Exception as e:
            text, err = "", f"{type(e).__name__}: {e}"
        elapsed = round(time.perf_counter() - t0, 2)
        rec = {"id": case["id"], "q": case["q"], "response": text,
               "elapsed_s": elapsed, "error": err}
        rec["auto"] = score_terms(text, case) if not err else None
        out.append(rec)
    return out


# =========================================================
# 2. 퀴즈 생성 파이프라인
# =========================================================

QUIZ_CASES = [
    ("OX", "PER"), ("OX", "PBR"), ("OX", "ROE"), ("OX", "배당"),
    ("OX", "시가총액"), ("OX", "부채비율"),
    ("MULTIPLE_CHOICE", "PER"), ("MULTIPLE_CHOICE", "PBR"),
    ("MULTIPLE_CHOICE", "ROE"), ("MULTIPLE_CHOICE", "시가총액"),
    ("MULTIPLE_CHOICE", "영업이익"), ("MULTIPLE_CHOICE", "유상증자"),
]


async def run_quiz(repeat: int) -> list:
    from app.services.quiz.generator import generate_quiz

    out = []
    n = 0
    total = len(QUIZ_CASES) * repeat
    for _ in range(repeat):
        for qtype, topic in QUIZ_CASES:
            n += 1
            print(f"  [quiz {n}/{total}] {qtype} / {topic}", flush=True)
            t0 = time.perf_counter()
            try:
                result = await generate_quiz(USER, qtype, topic=topic)
                err = None
            except Exception as e:
                result, err = None, f"{type(e).__name__}: {e}"
            out.append({
                "quiz_type": qtype, "topic": topic,
                "elapsed_s": round(time.perf_counter() - t0, 2),
                "error": err, "result": result,
            })
    return out


# =========================================================
# 3. RAG 답변
# =========================================================

RAG_CASES = [
    {"q": "삼성전자 지금 주가 어때?", "tickers": ["삼성전자"]},
    {"q": "삼성전자 최근 공시 뭐 있어?", "tickers": ["삼성전자"]},
    {"q": "SK하이닉스 최근 뉴스 요약해줘", "tickers": ["SK하이닉스"]},
    {"q": "삼성전자 PER은 지금 어느 수준이야?", "tickers": ["삼성전자"]},
    {"q": "카카오 요즘 어때? 투자해도 될까?", "tickers": ["카카오"]},
    {"q": "SK하이닉스랑 삼성전자 중에 뭐가 더 나아?", "tickers": ["삼성전자", "SK하이닉스"]},
    {"q": "PER이 낮은 종목은 다 사도 되는 거야?", "tickers": []},
]


async def run_rag() -> list:
    from app.services.rag.chain import run_rag_chain

    out = []
    for i, case in enumerate(RAG_CASES, 1):
        print(f"  [rag {i}/{len(RAG_CASES)}] {case['q']}", flush=True)
        t0 = time.perf_counter()
        try:
            answer = await run_rag_chain(case["q"], case["tickers"], [], "초급")
            err = None
        except Exception as e:
            answer, err = "", f"{type(e).__name__}: {e}"
        try:
            quality.assert_korean(answer, where="RAG답변")
            korean_ok = True
        except Exception:
            korean_ok = False
        out.append({
            "q": case["q"], "tickers": case["tickers"], "answer": answer,
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "error": err, "korean_ok": korean_ok,
        })
    return out


# =========================================================

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="결과 파일 이름에 쓸 라벨 (예: 9b, 35b)")
    ap.add_argument("--only", choices=["terms", "quiz", "rag"], help="한 영역만 실행")
    ap.add_argument("--repeat", type=int, default=1, help="퀴즈 세트 반복 횟수")
    args = ap.parse_args()

    sections = [args.only] if args.only else ["terms", "quiz", "rag"]

    report = {
        "tag": args.tag,
        "backend": settings.LLM_BACKEND,
        "model": settings.LLM_MODEL,
        "base_url": (settings.LLM_BASE_URL if settings.LLM_BACKEND == "openai"
                     else settings.OLLAMA_BASE_URL),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    print(f"백엔드={report['backend']}  모델={report['model']}  URL={report['base_url']}\n")

    for sec in sections:
        print(f"[{sec}] 시작")
        t0 = time.perf_counter()
        try:
            if sec == "terms":
                report["terms"] = await run_terms()
            elif sec == "quiz":
                report["quiz"] = await run_quiz(args.repeat)
            else:
                report["rag"] = await run_rag()
        except Exception as e:
            report[sec] = {"fatal": f"{type(e).__name__}: {e}"}
            print(f"[{sec}] 중단: {e}")
        print(f"[{sec}] 완료 ({round(time.perf_counter() - t0, 1)}s)\n")

    out_dir = ROOT / "compare_out"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"compare_{args.tag}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 요약
    if isinstance(report.get("terms"), list):
        ok = sum(1 for r in report["terms"] if r.get("auto") and r["auto"]["pass"])
        print(f"용어 자동채점: {ok}/{len(report['terms'])} 통과")
    if isinstance(report.get("quiz"), list):
        ok = sum(1 for r in report["quiz"] if not r["error"])
        print(f"퀴즈 생성 성공: {ok}/{len(report['quiz'])}")
    if isinstance(report.get("rag"), list):
        ok = sum(1 for r in report["rag"] if not r["error"])
        print(f"RAG 응답 성공: {ok}/{len(report['rag'])}")
    print(f"\n저장: {path}")


if __name__ == "__main__":
    asyncio.run(main())