"""카테고리 커버리지 점검 - 카탈로그의 모든 카테고리로 실제 문제가 생성되는지 확인한다.

학습 기준표(초/중/고급 × 카테고리 × 상세 주제)가 실제 생성 파이프라인에서
전부 동작하는지 보기 위한 스크립트다. 카테고리마다 대표 상세 주제 하나를 골라
문제를 1개 만들고, 실패하면 사유를 남긴다.

실행:
    python scripts/test_category_coverage.py            # 전체 등급
    python scripts/test_category_coverage.py 고급        # 특정 등급만
    python scripts/test_category_coverage.py 고급 --all  # 상세 주제 전부

주의: LLM을 실제로 호출한다. 카테고리 1개당 수 초 걸린다.
"""
import asyncio
import sys
import time
import traceback

sys.path.insert(0, ".")

from app.schemas.chat import UserContext
from app.services.quiz.categories import (
    QUIZ_CATEGORY_CATALOG, get_detail_description, get_problem_direction,
)
from app.services.quiz.generator import generate_quiz
from app.services.quiz.quality import get_llm  # noqa: F401  (설정 로딩 확인용)


LEVEL_TO_USER_LEVEL = {"초급": "초급", "중급": "중급", "고급": "고급"}


def _user(level: str) -> UserContext:
    return UserContext(user_id=99001, investment_level=level)


async def _one(cat, detail, quiz_type: str) -> dict:
    """카테고리 1개 + 상세 1개로 문제를 만든다. 결과 dict 반환."""
    topic = detail.detail_name
    topic_desc = get_detail_description(cat.category_code, detail.detail_code)
    direction = get_problem_direction(cat.category_code)
    user = _user(LEVEL_TO_USER_LEVEL.get(cat.investment_level, "미설정"))

    started = time.time()
    try:
        q = await generate_quiz(
            user, quiz_type, topic, topic_desc=topic_desc, direction=direction)
        return {
            "ok": True,
            "sec": time.time() - started,
            "text": q["question_text"],
            "type": q["question_type"],
            "n_choices": len(q["choices"]),
        }
    except Exception as e:  # noqa: BLE001 - 실패 사유를 그대로 보고한다
        return {
            "ok": False,
            "sec": time.time() - started,
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=2),
        }


async def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    all_details = "--all" in sys.argv
    quiz_type = "OX" if "--ox" in sys.argv else "MULTIPLE_CHOICE"
    want_level = args[0] if args else None

    cats = [c for c in QUIZ_CATEGORY_CATALOG
            if not want_level or c.investment_level == want_level]

    print(f"대상 카테고리 {len(cats)}개 / 유형 {quiz_type} / "
          f"{'상세 전부' if all_details else '카테고리당 대표 1개'}\n")

    results = []
    for cat in cats:
        details = cat.details if all_details else cat.details[:1]
        for d in details:
            r = await _one(cat, d, quiz_type)
            r["level"] = cat.investment_level
            r["category"] = cat.category_name
            r["code"] = cat.category_code
            r["topic"] = d.detail_name
            results.append(r)

            mark = "OK  " if r["ok"] else "FAIL"
            head = f"[{mark}] ({cat.investment_level}) {cat.category_name} / {d.detail_name}"
            print(f"{head}  ({r['sec']:.1f}s)")
            if r["ok"]:
                print(f"        {r['text']}")
            else:
                print(f"        {r['error']}")
            print()

    # 요약
    print("=" * 78)
    fails = [r for r in results if not r["ok"]]
    by_level: dict[str, list] = {}
    for r in results:
        by_level.setdefault(r["level"], []).append(r)
    for lvl, rs in by_level.items():
        ok = sum(1 for r in rs if r["ok"])
        print(f"{lvl}: {ok}/{len(rs)} 성공")
    print(f"전체: {len(results) - len(fails)}/{len(results)} 성공, "
          f"평균 {sum(r['sec'] for r in results) / max(len(results), 1):.1f}초")
    if fails:
        print("\n실패 목록:")
        for r in fails:
            print(f"  - ({r['level']}) {r['category']} / {r['topic']}: {r['error'][:110]}")


if __name__ == "__main__":
    asyncio.run(main())
