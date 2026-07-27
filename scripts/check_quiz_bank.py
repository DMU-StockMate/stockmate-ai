"""퀴즈 뱅크 상태 확인 + 중복 임계값 튜닝 도우미.

QUIZ_DUP_THRESHOLD 를 감으로 정하면 너무 높아서 중복을 못 잡거나
너무 낮아서 멀쩡한 문제까지 폐기한다. 실제로 저장된 문제들의
유사도 분포를 보고 정하는 게 맞다.

사용법:
    # 저장된 문제 목록과 유사도 분포 확인
    uv run python scripts/check_quiz_bank.py

    # 특정 문장이 기존 문제와 얼마나 비슷한지 확인
    uv run python scripts/check_quiz_bank.py "PER을 계산할 때 분모는 무엇인가"

    # 컬렉션 비우기
    uv run python scripts/check_quiz_bank.py --clear
"""
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.services.rag.vectorstore import get_embeddings, get_qdrant_client  # noqa: E402

COLLECTION = settings.QUIZ_BANK_COLLECTION


def _cosine(a, b):
    # bge-m3는 normalize_embeddings=True 라 내적이 곧 코사인 유사도
    return sum(x * y for x, y in zip(a, b))


def load_points():
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        print(f"컬렉션 '{COLLECTION}' 이 없습니다. 문제를 한 번 생성하면 만들어집니다.")
        return []

    points, _ = client.scroll(
        collection_name=COLLECTION, limit=500,
        with_payload=True, with_vectors=True,
    )
    return points


def show_overview(points):
    print(f"\n컬렉션: {COLLECTION}")
    print(f"저장된 문제: {len(points)}개")
    if not points:
        return

    by_owner = {}
    for p in points:
        owner = p.payload.get("owner_user_id", "?")
        by_owner.setdefault(owner, []).append(p)
    print("소유자별:", {k: len(v) for k, v in sorted(by_owner.items(), key=lambda x: str(x[0]))})

    print("\n--- 저장된 문제 ---")
    for i, p in enumerate(points, 1):
        payload = p.payload
        print(f"{i:>3}. [{payload.get('detail_code') or '-'}] {payload.get('question_text', '')[:60]}")


def show_similarity_distribution(points):
    """문제 쌍의 유사도 분포. 임계값을 어디에 둘지 판단하는 근거."""
    if len(points) < 2:
        print("\n(문제가 2개 미만이라 유사도 분석 생략)")
        return

    pairs = []
    for a, b in combinations(points, 2):
        score = _cosine(a.vector, b.vector)
        pairs.append((score, a.payload.get("question_text", ""), b.payload.get("question_text", "")))
    pairs.sort(reverse=True)

    print(f"\n--- 유사도 분포 (총 {len(pairs)}쌍) ---")
    buckets = [(0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85), (0.0, 0.80)]
    for lo, hi in buckets:
        n = sum(1 for s, _, _ in pairs if lo <= s < hi)
        bar = "#" * min(n, 50)
        label = f"{lo:.2f} 이상" if hi > 1 else f"{lo:.2f}~{hi:.2f}"
        print(f"  {label:>12} : {n:>4}쌍 {bar}")

    print(f"\n--- 가장 비슷한 쌍 5개 (현재 임계값 {settings.QUIZ_DUP_THRESHOLD}) ---")
    for score, t1, t2 in pairs[:5]:
        mark = "[걸림]" if score >= settings.QUIZ_DUP_THRESHOLD else "[통과]"
        print(f"\n  {mark} 유사도 {score:.3f}")
        print(f"    A: {t1[:70]}")
        print(f"    B: {t2[:70]}")

    top = pairs[0][0]
    print("\n--- 판단 ---")
    if top < settings.QUIZ_DUP_THRESHOLD:
        print(f"  가장 비슷한 쌍도 {top:.3f} 으로 임계값 {settings.QUIZ_DUP_THRESHOLD} 미만입니다.")
        print(f"  위 '가장 비슷한 쌍'을 눈으로 보고 실제로 중복이라 느껴지면")
        print(f"  QUIZ_DUP_THRESHOLD 를 {max(top - 0.02, 0.5):.2f} 근처로 낮추세요.")
    else:
        n_caught = sum(1 for s, _, _ in pairs if s >= settings.QUIZ_DUP_THRESHOLD)
        print(f"  현재 임계값이면 {n_caught}쌍이 중복으로 걸립니다.")


def query(text):
    points = load_points()
    if not points:
        return
    vector = get_embeddings().embed_query(text)
    scored = sorted(
        ((_cosine(vector, p.vector), p.payload.get("question_text", "")) for p in points),
        reverse=True,
    )
    print(f"\n입력: {text}")
    print(f"임계값: {settings.QUIZ_DUP_THRESHOLD}\n")
    for score, t in scored[:10]:
        mark = "[중복]" if score >= settings.QUIZ_DUP_THRESHOLD else "      "
        print(f"  {mark} {score:.3f}  {t[:65]}")


def clear():
    client = get_qdrant_client()
    if COLLECTION in [c.name for c in client.get_collections().collections]:
        client.delete_collection(COLLECTION)
        print(f"컬렉션 '{COLLECTION}' 삭제 완료")
    else:
        print(f"컬렉션 '{COLLECTION}' 이 없습니다")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--clear":
        clear()
    elif args:
        query(" ".join(args))
    else:
        pts = load_points()
        show_overview(pts)
        show_similarity_distribution(pts)
