"""기존에 '제목만' 저장된 공시(dart) 문서를 삭제해, 다음 질의 때 본문 포함으로 재적재되게 한다.

공시 본문 적재 기능이 추가되기 전에 쌓인 dart 문서는 제목만 있어 LLM이 '공시 내용'을
지어낼 수 있다(예: 자기주식처분결정을 'IR 개최+잠정실적'으로 오설명). 이 스크립트로
source=dart 포인트를 지우면, 이후 해당 종목을 질문할 때 원문 본문과 함께 다시 적재된다.
(뉴스/재무 문서는 건드리지 않는다)

사용:
    uv run python scripts/reingest_disclosures.py            # 모든 종목의 공시 삭제
    uv run python scripts/reingest_disclosures.py 삼성전자    # 특정 종목만

주의: 서버가 실행 중이면 30분 in-memory 캐시 때문에 즉시 재수집이 안 될 수 있다.
     이 스크립트 실행 후 서버를 재시작하면 캐시가 초기화돼 확실하게 재적재된다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
from app.services.rag.vectorstore import get_qdrant_client
from app.core.config import settings


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else None

    must = [FieldCondition(key="metadata.source", match=MatchValue(value="dart"))]
    if ticker:
        must.append(FieldCondition(key="metadata.ticker", match=MatchValue(value=ticker)))
    flt = Filter(must=must)

    client = get_qdrant_client()
    try:
        cnt = client.count(
            collection_name=settings.QDRANT_COLLECTION, count_filter=flt, exact=True
        ).count
    except Exception as e:
        print(f"컬렉션 조회 실패 (Qdrant 실행 중인지 확인): {e}")
        return

    if cnt == 0:
        print(f"삭제할 공시 문서가 없습니다 (source=dart{f', ticker={ticker}' if ticker else ''}).")
        return

    client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=FilterSelector(filter=flt),
    )
    scope = f"ticker={ticker}" if ticker else "전체 종목"
    print(f"삭제 완료: source=dart, {scope} → {cnt}건 제거")
    print("이제 해당 종목을 다시 질문하면 공시 '본문'과 함께 재적재됩니다.")
    print("서버가 실행 중이었다면 재시작하세요(30분 캐시 초기화 → 즉시 재수집).")


if __name__ == "__main__":
    main()
