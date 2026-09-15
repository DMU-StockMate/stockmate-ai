"""공시 유형 분류(저정보 / 중요)에 대한 회귀 테스트.

네트워크·LLM 없이 순수 함수만 검사한다.

    uv run python scripts/test_dart_classify.py

배경: 검색 채택 순서는 최신순이라, 중요도를 표현하는 신호가 없으면 매일 나오는 형식적
공시가 잠정실적·중대재해를 밀어낸다. 실측에서 SK하이닉스 "중요한 공시" 질의의 채택 5건이
전부 조회공시요구·풍문해명 계열이었다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.external.dart import is_key_report, is_low_info_report

LOW_INFO = [
    "임원ㆍ주요주주특정증권등소유상황보고서",
    "[기재정정]임원ㆍ주요주주특정증권등소유상황보고서",
    "최대주주등소유주식변동신고서",
    "주식등의대량보유상황보고서(일반)",
    "조회공시요구(풍문또는보도)",                      # 요구 자체는 내용이 없다
    "조회공시요구(풍문또는보도)에대한답변(미확정)",      # 미확정 = 결정된 바 없음
    "풍문또는보도에대한해명(미확정)",
]

KEY = [
    "연결재무제표기준영업(잠정)실적(공정공시)",
    "반기보고서 (2026.06)",
    "사업보고서 (2025.12)",
    "주요사항보고서(자기주식취득신탁계약체결결정)",
    "주요사항보고서(자기주식처분결정)",
    "유상증자결정",
    "전환사채권발행결정",
    "단일판매ㆍ공급계약체결",
    "회사합병결정",
    "최대주주변경",
    "현금ㆍ현물배당결정",
    "중대재해발생",
    "매출액또는손익구조30%(대규모법인은15%)이상변동",
    "신규시설투자등",
    "타법인주식및출자증권취득결정",
]

NORMAL = [
    "기업설명회(IR)개최(안내공시)",
    "수시공시의무관련사항(공정공시)",
    "풍문또는보도에대한해명",        # 확정된 해명은 내용이 있다
    "특수관계인여신등",
    "기타경영사항(자율공시)",
]


def run(name, titles, expect_low, expect_key) -> int:
    failed = 0
    for title in titles:
        low, key = is_low_info_report(title), is_key_report(title)
        if low != expect_low or key != expect_key:
            failed += 1
            print(f"  FAIL  [{name}] {title}")
            print(f"        low_info={low}(기대 {expect_low}) key={key}(기대 {expect_key})")
        else:
            print(f"  PASS  [{name}] {title[:46]}")
    return failed


def main() -> int:
    failed = 0
    print("[저정보 공시 - 후순위로 밀려야 한다]")
    failed += run("저정보", LOW_INFO, expect_low=True, expect_key=False)
    print("\n[중요 공시 - 최우선으로 올라가야 한다]")
    failed += run("중요", KEY, expect_low=False, expect_key=True)
    print("\n[일반 공시 - 어느 쪽도 아니다]")
    failed += run("일반", NORMAL, expect_low=False, expect_key=False)

    print()
    if failed:
        print(f"실패 {failed}건")
        return 1
    print("전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
