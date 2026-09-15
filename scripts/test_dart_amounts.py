"""공시 본문의 금액 단위 환산과 저정보 공시 중복 접힘에 대한 회귀 테스트.

네트워크·LLM 없이 순수 함수만 검사한다.

    uv run python scripts/test_dart_amounts.py

배경: 공시는 한 문서 안에서도 단위가 섞인다. 단위 선언 하나를 문서 전체에 적용하면
주식 수에 원화가 붙거나(712,702,365주 -> "약 712.70조원"), 자기 단위를 밝힌 칸이
100만 배로 부풀려진다(43,140,750,000,000원 -> "약 43,140,750조원").
컨텍스트에 "그대로 인용할 것"이 함께 들어가므로 LLM 이 이 값을 사실로 인용한다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.external.dart import annotate_amounts, format_krw_human
from app.services.rag.ingestion import _disclosure_doc_id


# (이름, 입력, 반드시 있어야 할 것, 절대 있으면 안 되는 것)
AMOUNT_CASES = [
    (
        "금액 칸과 주식 수 칸이 한 표에 섞여 있다",
        "단위 : 백만원 매출액 79,318,746 영업이익 12,345,678 "
        "발행주식총수 (주) 712,702,365 보통주식 (주) 17,790,000",
        ["79,318,746(약 79.32조원)", "12,345,678(약 12.35조원)"],
        ["712,702,365(약", "17,790,000(약"],
    ),
    (
        "칸 라벨이 스스로 원 단위를 밝히면 문서 선언보다 우선한다",
        "단위 : 백만원 요약재무정보 자산총계 1,234,567 "
        "4. 자금조달의 목적- 시설자금(원) 43,140,750,000,000",
        ["1,234,567(약 1.23조원)"],
        ["43,140,750,000,000(약"],
    ),
    (
        "단위 선언보다 앞에 있는 숫자는 그 선언의 지배를 받지 않는다",
        "4. 자금조달의 목적- 시설자금(원) 43,140,750,000,000 …… "
        "자산 단위 : 백만원 매출액 9,876,543",
        ["9,876,543(약 9.88조원)"],
        ["43,140,750,000,000(약"],
    ),
    (
        "숫자 바로 뒤의 '주' 표기도 수량 신호로 본다",
        "단위 : 백만원 취득금액 1,000,000 보유수량 1,289,120 주1)",
        ["1,000,000(약 1.00조원)"],
        ["1,289,120(약"],
    ),
    (
        "한 행에서 값이 반복되고 사이에 빈 칸('-')이 끼어도 라벨을 찾는다",
        "단위 : 백만원 취득금액 수량 비율 금 액 보통주식 1,625,769 0.2 92,683 "
        "- - - 1,625,769 0.2 92,683",
        [],
        ["1,625,769(약"],
    ),
    (
        "단위 선언이 없으면 아무것도 붙이지 않는다",
        "매출액 79,318,746 영업이익 12,345,678",
        [],
        ["(약"],
    ),
    (
        "금액 표가 아니면 건드리지 않는다",
        "단위 : 백만원 일련번호 12,345,678 관리번호 87,654,321",
        [],
        ["(약"],
    ),
]


def check_amounts() -> int:
    failed = 0
    for name, src, must, must_not in AMOUNT_CASES:
        out = annotate_amounts(src)
        missing = [s for s in must if s not in out]
        leaked = [s for s in must_not if s in out]
        if missing or leaked:
            failed += 1
            print(f"  FAIL  {name}")
            if missing:
                print(f"        빠짐: {missing}")
            if leaked:
                print(f"        붙으면 안 되는데 붙음: {leaked}")
            print(f"        결과: {out}")
        else:
            print(f"  PASS  {name}")
    return failed


def check_format() -> int:
    failed = 0
    for amount, expected in [
        (306_220_000_000_000, "약 306.22조원"),
        (1_500_000_000, "약 15.0억원"),
        (-2_000_000_000_000, "약 -2.00조원"),
        (12_345, "12,345원"),
        ("not a number", ""),
    ]:
        got = format_krw_human(amount)
        if got != expected:
            failed += 1
            print(f"  FAIL  format_krw_human({amount!r}) -> {got!r} (기대 {expected!r})")
        else:
            print(f"  PASS  format_krw_human({amount!r}) -> {got!r}")
    return failed


def check_dedup() -> int:
    """같은 날 여러 임원이 제출한 저정보 공시는 한 문서로 접혀야 한다."""
    failed = 0
    low_info = [
        {"title": "임원ㆍ주요주주특정증권등소유상황보고서", "date": "20260721",
         "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo=2026072100{i:04d}"}
        for i in range(5)
    ]
    ids = {_disclosure_doc_id("삼성전자", item) for item in low_info}
    if len(ids) != 1:
        failed += 1
        print(f"  FAIL  저정보 공시 5건이 {len(ids)}개 문서로 남음 (1개여야 함)")
    else:
        print("  PASS  같은 (종목·보고서명·날짜) 저정보 공시 5건 -> 문서 1건")

    # 날짜가 다르면 접히면 안 된다
    other_day = dict(low_info[0], date="20260722", url="https://dart.fss.or.kr/x?rcpNo=1")
    if _disclosure_doc_id("삼성전자", other_day) in ids:
        failed += 1
        print("  FAIL  날짜가 다른 저정보 공시까지 접혔다")
    else:
        print("  PASS  날짜가 다르면 별개 문서로 남는다")

    # 종목이 다르면 접히면 안 된다
    if _disclosure_doc_id("SK하이닉스", low_info[0]) in ids:
        failed += 1
        print("  FAIL  종목이 다른 저정보 공시까지 접혔다")
    else:
        print("  PASS  종목이 다르면 별개 문서로 남는다")

    # 일반 공시는 접수번호별로 그대로 남아야 한다
    normal = [
        {"title": "주요사항보고서(자기주식취득결정)", "date": "20260903",
         "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo=2026090300{i:04d}"}
        for i in range(3)
    ]
    normal_ids = {_disclosure_doc_id("대창단조", item) for item in normal}
    if len(normal_ids) != 3:
        failed += 1
        print(f"  FAIL  일반 공시 3건이 {len(normal_ids)}개로 접혔다 (3개여야 함)")
    else:
        print("  PASS  일반 공시는 접수번호별로 유지된다")
    return failed


def main() -> int:
    print("[단위 환산]")
    failed = check_amounts()
    print("\n[금액 표기]")
    failed += check_format()
    print("\n[저정보 공시 접힘]")
    failed += check_dedup()

    print()
    if failed:
        print(f"실패 {failed}건")
        return 1
    print("전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
