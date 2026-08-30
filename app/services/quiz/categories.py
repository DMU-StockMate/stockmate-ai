"""퀴즈 카테고리/상세 코드 카탈로그 (DB quiz_categories / quiz_category_details 미러).

AI 서버는 DB 연결이 없으므로, 프롬프트 기반 문제 생성 시 매핑에 사용할
카테고리/상세 코드 목록을 여기 상수로 둔다 (팀 결정: 요청으로 받지 않고 하드코딩).

⚠️ 주의: 이 데이터는 stockmate_improved_full.sql의 초기 데이터와 동일해야 한다.
DB에서 카테고리를 추가/수정/삭제하면 이 파일도 같이 수정할 것.
코드가 어긋나면 NestJS가 저장할 때 FK 조회에 실패한다.
"""
from app.schemas.quiz import QuizCategoryIn, QuizCategoryDetailIn


def _cat(level: str, code: str, name: str, desc: str, direction: str, details: list[tuple[str, str, str]]) -> QuizCategoryIn:
    return QuizCategoryIn(
        investment_level=level,
        category_code=code,
        category_name=name,
        description=desc,
        problem_direction=direction,
        details=[
            QuizCategoryDetailIn(detail_code=d_code, detail_name=d_name, description=d_desc)
            for d_code, d_name, d_desc in details
        ],
    )


# 사용자 investment_level → 노출할 카테고리 등급.
# DB에는 입문용 카테고리가 없어서 입문은 초급 카테고리를 사용한다.
# 미설정은 매핑되는 등급이 없으므로 전체 카탈로그를 사용한다 (get_catalog_for_level 참고).
_LEVEL_TO_CATALOG_LEVEL = {
    "입문": "초급",
    "초급": "초급",
    "중급": "중급",
    "고급": "고급",
}


def get_catalog_for_level(investment_level: str) -> "list[QuizCategoryIn]":
    """사용자 수준에 해당하는 카테고리만 반환한다. 매핑 불가(미설정 등)면 전체 반환.

    초급 유저의 문제가 중급/고급 카테고리로 분류되는 것을 막기 위해,
    LLM에게 애초에 해당 등급의 카테고리만 보여준다 (프롬프트 지시보다 확실함).
    """
    catalog_level = _LEVEL_TO_CATALOG_LEVEL.get(investment_level)
    if not catalog_level:
        return QUIZ_CATEGORY_CATALOG
    return [c for c in QUIZ_CATEGORY_CATALOG if c.investment_level == catalog_level]


# =========================================================
# 카탈로그 조회 (문제 생성 프롬프트에 넣을 설명·출제 방향을 꺼내는 헬퍼)
# =========================================================

def get_category(category_code: str | None) -> "QuizCategoryIn | None":
    """category_code로 카테고리를 찾는다. 없으면 None."""
    if not category_code:
        return None
    return next(
        (c for c in QUIZ_CATEGORY_CATALOG if c.category_code == category_code), None
    )


def get_detail_description(category_code: str | None, detail_code: str | None) -> str:
    """상세 코드의 설명을 돌려준다. 없으면 빈 문자열.

    문제 생성 프롬프트의 주제 힌트로는 **카테고리 설명이 아니라 상세 설명**을 쓴다.
    카테고리 설명은 형제 주제를 나열하므로("PER, ROE, EPS, PBR"),
    topic이 PER일 때 그대로 넣으면 오히려 ROE로 새라고 부추기는 꼴이 된다.
    상세 설명("주가수익비율의 의미")은 그 주제 하나로 범위를 좁혀준다.
    """
    cat = get_category(category_code)
    if not cat or not detail_code:
        return ""
    detail = next((d for d in cat.details if d.detail_code == detail_code), None)
    return (detail.description or "") if detail else ""


def get_problem_direction(category_code: str | None) -> str:
    """카테고리의 출제 방향("주식 용어 이해" 등). 없으면 빈 문자열."""
    cat = get_category(category_code)
    return (cat.problem_direction or "") if cat else ""


def _norm(text: str) -> str:
    """공백을 지워 "볼린저 밴드"와 "볼린저밴드"를 같게 본다."""
    return "".join(text.split())


def find_topic_context(topic: str, investment_level: str = "") -> tuple[str, str]:
    """주제명만 아는 경우(/quiz/generate)에 카탈로그에서 문맥을 찾아준다.

    반환: (주제 설명, 출제 방향). 못 찾으면 ("", "").

    프롬프트 기반 생성(/quiz/generate/prompt)은 LLM이 category_code를 정해주지만,
    /quiz/generate는 topic 문자열만 받는다. 그 경로에서도 카테고리의 출제 방향을
    쓸 수 있도록 이름으로 결정론적으로 매칭한다 (LLM 호출 없음).

    사용자 등급의 카탈로그를 먼저 보고, 없으면 전체 카탈로그에서 찾는다.
    같은 이름의 상세가 여러 등급에 있을 때(예: "RSI") 사용자 등급 것을 고르기 위함이다.
    """
    if not topic:
        return "", ""

    target = _norm(topic)
    level_catalog = get_catalog_for_level(investment_level) if investment_level else []
    # 등급 카탈로그 -> 전체 카탈로그 순으로 본다 (앞에서 찾으면 거기서 멈춤)
    for catalog in (level_catalog, QUIZ_CATEGORY_CATALOG):
        for cat in catalog:
            for d in cat.details:
                if _norm(d.detail_name) == target:
                    return d.description or "", cat.problem_direction or ""
        # 상세에 없으면 카테고리 이름으로 찾는다.
        # 이 경우엔 카테고리 자체가 주제이므로 형제 나열(description)이 오히려 정확하다.
        for cat in catalog:
            if _norm(cat.category_name) == target:
                return cat.description or "", cat.problem_direction or ""
    return "", ""


QUIZ_CATEGORY_CATALOG: list[QuizCategoryIn] = [
    # ----- 초급 (investment_level_id=1) -----
    _cat("초급", "BEGINNER_STOCK_BASIC", "주식 기본 개념",
         "주식, 주가, 시가총액, 배당, 거래량", "주식 용어 이해", [
        ("STOCK", "주식", "기업의 소유권 일부를 나타내는 증권"),
        ("STOCK_PRICE", "주가", "시장에서 거래되는 주식의 가격"),
        ("MARKET_CAP", "시가총액", "주가와 발행주식 수를 곱한 기업 가치 지표"),
        ("DIVIDEND", "배당", "기업 이익을 주주에게 분배하는 것"),
        ("VOLUME_BASIC", "거래량", "일정 기간 거래된 주식 수량"),
    ]),
    _cat("초급", "BEGINNER_FINANCIAL_BASIC", "재무지표 기초",
         "PER, ROE, EPS, PBR", "지표가 의미하는 바 이해", [
        ("PER_BASIC", "PER", "주가수익비율의 의미"),
        ("ROE_BASIC", "ROE", "자기자본이익률의 의미"),
        ("EPS_BASIC", "EPS", "주당순이익의 의미"),
        ("PBR_BASIC", "PBR", "주가순자산비율의 의미"),
    ]),
    _cat("초급", "BEGINNER_CHART_BASIC", "차트 기초",
         "상승, 하락, 횡보, 거래량 증가 및 감소", "차트 흐름을 단순 해석", [
        ("UPTREND_BASIC", "상승", "주가가 전반적으로 오르는 흐름"),
        ("DOWNTREND_BASIC", "하락", "주가가 전반적으로 내리는 흐름"),
        ("SIDEWAYS_BASIC", "횡보", "주가가 일정 범위에서 움직이는 흐름"),
        ("VOLUME_CHANGE_BASIC", "거래량 증가/감소", "거래량 변화의 기본 의미"),
    ]),
    _cat("초급", "BEGINNER_FINANCIAL_PRODUCT", "금융상품 기초",
         "예금, 적금, 채권, 펀드, ETF", "상품별 특징 구분", [
        ("DEPOSIT", "예금", "일정 금액을 맡기고 이자를 받는 상품"),
        ("INSTALLMENT_SAVINGS", "적금", "일정 기간 금액을 나누어 납입하는 상품"),
        ("BOND", "채권", "정부나 기업에 자금을 빌려주고 이자를 받는 증권"),
        ("FUND", "펀드", "여러 투자자의 자금을 모아 운용하는 상품"),
        ("ETF", "ETF", "거래소에서 주식처럼 거래되는 펀드"),
    ]),
    _cat("초급", "BEGINNER_RISK_BASIC", "투자 위험 기초",
         "원금 손실, 변동성, 분산투자", "투자 위험 이해", [
        ("PRINCIPAL_LOSS", "원금 손실", "투자 원금이 줄어들 수 있는 위험"),
        ("VOLATILITY", "변동성", "가격이 오르내리는 정도"),
        ("DIVERSIFICATION", "분산투자", "여러 자산에 나누어 투자하여 위험을 줄이는 방법"),
    ]),
    # ----- 중급 (investment_level_id=2) -----
    _cat("중급", "INTERMEDIATE_FINANCIAL_COMBINATION", "재무지표 조합 판단",
         "PER + ROE, 매출 성장률, 영업이익률", "기업의 수익성과 성장성 판단", [
        ("PER_ROE_COMBINATION", "PER + ROE", "가격 수준과 수익성을 함께 판단"),
        ("REVENUE_GROWTH_RATE", "매출 성장률", "매출 증가 추세 판단"),
        ("OPERATING_MARGIN", "영업이익률", "본업의 수익성 판단"),
    ]),
    _cat("중급", "INTERMEDIATE_CHART_ANALYSIS", "차트 분석 기초",
         "추세, 지지선, 저항선, 거래량 변화", "주가 흐름 해석", [
        ("TREND", "추세", "주가의 전반적인 진행 방향"),
        ("SUPPORT_LINE", "지지선", "주가 하락이 지지될 가능성이 있는 가격대"),
        ("RESISTANCE_LINE", "저항선", "주가 상승이 제한될 가능성이 있는 가격대"),
        ("VOLUME_CHANGE", "거래량 변화", "거래량 증감과 주가 흐름 해석"),
    ]),
    _cat("중급", "INTERMEDIATE_INDICATOR_CONCEPT", "보조지표 개념",
         "이동평균선, RSI, MACD, 볼린저밴드", "보조지표의 의미와 역할 이해", [
        ("MOVING_AVERAGE_CONCEPT", "이동평균선", "일정 기간 평균 주가를 연결한 지표"),
        ("RSI_CONCEPT", "RSI", "과매수와 과매도 수준을 판단하는 지표"),
        ("MACD_CONCEPT", "MACD", "추세와 모멘텀 변화를 파악하는 지표"),
        ("BOLLINGER_BAND_CONCEPT", "볼린저밴드", "가격 변동 범위를 나타내는 지표"),
    ]),
    _cat("중급", "INTERMEDIATE_NEWS_ANALYSIS", "뉴스 해석",
         "호재, 악재, 시장 반응", "뉴스가 주가에 미치는 영향 이해", [
        ("POSITIVE_NEWS", "호재", "주가에 긍정적으로 작용할 수 있는 뉴스"),
        ("NEGATIVE_NEWS", "악재", "주가에 부정적으로 작용할 수 있는 뉴스"),
        ("MARKET_REACTION", "시장 반응", "뉴스 발표 후 실제 주가와 거래량의 반응"),
    ]),
    _cat("중급", "INTERMEDIATE_DISCLOSURE_ANALYSIS", "공시 해석",
         "실적 발표, 계약 공시, 유상증자, 배당", "공시 내용을 투자 관점으로 해석", [
        ("EARNINGS_DISCLOSURE", "실적 발표", "매출과 이익 등 경영 성과 공시"),
        ("CONTRACT_DISCLOSURE", "계약 공시", "공급계약 및 수주 관련 공시"),
        ("RIGHTS_OFFERING", "유상증자", "신주 발행으로 자본을 조달하는 공시"),
        ("DIVIDEND_DISCLOSURE", "배당", "배당금 또는 배당정책 관련 공시"),
    ]),
    _cat("중급", "INTERMEDIATE_INDUSTRY_COMPARISON", "업종 비교",
         "같은 업종 내 기업 비교", "상대적으로 나은 기업 판단", [
        ("SAME_INDUSTRY_COMPARISON", "같은 업종 내 기업 비교", "동일 업종 기업의 지표와 성과를 비교"),
    ]),
    _cat("중급", "INTERMEDIATE_INVESTOR_PROFILE", "투자 성향별 판단",
         "안정형, 성장형, 공격형", "성향에 맞는 투자 선택", [
        ("CONSERVATIVE_PROFILE", "안정형", "낮은 변동성과 원금 보전을 중시하는 성향"),
        ("GROWTH_PROFILE", "성장형", "성장 가능성과 적정 위험을 함께 고려하는 성향"),
        ("AGGRESSIVE_PROFILE", "공격형", "높은 위험을 감수하고 높은 수익을 추구하는 성향"),
    ]),
    # ----- 고급 (investment_level_id=3) -----
    _cat("고급", "ADVANCED_HISTORICAL_DECISION", "과거 시점 투자 판단",
         "특정 날짜 기준 차트, 재무, 뉴스 제공", "당시 기준으로 매수, 보유, 매도 판단", [
        ("HISTORICAL_CHART", "특정 날짜 기준 차트", "기준 날짜까지의 차트 데이터"),
        ("HISTORICAL_FINANCIAL", "특정 날짜 기준 재무", "기준 날짜 당시 확인 가능한 재무 데이터"),
        ("HISTORICAL_NEWS", "특정 날짜 기준 뉴스", "기준 날짜 당시 공개된 뉴스"),
    ]),
    _cat("고급", "ADVANCED_INDICATOR_DECISION", "보조지표 활용 판단",
         "RSI, MACD, 이동평균선, 거래량, 볼린저밴드", "기술적 지표를 활용한 투자 판단", [
        ("RSI_DECISION", "RSI", "RSI를 실제 투자 판단에 활용"),
        ("MACD_DECISION", "MACD", "MACD를 실제 투자 판단에 활용"),
        ("MOVING_AVERAGE_DECISION", "이동평균선", "이동평균선을 실제 투자 판단에 활용"),
        ("VOLUME_DECISION", "거래량", "거래량을 실제 투자 판단에 활용"),
        ("BOLLINGER_BAND_DECISION", "볼린저밴드", "볼린저밴드를 실제 투자 판단에 활용"),
    ]),
    _cat("고급", "ADVANCED_INVESTMENT_STRATEGY", "종합 투자 전략",
         "가치투자, 성장주 투자, 모멘텀 투자", "전략에 맞는 판단 선택", [
        ("VALUE_INVESTING", "가치투자", "기업 가치 대비 저평가 여부를 중시하는 전략"),
        ("GROWTH_INVESTING", "성장주 투자", "미래 성장성과 실적 증가를 중시하는 전략"),
        ("MOMENTUM_INVESTING", "모멘텀 투자", "가격과 수급의 상승 흐름을 중시하는 전략"),
    ]),
    _cat("고급", "ADVANCED_RISK_ANALYSIS", "리스크 분석",
         "고평가, 악재 공시, 실적 악화, 시장 하락", "투자 위험 요소 파악", [
        ("OVERVALUATION", "고평가", "기업 가치 대비 주가가 높을 가능성"),
        ("NEGATIVE_DISCLOSURE", "악재 공시", "투자 판단에 부정적인 공시"),
        ("EARNINGS_DETERIORATION", "실적 악화", "매출 또는 이익의 감소"),
        ("MARKET_DECLINE", "시장 하락", "전체 시장 약세로 발생하는 위험"),
    ]),
    _cat("고급", "ADVANCED_PORTFOLIO_DECISION", "포트폴리오 판단",
         "여러 종목 또는 금융상품 조합", "자산 배분 판단", [
        ("MULTI_STOCK_PORTFOLIO", "여러 종목 조합", "여러 종목으로 구성된 포트폴리오"),
        ("FINANCIAL_PRODUCT_ALLOCATION", "금융상품 조합", "주식 외 금융상품을 포함한 자산 배분"),
    ]),
    _cat("고급", "ADVANCED_EVENT_DECISION", "이벤트 기반 판단",
         "실적 발표, 금리 변화, 정책 이슈", "이벤트 전후 투자 판단", [
        ("EARNINGS_EVENT", "실적 발표", "실적 발표 전후의 투자 판단"),
        ("INTEREST_RATE_CHANGE", "금리 변화", "금리 변화가 시장과 기업에 미치는 영향"),
        ("POLICY_ISSUE", "정책 이슈", "정부 정책 및 규제 변화의 영향"),
    ]),
    _cat("고급", "ADVANCED_EVIDENCE_SELECTION", "투자 근거 선택",
         "차트, PER, ROE, 뉴스, 공시, 보조지표", "가장 타당한 근거 선택", [
        ("CHART_EVIDENCE", "차트", "가격과 거래량 흐름을 투자 근거로 활용"),
        ("PER_EVIDENCE", "PER", "PER을 투자 근거로 활용"),
        ("ROE_EVIDENCE", "ROE", "ROE를 투자 근거로 활용"),
        ("NEWS_EVIDENCE", "뉴스", "뉴스를 투자 근거로 활용"),
        ("DISCLOSURE_EVIDENCE", "공시", "공시를 투자 근거로 활용"),
        ("INDICATOR_EVIDENCE", "보조지표", "기술적 보조지표를 투자 근거로 활용"),
    ]),
]
