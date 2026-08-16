"""주제 이탈 판정 (의존성 없음).

quality.py 에서 떼어낸 이유
  이 판정을 scripts/inspect_dataset.py 도 써야 하는데, quality.py 는
  langchain 을 import 한다. 그래서 스크립트가 판정 로직을 따로 구현했고,
  상위 개념 규칙("보조지표" 주제의 RSI 문제는 정상)을 몰라서 정상 채택분
  10건을 이탈로 오보했다. 같은 판단을 두 곳에 두면 반드시 어긋난다.

  이 모듈은 순수 문자열 처리만 하므로 어디서든 가볍게 import 할 수 있다.
"""


# 지정 주제를 벗어났는지 판단할 때 기준이 되는 개념어.
# 모델이 다른 개념을 주인공으로 삼으면 학습 데이터의 라벨이 틀어지므로 걸러낸다.
_MAJOR_CONCEPTS = (
    "PER", "PBR", "ROE", "EPS", "배당수익률", "시가총액",
    "RSI", "MACD", "볼린저밴드", "이동평균선", "부채비율", "영업이익률",
)


def _normalize(text: str) -> str:
    """공백을 제거해 "볼린저밴드" 와 "볼린저 밴드" 를 같게 본다."""
    return "".join(text.split())


def _topic_parts(topic: str) -> list[str]:
    """복합 주제를 구성 요소로 쪼갠다.

    "PER + ROE" 처럼 두 개념을 함께 다루는 주제는 그 문자열이 본문에
    그대로 나올 리 없다. 구성 요소 중 하나라도 등장하면 주제를 지킨 것으로 본다.
    """
    parts = [p.strip() for p in topic.replace("/", "+").split("+")]
    return [p for p in parts if p]


# 상위 개념 -> 그 개념에 속하는 하위 개념.
# 주제 자체가 상위 개념이면 하위 개념으로 문제를 내는 것이 정상이므로 이탈이 아니다.
#
# 실측 사고: 주제 "보조지표"(설명 "기술적 보조지표를 투자 근거로 활용")에
# 볼린저밴드 문제를 낸 것을 이탈로 판정해 그 주제를 통째로 날렸다.
# 볼린저밴드는 보조지표의 한 종류이므로 정답에 가까운 출제였다.
#
# 앞에 오는 항목부터 검사하고 처음 맞는 것에서 멈춘다.
# "보조지표"가 "지표"보다 먼저 와야 한다 - 뒤에 두면 "지표"에 먼저 걸려
# 보조지표 주제에 PER 문제가 들어오는 것까지 허용해버린다.
_TECHNICAL = ("RSI", "MACD", "볼린저밴드", "이동평균선")
_FUNDAMENTAL = ("PER", "PBR", "ROE", "EPS", "배당수익률",
                "시가총액", "부채비율", "영업이익률")

_CONCEPT_MEMBERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("보조지표", _TECHNICAL),
    ("기술적", _TECHNICAL),
    ("차트", _TECHNICAL),
    ("재무지표", _FUNDAMENTAL),
    # "지표를 비교" 처럼 지표 사용 자체가 주제인 경우 (예: 같은 업종 내 기업 비교).
    # 어떤 지표를 쓰든 주제를 지킨 것이다.
    ("지표", _TECHNICAL + _FUNDAMENTAL),
    ("비교", _TECHNICAL + _FUNDAMENTAL),
)


# 같은 개념의 다른 이름. 약어만 세면 "주가수익비율" 로 쓴 문제를 놓친다.
_CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "PER": ("PER", "주가수익비율"),
    "PBR": ("PBR", "주가순자산비율"),
    "ROE": ("ROE", "자기자본이익률"),
    "EPS": ("EPS", "주당순이익"),
}


def _aliases(concept: str) -> tuple[str, ...]:
    return _CONCEPT_ALIASES.get(concept, (concept,))


# 어떤 개념을 설명하려면 반드시 따라 나오는 구성 요소.
# 주제가 PER 이면 분모인 EPS 는 몇 번이 나와도 이탈이 아니다.
# (실측 오탐: "PER 은 주가를 주당순이익으로 나눈 값" 문제가 EPS 2회 때문에 걸렸다)
_CONCEPT_COMPONENTS: dict[str, tuple[str, ...]] = {
    "PER": ("EPS",),
    "PBR": ("EPS",),
    "ROE": ("EPS",),
    "배당수익률": (),
    "시가총액": (),
}


def _allowed_concepts(scope: str) -> set[str]:
    """주제·설명 문자열을 보고 등장해도 되는 개념어를 정한다."""
    normalized = _normalize(scope)
    for umbrella, members in _CONCEPT_MEMBERS:
        if _normalize(umbrella) in normalized:
            return set(members)
    return set()


def assert_on_topic(question_text: str, topic: str, topic_desc: str = "") -> None:
    """지정한 주제를 벗어나 다른 개념 문제를 만들었으면 AssertionError.

    프롬프트로 "주제를 벗어나지 마세요"라고 해도 계속 샜다(실측 이탈률 32%).
    특히 "주식", "주가" 처럼 범위가 넓은 주제에서 모델이 PER 로 흘러갔다.
    detail_code 는 STOCK 인데 내용이 PER 이면 라벨이 틀린 학습 데이터가 되므로
    결정론적으로 막는다.

    판정은 관대하게 한다. 오탐은 정상 데이터를 negative 로 잘못 라벨링해
    "옳은 출제를 피하라"고 가르치는 셈이 되므로, 놓치는 것보다 해롭다.
      1. 지정 주제어가 본문에 있으면 통과
      2. 주제가 상위 개념이면 그에 속한 하위 개념 등장도 통과
      3. 그 외에 다른 개념이 등장할 때만 이탈로 본다
    (주제어도 없고 다른 개념도 없는 경우는 판단 근거가 없으므로 통과)

    topic_desc 는 카탈로그의 주제 설명이다. 주제명만으로는 범위를 알 수 없는
    경우가 많아("보조지표", "같은 업종 내 기업 비교") 함께 받는다.
    """
    if not topic or not question_text:
        return

    # 공백 차이("볼린저 밴드" vs "볼린저밴드")로 오탐이 나지 않게 정규화한다
    body = _normalize(question_text)
    parts = [_normalize(p) for p in _topic_parts(topic)]
    allowed = _allowed_concepts(f"{topic} {topic_desc}")

    # 주제 개념을 설명할 때 딸려 나올 수밖에 없는 구성 요소는 경쟁자로 보지 않는다
    for part in parts:
        for concept, components in _CONCEPT_COMPONENTS.items():
            if _normalize(concept) == part:
                allowed = allowed | set(components)

    rivals: dict[str, int] = {}
    for concept in _MAJOR_CONCEPTS:
        if concept in allowed:
            continue
        # 주제와 포함 관계면 경쟁 개념이 아니다.
        # "배당" 주제의 "배당수익률", "PER" 주제의 "PER" 등.
        if any(_normalize(concept) in p or p in _normalize(concept) for p in parts):
            continue
        hits = max(body.count(_normalize(a)) for a in _aliases(concept))
        if hits:
            rivals[concept] = hits

    if not rivals:
        return

    # 주제어를 셀 때는 경쟁 개념의 이름을 먼저 지운다.
    # PER 의 정의가 "주가/주당순이익"이라 PER 문제에는 "주가"가 반드시 나온다.
    # 지우지 않으면 주제 "주가"가 PER 문제를 자기 것으로 착각한다(실측 채택분의 25%).
    masked = body
    for concept in rivals:
        for alias in _aliases(concept):
            masked = masked.replace(_normalize(alias), "")
    topic_hits = sum(masked.count(p) for p in parts)

    # 주인공 판정: 주제어보다 **더 자주** 나오는 개념이 있으면 그 개념의 문제다.
    # 같은 횟수(>=)로 잡으면 "안정형 투자자가 PER 낮은 주식을 선호할 때" 처럼
    # 다른 개념을 도구로 쓴 정상 문제까지 걸린다. 오탐은 정상 출제를 negative 로
    # 잘못 라벨링해 "옳은 출제를 피하라"고 가르치므로, 놓치는 것보다 해롭다.
    intruders = [c for c, n in rivals.items() if n > topic_hits]
    assert not intruders, (
        f"주제 이탈: '{topic}'({topic_hits}회) 대신 "
        f"{[f'{c}({rivals[c]}회)' for c in intruders]} 문제 생성"
    )
