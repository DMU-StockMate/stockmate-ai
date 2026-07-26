import re
from rapidfuzz import process, fuzz
from app.services.external.dart import _corp_code_map

# 자주 쓰는 축약어 사전
ALIAS_MAP = {
    "삼전": "삼성전자",
    "하이닉스": "SK하이닉스",
    "카카오뱅크": "카카오뱅크",
    "카뱅": "카카오뱅크",
    "셀트리": "셀트리온",
    "현대차": "현대자동차",
    "기아차": "기아",
    "엘지": "LG전자",
    "네이버": "NAVER",
    "카카오": "카카오",
    "포스코": "POSCO홀딩스",
    "삼바": "삼성바이오로직스",
    "두산밥캣": "두산밥캣",
}

# 제거할 조사/어미 목록
JOSA_PATTERN = re.compile(
    r"(이랑|랑|와|과|은|는|이|가|을|를|의|도|만|에서|에|로|으로|까지|부터|이나|나)$"
)


def _remove_josa(token: str) -> str:
    return JOSA_PATTERN.sub("", token)


def extract_tickers(question: str, score_cutoff: int = 75) -> list[str]:
    if not _corp_code_map:
        return []

    corp_names = list(_corp_code_map.keys())
    corp_name_set = set(corp_names)
    found = []

    def _add(name: str) -> None:
        if name not in found:
            found.append(name)

    # 토큰 분리 + 조사 제거
    raw_tokens = question.replace(",", " ").replace(".", " ").split()
    tokens = [_remove_josa(t) for t in raw_tokens]

    candidates = set()
    for token in tokens:
        if len(token) >= 2:
            candidates.add(token)
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + 4, len(tokens) + 1)):
            combined = " ".join(tokens[i:j])
            if len(combined) >= 2:
                candidates.add(combined)

    # 각 후보는 "정확 일치 > 축약어 > 엄격 substring > 퍼지" 순으로 딱 한 단계에서만
    # 확정한다. 이전 구현은 substring 매칭 후에도 continue가 안쪽 루프만 돌아 퍼지가
    # 항상 실행됐고(엉뚱한 종목 혼입), substring 임계값(0.4)도 느슨해 "삼성"/"전자" 같은
    # 짧은 토큰이 다수 종목에 매칭돼 오추출이 잦았다.
    for candidate in candidates:
        # 1. 회사명과 정확히 일치 (가장 신뢰도 높음)
        if candidate in corp_name_set:
            _add(candidate)
            continue

        # 2. 축약어 사전
        if candidate in ALIAS_MAP:
            _add(ALIAS_MAP[candidate])
            continue

        # 3. 엄격 substring: 후보가 회사명의 60% 이상을 커버할 때만 인정하고,
        #    여러 개가 걸리면 후보와 가장 가까운(가장 짧은) 이름 하나만 채택해 과매칭 방지.
        #    ex) "하이닉스"(4) -> "SK하이닉스"(6) O / "전자"(2) -> "삼성전자"(4) X
        matched = [
            name for name in corp_names
            if candidate in name and len(candidate) >= len(name) * 0.6
        ]
        if matched:
            _add(min(matched, key=len))
            continue

        # 4. 위 단계에서 못 잡으면 퍼지 매칭 (오타 대비)
        result = process.extractOne(
            candidate,
            corp_names,
            scorer=fuzz.ratio,
            score_cutoff=score_cutoff,
        )
        if result:
            _add(result[0])

    return found


def extract_tickers_from_history(history: list) -> list[str]:
    found = []
    for msg in history:
        # user 메시지에서만 추출
        if msg.role != "user":
            continue
        tickers = extract_tickers(msg.content)
        for t in tickers:
            if t not in found:
                found.append(t)
    return found
