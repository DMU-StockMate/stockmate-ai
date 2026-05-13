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
    r"(이랑|이랑|랑|이랑|와|과|은|는|이|가|을|를|의|도|만|에서|에|로|으로|까지|부터|이나|나)$"
)


def _remove_josa(token: str) -> str:
    return JOSA_PATTERN.sub("", token)


def extract_tickers(question: str, score_cutoff: int = 75) -> list[str]:
    if not _corp_code_map:
        return []

    corp_names = list(_corp_code_map.keys())
    found = []

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

    for candidate in candidates:
        # 1. 축약어 사전 우선 적용
        if candidate in ALIAS_MAP:
            match_name = ALIAS_MAP[candidate]
            if match_name not in found:
                found.append(match_name)
            continue

        # 2. 후보가 회사명에 포함되는 경우만 허용 (방향 제한)
        #    ex) "하이닉스" → "SK하이닉스" O / "이닉스" → "SK하이닉스" X
        matched = [
            name for name in corp_names
            if candidate in name and len(candidate) >= len(name) * 0.4
        ]
        for match_name in matched:
            if match_name not in found:
                found.append(match_name)
            continue

        # 3. 직접 포함 안 되면 퍼지 매칭 (오타 대비)
        result = process.extractOne(
            candidate,
            corp_names,
            scorer=fuzz.ratio,
            score_cutoff=score_cutoff,
        )
        if result:
            match_name = result[0]
            if match_name not in found:
                found.append(match_name)

    return found