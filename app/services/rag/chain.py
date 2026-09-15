import asyncio
import re
from datetime import datetime, timedelta
from app.core.llm import build_llm
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range
from app.services.rag.vectorstore import get_vectorstore
from app.services.rag.ticker_extractor import extract_tickers
from app.services.external.kis import get_stocks_info
from app.services.external.dart import is_key_report, is_low_info_report
from app.core.config import settings
from app.schemas.chat import Message, QuizContext
from app.core.logger import setup_logger
logger = setup_logger(__name__)

# Qdrant 문서 source → 사람이 읽는 라벨. 컨텍스트에 붙여 LLM이 뉴스/공시/재무를 구분하고
# 최신성을 판단할 수 있게 한다.
_SOURCE_LABEL = {
    "naver_news": "뉴스",
    "dart": "공시",
    "dart_financials": "재무",
}

LEVEL_GUIDE = {
    "입문": "아주 쉽게, 비유를 들어 초등학생도 이해할 수 있게 설명해주세요.",
    "초급": "쉬운 용어로 기본 개념 위주로 설명해주세요.",
    "중급": "전문 용어를 사용하되 핵심을 간결하게 설명해주세요.",
    "고급": "심층적인 분석과 전문적인 시각으로 설명해주세요.",
    "미설정": "친절하게 설명해주세요.",
}

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 친절하고 전문적인 주식 투자 코치입니다.
아래 참고 자료를 바탕으로 사용자의 질문에 답변해주세요.

오늘 날짜: {today}
사용자 수준: {level_guide}

답변 시 다음을 지켜주세요:
- 답변은 반드시 한국어로만 작성하세요. 한자나 일본어 문자를 절대 쓰지 마세요.
  (숫자, 그리고 PER·EPS 같은 지표 약어는 그대로 써도 됩니다.)
- 참고 자료에 있는 내용만 근거로 삼고, 자료에 없는 내용은 추측하거나 지어내지 마세요.
  자료로 확인되지 않으면 "제공된 자료에서는 확인되지 않습니다"라고 솔직하게 말하세요.
- 각 자료 앞의 [출처 날짜] 표시(예: [뉴스 2026-07-15], [공시 2026-07-10], [재무])를 활용해
  더 최근 정보를 우선하고, 내용을 인용할 때 날짜·출처를 함께 밝히세요.
- '오래된 자료' 표시가 붙은 항목은 최근 자료가 없어 기간을 넓혀 찾은 것입니다.
  최신 소식으로 단정하지 말고 언제 자료인지 분명히 밝혀 설명하세요.
- 참고 자료에 담긴 구체적 사실(처분 주식 수·금액·기간, 지분 변동, 실적 수치, 날짜 등)을
  빠뜨리지 말고 인용해 충실히 설명하세요. "여러 공시를 통해 정보를 제공합니다" 같은 막연한
  요약이나 일반론으로 끝내지 마세요.
- 답변은 충분히 길고 자세해야 합니다. 두세 줄로 끝내면 불충분합니다.
  공시가 여러 건이면 되도록 빠짐없이 다루고, 주요 공시마다 항목으로 나눠 각각 최소 2~3문장으로
  "무슨 공시인지 · 핵심 수치(주식 수·금액·지분율 등) · 배경이나 의미"를 구체적으로 설명하세요.
- 긍정적/부정적 요인을 구분해서 객관적으로 설명해주세요.
- 매수/매도 같은 직접적인 투자 판단은 하지 마세요.
- 면책 문구는 쓰지 마세요. 서버가 답변 끝에 붙입니다.
- 금액 단위(조원/억원)는 참고 자료에 이미 변환되어 있으면 그 값을 그대로 인용하세요.
  직접 조/억 단위로 재계산하지 마세요 — 자릿수를 잘못 세면 실제 금액과 크게 어긋납니다.
- 서로 다른 항목(예: 매출액 증감과 자산 증감)의 수치를 섞어서 인용하지 마세요.
- 계산식은 수식 기호 없이 한국어 문장으로 쓰세요.
  달러 기호($)나 역슬래시 명령(\\frac, \\text 등)은 화면에 그대로 노출됩니다.
  (좋은 예: "PER은 주가를 주당순이익으로 나눈 값입니다")

[참고 자료]
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 친절하고 전문적인 주식 투자 코치입니다.
주식 투자와 관련된 질문에 성실하게 답변해주세요.

오늘 날짜: {today}
사용자 수준: {level_guide}

답변 시 다음을 지켜주세요:
- 답변은 반드시 한국어로만 작성하세요. 한자나 일본어 문자를 절대 쓰지 마세요.
  (숫자, 그리고 PER·EPS 같은 지표 약어는 그대로 써도 됩니다.)
- 계산식은 수식 기호 없이 한국어 문장으로 쓰세요.
  달러 기호($)나 역슬래시 명령(\\frac, \\text 등)은 화면에 그대로 노출됩니다.
  (좋은 예: "PER은 주가를 주당순이익으로 나눈 값입니다")
- 투자 판단은 사용자 본인이 하도록 안내하세요.
- 실시간 시세, 특정 종목의 최근 뉴스·실적처럼 최신 정보가 필요한 질문에는 단정하지 말고,
  정확한 최신 수치는 확인이 필요하다고 안내하세요. 기억에 의존해 특정 수치나 날짜를 지어내지 마세요."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

QUIZ_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 친절하고 전문적인 주식 투자 코치입니다.
아래 문제를 바탕으로 사용자의 질문에 답변해주세요.

오늘 날짜: {today}
사용자 수준: {level_guide}

[문제 정보]
- 문제 유형: {question_type}
- 문제: {question_text}
- 선택지: {choices}
- 해설: {explanation}
- 주제: {topic}

아래에 [추가 참고 자료]가 있다면, 실시간 수치나 최근 뉴스/공시를 묻는 질문에는
반드시 그 자료를 근거로 답변하고 자료에 없는 내용은 추측하지 마세요.
[추가 참고 자료]가 없다면 문제 정보만으로 답변하세요.
답변은 반드시 한국어로만 작성하세요.

{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


# 모델이 쓰던 면책 문구를 서버가 붙이는 고정 문장으로 대체한다.
# 측정(2026-09-15, RAG 답변 20건): 한자 혼입 7건 중 5건이 이 면책 문구 자리였다.
#   "투자 판단의 근거로直接使用하지 마시기 바랍니다"
# 프롬프트에 "한국어로만 작성하세요"가 있는데도 샜다. 매번 새로 쓰게 할 이유가 없는
# 정형 문장이므로 생성 대상에서 빼고 고정 문장을 붙인다 - 문구도 일관돼진다.
RAG_DISCLAIMER = (
    "\n\n---\n"
    "이 답변은 제공된 뉴스·공시·재무 자료를 정리한 것이며 투자 권유가 아닙니다. "
    "투자 판단과 그 결과는 투자자 본인에게 있습니다."
)

# 한자·가나. 한국 주식 설명에서 정당하게 쓰일 일이 없다.
_CJK_LEAK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")


def _warn_if_not_korean(answer: str, where: str) -> None:
    """한자·가나가 섞였으면 경고만 남긴다.

    퀴즈는 assert_korean 으로 거절하고 재생성하지만, 채팅은 스트리밍이라 되돌릴 수 없고
    답변 전체를 버리는 비용이 훨씬 크다. 대신 빈도를 로그로 드러내 추적 가능하게 한다.
    """
    m = _CJK_LEAK.search(answer or "")
    if m:
        i = m.start()
        logger.warning(
            f"{where} 답변에 한자/가나 혼입: {m.group()!r} "
            f"(…{answer[max(0, i - 20):i + 20]}…)"
        )


def _get_llm():
    """RAG 답변용 LLM (temperature 0.3 - 사실 기반이라 낮게).

    num_predict 는 1024 였는데, 공시 본문까지 컨텍스트에 들어가면서 답변이 길어져
    문장 중간에 잘리는 일이 생겼다 (실측: 삼성전자/SK하이닉스 질문 모두 잘림).
    llama-server 는 -c 16384 로 떠 있어 3072 여유는 충분하다.
    """
    return build_llm(
        temperature=0.3, num_predict=3072, num_ctx=6144,
        # 채팅은 퀴즈와 다른 사고 강도를 쓴다 (config.CHAT_REASONING_EFFORT).
        # 빈 문자열이면 전역값으로 되돌아간다.
        reasoning_effort=settings.CHAT_REASONING_EFFORT or None,
    )


def _convert_history(history: list[Message]) -> list:
    result = []
    for msg in history:
        if msg.role == "user":
            result.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            result.append(AIMessage(content=msg.content))
    return result


def _get_level_guide(investment_level: str) -> str:
    return LEVEL_GUIDE.get(investment_level, LEVEL_GUIDE["미설정"])


def _today_str() -> str:
    """프롬프트에 넣을 오늘 날짜. 모델이 '오늘/최근'을 판단하고 날짜를 정확히 말하도록 주입."""
    return datetime.now().strftime("%Y년 %m월 %d일")


def _ticker_condition(tickers: list[str]) -> FieldCondition:
    return FieldCondition(
        key="metadata.ticker",
        match=MatchValue(value=tickers[0]) if len(tickers) == 1
        else MatchAny(any=tickers),
    )


# source별 (날짜 cutoff, 후보 풀 크기, 최종 채택 개수, 최소 유사도 점수).
# 재무제표는 분기/연간 단위로만 갱신되므로 뉴스·공시보다 훨씬 긴 창을 둔다.
# min_score: 코사인 유사도(정규화 임베딩, 대략 0~1, 높을수록 유사)가 이 값 미만이면
#   최신이라도 컨텍스트에서 제외한다. 유사도 낮은 문서가 섞여 할루시네이션을 유발하는 걸 막는다.
#   (환경/데이터에 따라 튜닝 필요 — 우선 보수적으로 낮게 잡음)
_SOURCE_SEARCH_CONFIG = {
    # fallback_cutoff_days: 기본 창에서 후보가 0건일 때만 넓혀서 다시 찾는다.
    # 적재는 "최신 10건"을 무조건 가져오는데 검색 창은 7일이라, 뉴스가 뜸한 종목은
    # 적재한 10건이 전부 창 밖이라 후보가 0건이 된다
    # (실측: 한국주철관공업 10건 적재, 날짜 2026-03-31~08-07, 7일 내 0건).
    # take 2 -> 4: 대형주는 7일 안에도 후보가 충분한데(SK하이닉스 13건) 2건만 들어가
    # 투자 관점에서 더 중요한 기사가 잘렸다.
    "naver_news": {"cutoff_days": 7, "fallback_cutoff_days": 90,
                   "pool": 8, "take": 4, "min_score": 0.3},
    # 공시는 "공시 내용 알려줘" 같은 질문에 여러 건을 자세히 다뤄야 해서 take를 넉넉히 둔다.
    "dart": {"cutoff_days": 90, "pool": 12, "take": 5, "min_score": 0.3},
    "dart_financials": {"cutoff_days": 730, "pool": 4, "take": 2, "min_score": 0.2},
}


# 'score 통과분을 최신순 정렬' 하면, 대형주에서 거의 매일 나오는 임원 소유상황보고서가
# 자기주식처분결정·잠정실적 같은 핵심 공시를 항상 밀어낸다.
# (실측: 삼성전자 "최근 공시 중 중요한 거" 질문에 채택된 5건이 전부 임원 소유상황보고서였고,
#  점수가 더 높은 자기주식처분결정(0.6013)·자기주식처분결과보고서(0.5949)는 탈락했다)
# 완전히 버리지는 않는다 — 핵심 공시로 자리를 채운 뒤 남는 슬롯에만 넣는다.
# 유형 판정 기준은 적재(ingestion)와 공유해야 하므로 dart.py 에 둔다.


def _dart_title(content: str) -> str:
    """공시 청크의 첫 줄 "{회사명} 공시: {보고서명}" 에서 보고서명만 뽑는다."""
    first_line = (content or "").split("\n", 1)[0]
    return first_line.split("공시:", 1)[-1].strip()


def _is_low_info_dart(content: str) -> bool:
    return is_low_info_report(_dart_title(content))


def _dart_tier(content: str) -> int:
    """공시 채택 우선순위. 0 = 중요, 1 = 일반, 2 = 저정보."""
    title = _dart_title(content)
    if is_low_info_report(title):
        return 2
    return 0 if is_key_report(title) else 1


def _order_for_take(items: list, source: str, get_content, get_published) -> list:
    """관련성 통과분을 '채택 순서'로 정렬한다.

    dart 는 중요 → 일반 → 저정보 3계층으로 나눈 뒤 각 계층 안에서 최신순으로 본다.
    2계층(핵심/저정보)만 두었을 때는, 저정보로 분류되지 않는 형식적 공시(조회공시요구 등)가
    최신이라는 이유만으로 잠정실적·중대재해를 밀어냈다. 중요도를 표현하는 신호가 없으면
    최신성이 그 자리를 대신 차지한다.
    나머지 source(뉴스/재무)는 유형별 중요도 편차가 없으므로 기존대로 최신순만 본다.
    """
    def recency(x):
        return get_published(x) or 0

    if source != "dart":
        return sorted(items, key=recency, reverse=True)

    tiers: dict[int, list] = {0: [], 1: [], 2: []}
    for x in items:
        tiers[_dart_tier(get_content(x))].append(x)
    ordered = []
    for tier in (0, 1, 2):
        ordered.extend(sorted(tiers[tier], key=recency, reverse=True))
    return ordered


def _window_filter(tickers: list[str], source: str, cutoff_days: int) -> Filter:
    cutoff = int((datetime.now() - timedelta(days=cutoff_days)).strftime("%Y%m%d"))
    return Filter(must=[
        _ticker_condition(tickers),
        FieldCondition(key="metadata.source", match=MatchValue(value=source)),
        FieldCondition(key="metadata.published_at", range=Range(gte=cutoff)),
    ])


async def _search_window(vs, question: str, tickers: list[str], source: str,
                         cfg: dict, cutoff_days: int) -> list:
    """한 기간 창 안에서만 유사도 검색해 (문서, 점수) 목록을 돌려준다."""
    try:
        return await vs.asimilarity_search_with_score(
            question, k=cfg["pool"], filter=_window_filter(tickers, source, cutoff_days),
        )
    except Exception as e:
        logger.error(f"소스별 검색 실패 [{source}]: {e}")
        return []


async def _search_with_fallback(vs, question: str, tickers: list[str], source: str,
                                cfg: dict) -> tuple[list, bool]:
    """기본 창에서 아무것도 못 찾으면 넓은 창으로 한 번 더 찾는다.

    대형주의 신선도는 그대로 두고 뉴스가 뜸한 종목만 구제하기 위해, 넓히는 건 후보가
    0건일 때뿐이다. 넓혀서 찾았는지 여부를 함께 돌려주어 프롬프트에서 '오래된 자료'로
    표시할 수 있게 한다 - 표시 없이 넣으면 모델이 두 달 전 기사를 오늘 소식처럼 말한다.
    """
    results = await _search_window(vs, question, tickers, source, cfg, cfg["cutoff_days"])
    fallback = cfg.get("fallback_cutoff_days")
    if results or not fallback:
        return results, False

    results = await _search_window(vs, question, tickers, source, cfg, fallback)
    if results:
        logger.info(
            f"[{source}] 최근 {cfg['cutoff_days']}일 내 자료가 없어 "
            f"{fallback}일로 넓혀 {len(results)}건 찾음"
        )
    return results, bool(results)


async def _search_source(vs, question: str, tickers: list[str], source: str, cfg: dict) -> list:
    """특정 source(news/dart/financials) 안에서만 유사도 검색 후 최신순으로 재정렬한다.

    기존엔 news/dart/financials를 한 번의 벡터 검색에 섞어서(Filter의 should절) top-k를 뽑았는데,
    두 가지 문제가 있었다:
    1) 문서 타입 혼동 — "공시" 질문인데 임베딩 유사도만으로는 뉴스 기사가 공시 문서보다
       더 유사하다고 판단되어 뽑히는 경우가 있었다 (RAGAS 평가로 실측).
    2) 최신성 무시 — 순수 유사도 정렬이라 필터 범위(예: 90일) 안에 있어도 정작 최근 문서가
       아니라 예전 문서가 뽑히는 경우가 많았다 (예: "최근 공시" 질문에 두 달 전 공시가 뽑힘).
    그래서 source별로 유사도 상위 `pool`개를 먼저 뽑고, 그 안에서 published_at 내림차순으로
    재정렬해 상위 `take`개만 채택하는 2단계 방식으로 바꿨다 — 관련성(유사도)과 최신성을 분리해서 보장.
    """
    results, widened = await _search_with_fallback(vs, question, tickers, source, cfg)

    # 유사도 임계값 미만 문서는 버린다 (관련성 없는 문서가 최신순 정렬로 채택되는 것 방지).
    min_score = cfg.get("min_score", 0.0)
    docs = [doc for doc, score in results if score >= min_score]
    if widened:
        for doc in docs:
            doc.metadata = {**doc.metadata, "widened": True}
    # 관련성 통과분을 채택 순서(dart는 중요도 계층 → 최신순)로 정렬 후 상위 take개 채택.
    ordered = _order_for_take(
        docs, source,
        get_content=lambda d: d.page_content,
        get_published=lambda d: d.metadata.get("published_at"),
    )
    return ordered[: cfg["take"]]


class _MultiSourceRetriever:
    """news/dart/financials를 소스별로 따로 검색해서 합치는 리트리버.

    langchain 리트리버와 동일하게 `.ainvoke(question)` -> list[Document] 인터페이스를 유지해서
    호출부(run_rag_chain, stream_rag_chain, evaluate_rag.py)는 그대로 둘 수 있다.
    """

    def __init__(self, tickers: list[str]):
        self.tickers = tickers

    async def ainvoke(self, question: str) -> list:
        vs = get_vectorstore()
        results = await asyncio.gather(*[
            _search_source(vs, question, self.tickers, source, cfg)
            for source, cfg in _SOURCE_SEARCH_CONFIG.items()
        ])
        docs = []
        for source_docs in results:
            docs.extend(source_docs)
        return docs


def _get_retriever(tickers: list[str]) -> _MultiSourceRetriever:
    return _MultiSourceRetriever(tickers)


async def _search_source_debug(vs, question: str, tickers: list[str], source: str, cfg: dict) -> list[dict]:
    """_search_source와 동일한 검색을 하되, 디버깅용으로 각 후보의 score/날짜/채택여부를
    그대로 노출한다. (실제 답변 경로는 score를 버려서 '무엇이 왜 뽑혔는지'가 안 보였음)
    """
    results, widened = await _search_with_fallback(vs, question, tickers, source, cfg)

    min_score = cfg.get("min_score", 0.0)
    records = []
    for doc, score in results:
        records.append({
            "source": source,
            "score": round(float(score), 4),
            "published_at": doc.metadata.get("published_at"),
            "passed_score": float(score) >= min_score,
            "taken": False,
            # 기본 창에 자료가 없어 넓힌 결과인지. 프롬프트에 '오래된 자료'로 표시된다.
            "widened": widened,
            # 저정보 공시로 분류돼 후순위로 밀렸는지. 왜 안 뽑혔는지 디버깅용.
            "low_info": source == "dart" and _is_low_info_dart(doc.page_content),
            # 채택 계층: key(중요) / normal(일반) / low_info(저정보). dart 만 해당.
            "tier": ("key", "normal", "low_info")[_dart_tier(doc.page_content)]
            if source == "dart" else "",
            "content": doc.page_content,
        })

    # 실제 채택 로직과 동일: score 통과분을 채택 순서로 정렬 후 상위 take개만 taken=True.
    passed = _order_for_take(
        [r for r in records if r["passed_score"]], source,
        get_content=lambda r: r["content"],
        get_published=lambda r: r["published_at"],
    )
    for r in passed[: cfg["take"]]:
        r["taken"] = True
    return records


async def debug_retrieve(question: str, tickers: list[str]) -> list[dict]:
    """모든 source의 검색 후보를 score/날짜/채택여부와 함께 반환한다."""
    vs = get_vectorstore()
    results = await asyncio.gather(*[
        _search_source_debug(vs, question, tickers, source, cfg)
        for source, cfg in _SOURCE_SEARCH_CONFIG.items()
    ])
    records: list[dict] = []
    for source_records in results:
        records.extend(source_records)
    return records


def _fmt_published_at(value) -> str:
    """metadata.published_at(YYYYMMDD int) → "2026-07-14" 문자열. 파싱 실패 시 빈 문자열."""
    try:
        s = str(int(value))
        if len(s) == 8:
            return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    except (ValueError, TypeError):
        pass
    return ""


# 프롬프트에 넣을 문서 1건당 최대 글자 수(source별). 공시 본문 적재 후 문서가 길어져
# num_ctx(4096)를 넘겨 프롬프트가 잘리는 걸 막으면서, 사용자가 실제로 묻는 공시 본문에는
# 더 넉넉히 배정한다. 뉴스는 스니펫이라 짧게 잘라 컨텍스트 예산을 아낀다.
# (검색/임베딩은 전체 본문 사용, 프롬프트에 넣을 때만 발췌)
_DOC_CHAR_CAP = {
    "dart": 1000,
    "dart_financials": 700,
    "naver_news": 400,
}
_DEFAULT_DOC_CAP = 600


def _format_docs(docs) -> str:
    if not docs:
        return "관련 자료를 찾을 수 없습니다."
    blocks = []
    for doc in docs:
        source = doc.metadata.get("source")
        label = _SOURCE_LABEL.get(source, "자료")
        date_str = _fmt_published_at(doc.metadata.get("published_at"))
        marks = [m for m in (date_str, "오래된 자료" if doc.metadata.get("widened") else "") if m]
        header = f"[{label} {' · '.join(marks)}]" if marks else f"[{label}]"
        content = doc.page_content
        cap = _DOC_CHAR_CAP.get(source, _DEFAULT_DOC_CAP)
        if len(content) > cap:
            content = content[:cap] + " …(이하 생략)"
        blocks.append(f"{header}\n{content}")
    return "\n\n".join(blocks)


def _format_choices(choices) -> str:
    if not choices:
        return "없음"
    return "\n".join(
        f"{c.choice_no}. {c.text} {'(정답)' if c.is_correct else ''}"
        for c in choices
    )


async def run_rag_chain(question: str, tickers: list[str], history: list[Message], investment_level: str = "미설정") -> str:
    logger.info(f"RAG 체인 실행 [{', '.join(tickers)}]: {question[:30]}...")
    retriever = _get_retriever(tickers)
    docs = await retriever.ainvoke(question)
    context = _format_docs(docs)
    stocks = await get_stocks_info(tickers)
    stock_context = _format_stock_data(stocks)
    if stock_context:
        context = f"{stock_context}\n\n{context}"
    chain = RAG_PROMPT | _get_llm() | StrOutputParser()
    answer = await chain.ainvoke({
        "context": context,
        "history": _convert_history(history),
        "question": question,
        "today": _today_str(),
        "level_guide": _get_level_guide(investment_level),
    })
    _warn_if_not_korean(answer, "RAG")
    return answer.rstrip() + RAG_DISCLAIMER


async def run_rag_evaluate(question: str, tickers: list[str], history: list[Message], investment_level: str = "미설정") -> dict:
    """RAG 검색 가시성용. 실제 답변 경로와 동일하게 검색·답변하되,
    LLM에 넣은 컨텍스트/실시간 시세/검색 후보(score·날짜·채택여부)를 함께 반환한다.
    /chat/evaluate 엔드포인트와 테스트 스크립트가 이걸 근거로 최신성·정확성을 검증한다.
    """
    logger.info(f"RAG 평가 [{', '.join(tickers)}]: {question[:30]}...")
    retriever = _get_retriever(tickers)
    docs = await retriever.ainvoke(question)
    context = _format_docs(docs)
    stocks = await get_stocks_info(tickers)
    stock_context = _format_stock_data(stocks)
    full_context = f"{stock_context}\n\n{context}" if stock_context else context

    chain = RAG_PROMPT | _get_llm() | StrOutputParser()
    answer = await chain.ainvoke({
        "context": full_context,
        "history": _convert_history(history),
        "question": question,
        "today": _today_str(),
        "level_guide": _get_level_guide(investment_level),
    })
    _warn_if_not_korean(answer, "RAG")
    answer = answer.rstrip() + RAG_DISCLAIMER

    # score/날짜/채택여부가 담긴 상세 검색 기록 (답변과 동일 로직이라 taken 집합이 일치)
    retrieved = await debug_retrieve(question, tickers)
    return {
        "answer": answer,
        "stock_data": stocks,
        "retrieved": retrieved,
        "context_sent_to_llm": full_context,
    }


async def run_general_chain(question: str, history: list[Message], investment_level: str = "미설정") -> str:
    logger.info(f"일반 체인 실행: {question[:30]}...")
    chain = GENERAL_PROMPT | _get_llm() | StrOutputParser()
    return await chain.ainvoke({
        "history": _convert_history(history),
        "question": question,
        "today": _today_str(),
        "level_guide": _get_level_guide(investment_level),
    })


async def _build_quiz_context(question: str, quiz_context: QuizContext) -> tuple[str, dict]:
    """quiz_context.category에 따라 추가 참고 자료를 조회해 프롬프트에 넣을 텍스트를 만든다.

    category 값:
    - "metric": KIS 실시간 시세/지표
    - "news": 최근 뉴스 (Qdrant)
    - "disclosure": 최근 공시 (Qdrant)
    - "concept" 또는 None/그 외: 추가 자료 없음 (기존 동작 그대로, LLM이 문제 정보만으로 답변)

    NestJS가 이미 category를 보내주고 있었는데(quiz_context 스키마에 없어서 버려지고 있었음)
    지금까지는 이 값을 안 쓰고 항상 "RAG 없이 문제 정보만으로" 답했다 - 실시간 수치나
    최근 뉴스/공시를 물어보는 후속 질문엔 근거 없이 답할 위험이 있었다.

    반환값은 (프롬프트에 넣을 텍스트, ingested 카운트 dict) 튜플이다.
    ingested는 /chat/stream의 done 이벤트에 그대로 실어 보낼 수 있도록
    일반 채팅 플로우(ingest_news/ingest_disclosures 호출부)와 동일한 형태
    ({ticker: {"news": N}} / {ticker: {"dart": N}})로 맞춘다.
    metric은 Qdrant에 아무것도 적재하지 않고 KIS를 그때그때 조회만 하므로 빈 dict를 반환한다.
    """
    category = quiz_context.category
    ingested: dict = {}
    if not category or category == "concept":
        return "", ingested

    tickers = extract_tickers(question)
    if not tickers:
        tickers = extract_tickers(quiz_context.question_text)
    if not tickers and quiz_context.topic:
        tickers = extract_tickers(quiz_context.topic)
    if not tickers:
        logger.info(f"퀴즈 category={category}이지만 티커를 못 찾음 - 추가 자료 없이 진행")
        return "", ingested

    if category == "metric":
        stocks = await get_stocks_info(tickers)
        return _format_stock_data(stocks), ingested

    if category == "news":
        from app.services.rag.ingestion import ingest_news
        for ticker in tickers:
            count = await ingest_news(ticker)
            ingested.setdefault(ticker, {})["news"] = count
        vs = get_vectorstore()
        docs = await _search_source(vs, question, tickers, "naver_news", _SOURCE_SEARCH_CONFIG["naver_news"])
        return (_format_docs(docs) if docs else ""), ingested

    if category == "disclosure":
        from app.services.rag.ingestion import ingest_disclosures
        for ticker in tickers:
            count = await ingest_disclosures(ticker)
            ingested.setdefault(ticker, {})["dart"] = count
        vs = get_vectorstore()
        docs = await _search_source(vs, question, tickers, "dart", _SOURCE_SEARCH_CONFIG["dart"])
        return (_format_docs(docs) if docs else ""), ingested

    logger.warning(f"알 수 없는 퀴즈 category: {category} - 추가 자료 없이 진행")
    return "", ingested


def _format_quiz_context_block(context: str) -> str:
    return f"[추가 참고 자료]\n{context}" if context else ""


async def run_quiz_chain(question: str, history: list[Message], quiz_context: QuizContext, investment_level: str = "미설정") -> str:
    extra_context, _ingested = await _build_quiz_context(question, quiz_context)
    chain = QUIZ_PROMPT | _get_llm() | StrOutputParser()
    return await chain.ainvoke({
        "history": _convert_history(history),
        "question": question,
        "today": _today_str(),
        "level_guide": _get_level_guide(investment_level),
        "question_type": quiz_context.question_type,
        "question_text": quiz_context.question_text,
        "choices": _format_choices(quiz_context.choices),
        "explanation": quiz_context.explanation or "없음",
        "topic": quiz_context.topic,
        "context": _format_quiz_context_block(extra_context),
    })


def _format_stock_data(stocks: dict) -> str:
    if not stocks:
        return ""
    lines = ["[실시간 주가 데이터]"]
    for ticker, data in stocks.items():
        per = data["per"] if data["per"] is not None else "정보없음"
        pbr = data["pbr"] if data["pbr"] is not None else "정보없음"
        eps = f"{data['eps']:,}" if data["eps"] is not None else "정보없음"
        lines.append(
            f"{ticker}: 현재가 {data['current_price']:,}원 "
            f"({data['change_rate']:+.2f}%) | "
            f"PER {per} | PBR {pbr} | EPS {eps}"
        )
    return "\n".join(lines)


async def stream_rag_chain(question: str, tickers: list[str], history: list[Message], investment_level: str = "미설정"):
    logger.info(f"RAG 스트리밍 [{', '.join(tickers)}]: {question[:30]}...")
    retriever = _get_retriever(tickers)
    docs = await retriever.ainvoke(question)
    context = _format_docs(docs)
    stocks = await get_stocks_info(tickers)
    stock_context = _format_stock_data(stocks)
    if stock_context:
        context = f"{stock_context}\n\n{context}"
    chain = RAG_PROMPT | _get_llm() | StrOutputParser()
    parts: list[str] = []
    async for chunk in chain.astream({
        "context": context,
        "history": _convert_history(history),
        "question": question,
        "today": _today_str(),
        "level_guide": _get_level_guide(investment_level),
    }):
        parts.append(chunk)
        yield chunk
    _warn_if_not_korean("".join(parts), "RAG 스트리밍")
    yield RAG_DISCLAIMER


async def stream_general_chain(question: str, history: list[Message], investment_level: str = "미설정"):
    logger.info(f"일반 스트리밍: {question[:30]}...")
    chain = GENERAL_PROMPT | _get_llm() | StrOutputParser()
    async for chunk in chain.astream({
        "history": _convert_history(history),
        "question": question,
        "today": _today_str(),
        "level_guide": _get_level_guide(investment_level),
    }):
        yield chunk


async def stream_quiz_chain(
    question: str,
    history: list[Message],
    quiz_context: QuizContext,
    investment_level: str = "미설정",
    ingested_out: dict | None = None,
):
    """퀴즈 Q&A 스트리밍.

    ingested_out: 호출부(chat.py)가 빈 dict를 넘겨주면, category별로 실제 Qdrant에
    적재된 뉴스/공시 건수를 그 dict에 채워 넣는다(같은 dict 객체를 in-place로 갱신).
    제너레이터는 yield만 가능하고 return 값을 async for로 받을 수 없어서,
    호출부가 미리 만들어둔 dict를 참조로 넘기는 방식을 쓴다.
    """
    logger.info(f"퀴즈 스트리밍: {question[:30]}...")
    extra_context, ingested = await _build_quiz_context(question, quiz_context)
    if ingested_out is not None:
        ingested_out.update(ingested)
    chain = QUIZ_PROMPT | _get_llm() | StrOutputParser()
    async for chunk in chain.astream({
        "history": _convert_history(history),
        "question": question,
        "today": _today_str(),
        "level_guide": _get_level_guide(investment_level),
        "question_type": quiz_context.question_type,
        "question_text": quiz_context.question_text,
        "choices": _format_choices(quiz_context.choices),
        "explanation": quiz_context.explanation or "없음",
        "topic": quiz_context.topic,
        "context": _format_quiz_context_block(extra_context),
    }):
        yield chunk