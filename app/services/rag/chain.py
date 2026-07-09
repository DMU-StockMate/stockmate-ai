import asyncio
from datetime import datetime, timedelta
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range
from app.services.rag.vectorstore import get_vectorstore
from app.services.rag.ticker_extractor import extract_tickers
from app.core.config import settings
from app.schemas.chat import Message, QuizContext
from app.core.logger import setup_logger
logger = setup_logger(__name__)

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

사용자 수준: {level_guide}

답변 시 다음을 지켜주세요:
- 뉴스와 공시 내용을 바탕으로 현재 상황을 객관적으로 분석해주세요.
- 긍정적/부정적 요인을 구분해서 설명해주세요.
- 매수/매도 같은 직접적인 투자 판단은 하지 마세요.
- 면책 문구는 마지막에 한 번만 간략하게 언급하세요.
- 금액 단위(조원/억원)는 참고 자료에 이미 변환되어 있으면 그 값을 그대로 인용하세요.
  직접 조/억 단위로 재계산하지 마세요 — 자릿수를 잘못 세면 실제 금액과 크게 어긋납니다.
- 서로 다른 항목(예: 매출액 증감과 자산 증감)의 수치를 섞어서 인용하지 마세요.

[참고 자료]
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 친절하고 전문적인 주식 투자 코치입니다.
주식 투자와 관련된 질문에 성실하게 답변해주세요.
투자 판단은 사용자 본인이 하도록 안내하세요.

사용자 수준: {level_guide}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

QUIZ_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 친절하고 전문적인 주식 투자 코치입니다.
아래 문제를 바탕으로 사용자의 질문에 답변해주세요.

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

{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


def _get_llm() -> ChatOllama:
    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=0.3,
        reasoning=False,
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


def _ticker_condition(tickers: list[str]) -> FieldCondition:
    return FieldCondition(
        key="metadata.ticker",
        match=MatchValue(value=tickers[0]) if len(tickers) == 1
        else MatchAny(any=tickers),
    )


# source별 (날짜 cutoff, 후보 풀 크기, 최종 채택 개수).
# 재무제표는 분기/연간 단위로만 갱신되므로 뉴스·공시보다 훨씬 긴 창을 둔다.
_SOURCE_SEARCH_CONFIG = {
    "naver_news": {"cutoff_days": 7, "pool": 8, "take": 2},
    "dart": {"cutoff_days": 90, "pool": 10, "take": 3},
    "dart_financials": {"cutoff_days": 730, "pool": 4, "take": 2},
}


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
    now = datetime.now()
    cutoff = int((now - timedelta(days=cfg["cutoff_days"])).strftime("%Y%m%d"))
    source_filter = Filter(must=[
        _ticker_condition(tickers),
        FieldCondition(key="metadata.source", match=MatchValue(value=source)),
        FieldCondition(key="metadata.published_at", range=Range(gte=cutoff)),
    ])

    try:
        results = await vs.asimilarity_search_with_score(
            question, k=cfg["pool"], filter=source_filter,
        )
    except Exception as e:
        logger.error(f"소스별 검색 실패 [{source}]: {e}")
        return []

    docs = [doc for doc, _score in results]
    docs.sort(key=lambda d: d.metadata.get("published_at", 0), reverse=True)
    return docs[: cfg["take"]]


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


def _format_docs(docs) -> str:
    if not docs:
        return "관련 자료를 찾을 수 없습니다."
    return "\n\n".join(doc.page_content for doc in docs)


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
    from app.services.external.kis import get_stocks_info
    stocks = await get_stocks_info(tickers)
    from app.services.rag.chain import _format_stock_data
    stock_context = _format_stock_data(stocks)
    if stock_context:
        context = f"{stock_context}\n\n{context}"
    chain = RAG_PROMPT | _get_llm() | StrOutputParser()
    return await chain.ainvoke({
        "context": context,
        "history": _convert_history(history),
        "question": question,
        "level_guide": _get_level_guide(investment_level),
    })


async def run_general_chain(question: str, history: list[Message], investment_level: str = "미설정") -> str:
    logger.info(f"일반 체인 실행: {question[:30]}...")
    chain = GENERAL_PROMPT | _get_llm() | StrOutputParser()
    return await chain.ainvoke({
        "history": _convert_history(history),
        "question": question,
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
        from app.services.external.kis import get_stocks_info
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
        lines.append(
            f"{ticker}: 현재가 {data['current_price']:,}원 "
            f"({data['change_rate']:+.2f}%) | "
            f"PER {data['per']} | PBR {data['pbr']} | EPS {data['eps']:,}"
        )
    return "\n".join(lines)


async def stream_rag_chain(question: str, tickers: list[str], history: list[Message], investment_level: str = "미설정"):
    logger.info(f"RAG 스트리밍 [{', '.join(tickers)}]: {question[:30]}...")
    retriever = _get_retriever(tickers)
    docs = await retriever.ainvoke(question)
    context = _format_docs(docs)
    from app.services.external.kis import get_stocks_info
    stocks = await get_stocks_info(tickers)
    stock_context = _format_stock_data(stocks)
    if stock_context:
        context = f"{stock_context}\n\n{context}"
    chain = RAG_PROMPT | _get_llm() | StrOutputParser()
    async for chunk in chain.astream({
        "context": context,
        "history": _convert_history(history),
        "question": question,
        "level_guide": _get_level_guide(investment_level),
    }):
        yield chunk


async def stream_general_chain(question: str, history: list[Message], investment_level: str = "미설정"):
    logger.info(f"일반 스트리밍: {question[:30]}...")
    chain = GENERAL_PROMPT | _get_llm() | StrOutputParser()
    async for chunk in chain.astream({
        "history": _convert_history(history),
        "question": question,
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
        "level_guide": _get_level_guide(investment_level),
        "question_type": quiz_context.question_type,
        "question_text": quiz_context.question_text,
        "choices": _format_choices(quiz_context.choices),
        "explanation": quiz_context.explanation or "없음",
        "topic": quiz_context.topic,
        "context": _format_quiz_context_block(extra_context),
    }):
        yield chunk