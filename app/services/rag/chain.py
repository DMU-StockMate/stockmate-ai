from datetime import datetime, timedelta
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from app.services.rag.vectorstore import get_vectorstore
from app.core.config import settings
from app.schemas.chat import Message, QuizContext

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
- 주제: {topic}"""),
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


def _get_qdrant_filter(tickers: list[str]) -> Filter:
    """Qdrant 필터 생성 — ticker + 날짜"""
    now = datetime.now()
    news_cutoff = int((now - timedelta(days=7)).strftime("%Y%m%d"))
    dart_cutoff = int((now - timedelta(days=90)).strftime("%Y%m%d"))

    ticker_condition = FieldCondition(
        key="metadata.ticker",
        match=MatchValue(value=tickers[0]) if len(tickers) == 1
        else MatchValue(any=tickers),
    )

    news_filter = Filter(must=[
        FieldCondition(key="metadata.source", match=MatchValue(value="naver_news")),
        FieldCondition(key="metadata.published_at", range=Range(gte=news_cutoff)),
    ])

    dart_filter = Filter(must=[
        FieldCondition(key="metadata.source", match=MatchValue(value="dart")),
        FieldCondition(key="metadata.published_at", range=Range(gte=dart_cutoff)),
    ])

    return Filter(
        must=[ticker_condition],
        should=[news_filter, dart_filter],
    )


def _get_retriever(tickers: list[str]):
    vs = get_vectorstore()
    qdrant_filter = _get_qdrant_filter(tickers)
    return vs.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": qdrant_filter,
        }
    )


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
    chain = GENERAL_PROMPT | _get_llm() | StrOutputParser()
    return await chain.ainvoke({
        "history": _convert_history(history),
        "question": question,
        "level_guide": _get_level_guide(investment_level),
    })


async def run_quiz_chain(question: str, history: list[Message], quiz_context: QuizContext, investment_level: str = "미설정") -> str:
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
    chain = GENERAL_PROMPT | _get_llm() | StrOutputParser()
    async for chunk in chain.astream({
        "history": _convert_history(history),
        "question": question,
        "level_guide": _get_level_guide(investment_level),
    }):
        yield chunk


async def stream_quiz_chain(question: str, history: list[Message], quiz_context: QuizContext, investment_level: str = "미설정"):
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
    }):
        yield chunk