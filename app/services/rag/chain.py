from datetime import datetime, timedelta
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from app.services.rag.vectorstore import get_vectorstore
from app.core.config import settings
from app.schemas.chat import Message

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 친절하고 전문적인 주식 투자 코치입니다.
아래 참고 자료를 바탕으로 사용자의 질문에 답변해주세요.
모르는 내용은 모른다고 솔직하게 말하고, 투자 판단은 사용자 본인이 하도록 안내하세요.

[참고 자료]
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 친절하고 전문적인 주식 투자 코치입니다.
주식 투자와 관련된 질문에 성실하게 답변해주세요.
투자 판단은 사용자 본인이 하도록 안내하세요."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


def _convert_history(history: list[Message]) -> list:
    result = []
    for msg in history:
        if msg.role == "user":
            result.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            result.append(AIMessage(content=msg.content))
    return result


def _get_llm() -> ChatOllama:
    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=0.3,
        reasoning=False,
    )


def _get_retriever(tickers: list[str]):
    vs = get_vectorstore()
    now = datetime.now()
    news_cutoff = int((now - timedelta(days=7)).strftime("%Y%m%d"))
    dart_cutoff = int((now - timedelta(days=90)).strftime("%Y%m%d"))

    ticker_filter = (
        {"ticker": {"$eq": tickers[0]}}
        if len(tickers) == 1
        else {"ticker": {"$in": tickers}}
    )

    return vs.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {
                "$and": [
                    ticker_filter,
                    {
                        "$or": [
                            {"$and": [
                                {"source": {"$eq": "naver_news"}},
                                {"published_at": {"$gte": news_cutoff}},
                            ]},
                            {"$and": [
                                {"source": {"$eq": "dart"}},
                                {"published_at": {"$gte": dart_cutoff}},
                            ]},
                        ]
                    }
                ]
            },
        }
    )


def _format_docs(docs) -> str:
    if not docs:
        return "관련 자료를 찾을 수 없습니다."
    return "\n\n".join(doc.page_content for doc in docs)


async def run_rag_chain(question: str, tickers: list[str], history: list[Message]) -> str:
    retriever = _get_retriever(tickers)
    docs = await retriever.ainvoke(question)
    context = _format_docs(docs)
    chain = RAG_PROMPT | _get_llm() | StrOutputParser()
    return await chain.ainvoke({
        "context": context,
        "history": _convert_history(history),
        "question": question,
    })


async def run_general_chain(question: str, history: list[Message]) -> str:
    chain = GENERAL_PROMPT | _get_llm() | StrOutputParser()
    return await chain.ainvoke({
        "history": _convert_history(history),
        "question": question,
    })


async def stream_rag_chain(question: str, tickers: list[str], history: list[Message]):
    retriever = _get_retriever(tickers)
    docs = await retriever.ainvoke(question)
    context = _format_docs(docs)
    chain = RAG_PROMPT | _get_llm() | StrOutputParser()
    async for chunk in chain.astream({
        "context": context,
        "history": _convert_history(history),
        "question": question,
    }):
        yield chunk


async def stream_general_chain(question: str, history: list[Message]):
    chain = GENERAL_PROMPT | _get_llm() | StrOutputParser()
    async for chunk in chain.astream({
        "history": _convert_history(history),
        "question": question,
    }):
        yield chunk