"""LLM 백엔드 팩토리.

Ollama와 OpenAI 호환 서버(llama.cpp llama-server 등)를 같은 인터페이스로 돌려준다.
settings.LLM_BACKEND 로 전환하며, 호출부(quality.py / chain.py)는 바뀌지 않는다.

llama-server 띄우는 예:
    llama-server.exe -m <gguf> -ngl 999 --n-cpu-moe 34 --flash-attn on -c 16384 --port 8080
그리고 .env 에:
    LLM_BACKEND=openai
    LLM_BASE_URL=http://127.0.0.1:8080/v1
    LLM_MODEL=qwen3.6-35b-a3b
"""
from app.core.config import settings


def build_llm(temperature: float, num_predict: int = 1024, num_ctx: int = 6144):
    """설정된 백엔드에 맞는 LangChain 채팅 모델을 만든다.

    num_ctx 는 Ollama 전용이다. llama-server 는 서버 기동 시 -c 로 정하므로
    클라이언트가 관여하지 않는다.
    """
    if settings.LLM_BACKEND == "openai":
        from langchain_openai import ChatOpenAI

        extra_body = {}
        if settings.LLM_DISABLE_THINKING:
            # Qwen3.5/3.6 채팅 템플릿이 enable_thinking 을 받는다.
            # RAG/퀴즈는 속도가 중요해 추론 모드를 끈다.
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        return ChatOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            temperature=temperature,
            max_tokens=num_predict,
            extra_body=extra_body or None,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=temperature,
        reasoning=False,
        num_predict=num_predict,
        num_ctx=num_ctx,
    )