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


def build_llm(temperature: float, num_predict: int = 1024, num_ctx: int = 6144,
              reasoning_effort: str | None = None,
              max_tokens: int | None = None):
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
        # Qwen3.8 계열은 enable_thinking 이 아니라 reasoning_effort 를 받는다.
        # 기본값이 xhigh 라 지정하지 않으면 과하게 오래 생각한다.
        #
        # strip() 이 붙은 이유: 이 값은 런북 절차상 사람이 셸에서 손으로 넣는다
        # (§6 의 $env:LLM_REASONING_EFFORT=... , cmd 의 set ...). cmd 의
        #   set LLM_REASONING_EFFORT=low && ...
        # 은 값에 **후행 공백을 포함시킨다.** 그러면 "low " 가 서버로 나가
        #   400 reasoning_effort: Invalid option: expected one of
        #   "max"|"xhigh"|"high"|"medium"|"low"|"minimal"|"none"
        # 로 죽는데, 이 400 은 생성 재시도 3회를 전부 태우고 500 으로 나간다.
        # 보이지 않는 공백 한 칸 때문에 원인 파악이 어려운 실패라 여기서 막는다.
        # 호출부가 명시하면 그것을, 아니면 전역 설정을 쓴다.
        # 퀴즈와 채팅이 서로 다른 강도를 쓰기 위한 통로다 (config.py 참고).
        effort = (reasoning_effort
                  if reasoning_effort is not None
                  else settings.LLM_REASONING_EFFORT).strip()
        if effort:
            extra_body["reasoning_effort"] = effort

        return ChatOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            temperature=temperature,
            # 우선순위: 호출부가 명시한 값 > 전역 설정 > 함수 기본값.
            # 묶음 생성처럼 출력이 문제 수에 비례하는 호출이 자기 예산을 직접
            # 정할 수 있어야 한다. 전역값을 올리면 채팅 응답까지 길어진다.
            max_tokens=(max_tokens or settings.LLM_MAX_TOKENS or num_predict),
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