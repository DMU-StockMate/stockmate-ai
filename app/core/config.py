from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen3.5:9b"
    # LLM 백엔드 전환 (ollama | openai)
    #   ollama : Ollama 서버 (기존). OLLAMA_BASE_URL + LLM_MODEL 사용
    #   openai : OpenAI 호환 서버(llama.cpp llama-server 등). LLM_BASE_URL + LLM_MODEL 사용
    # llama-server 예시:
    #   llama-server.exe -m <gguf> -ngl 999 --n-cpu-moe 34 --flash-attn on -c 16384 --port 8080
    #   LLM_BACKEND=openai / LLM_BASE_URL=http://127.0.0.1:8080/v1
    LLM_BACKEND: str = "ollama"
    LLM_BASE_URL: str = "http://127.0.0.1:8080/v1"
    LLM_API_KEY: str = "no-key"
    # openai 백엔드에서 thinking 비활성화 (Qwen3.5/3.6 채팅 템플릿 인자)
    LLM_DISABLE_THINKING: bool = True
    # 모델 비교 실험용 오버라이드. 기본값은 기존 동작을 그대로 둔다.
    #   LLM_REASONING_EFFORT : Qwen3.8 계열의 추론 강도 (low | medium | xhigh).
    #                          빈 문자열이면 보내지 않는다.
    #   LLM_MAX_TOKENS       : >0 이면 build_llm 의 num_predict 를 덮어쓴다.
    #                          추론을 켜면 사고 토큰이 응답 예산을 먹어 본문이
    #                          잘리므로, 비교 실행에서는 넉넉히 올려야 공정하다.
    LLM_REASONING_EFFORT: str = ""
    LLM_MAX_TOKENS: int = 0
    # 이 AI 서버 자체의 인증 키 (외부 노출 시 필수)
    # 비워두면 인증을 끈다 - 로컬 개발용. 포트포워딩할 때는 반드시 채울 것.
    AI_API_KEY: str = ""
    
    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "stockmate"

    # 퀴즈 뱅크 (요청 간 중복 방지용 - RAG 컬렉션과 분리)
    QUIZ_BANK_COLLECTION: str = "quiz_bank"
    QUIZ_BANK_ENABLED: bool = True
    # 코사인 유사도 임계값 (bge-m3 실측 기반).
    #   0.89 "PER 낮아짐의 의미?" vs "PER 낮아지면 주가/이익 관계?"  -> 중복
    #   0.84 "비교해야 할 기준?"   vs "업종 간 비교가 필요한 이유?"   -> 중복
    #   0.80 해석 관점 vs 정의 관점                                  -> 중복 아님
    # 0.90은 거의 같은 문장만 잡아 실질적으로 무용했다. 0.83이 중복만 걸러낸다.
    QUIZ_DUP_THRESHOLD: float = 0.83

    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    DART_API_KEY: str = ""
    NAVER_CLIENT_ID: str = ""
    NAVER_CLIENT_SECRET: str = ""
    KIS_APP_KEY: str = ""
    KIS_APP_SECRET: str = ""
    KIS_ACCOUNT: str = ""
    HF_TOKEN: str = ""

    class Config:
        env_file = ".env"

settings = Settings()