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
    # 과거 요청에서 만든 문제와도 대조할지 (Qdrant 왕복).
    #
    # False = 이번 요청 안에서 만든 문제끼리만 겹치지 않게 한다. 기본값.
    # True  = 같은 유저가 예전에 받은 문제와도 겹치지 않게 한다.
    #
    # 중복 검사는 세 단계인데(concurrency.is_duplicate) 이 설정은 3단계만 끈다.
    #   1) 세트 내 문자 유사도   - 항상 동작
    #   2) 세트 내 의미 유사도   - 항상 동작 (임베딩)
    #   3) 과거 문제와 대조      - 이 설정으로 제어 (Qdrant)
    #
    # 끄면 같은 유저가 예전에 본 문제가 다시 나올 수 있다. 대신 문항마다 붙던
    # Qdrant 왕복이 사라지고, 뱅크가 쌓일수록 재생성이 잦아지던 문제도 없어진다.
    # (뱅크 저장 자체는 계속한다 - angle_offset 이 주제별 이력 수를 읽어
    #  재요청 시 다른 출제 관점부터 시작하게 만들기 때문이다. 그건 문제를
    #  '거르는' 게 아니라 '다르게 만드는' 장치라 끌 이유가 없다.)
    QUIZ_DUP_CHECK_PAST: bool = False
    # 한 요청 안에서 문제를 몇 개까지 동시에 생성할지.
    #
    # 세 생성 경로가 원래 순차였다. 근거는 "로컬 Ollama 가 단일 인스턴스라
    # 동시 요청을 못 받는다" 였는데, 서빙이 vLLM(연속 배칭)으로 바뀌며 무효가 됐다.
    # 순차 생성 탓에 /quiz/generate/prompt 가 LLM 을 7번 직렬 호출해 2분을 넘겼고,
    # NestJS 타임아웃과 RunPod 프록시(Cloudflare 100초)에 함께 걸렸다.
    #
    # 상한을 두는 이유는 서버가 못 견뎌서가 아니다. vLLM 여유는 동시 49개인데
    # 그건 서버 전체 기준이고, 동시 사용자 20명이 각자 5개짜리 세트를 요청하면
    # 100건이 된다. 요청 하나가 큐를 독점하지 않도록 요청 단위로 묶는다.
    # 5 = 프롬프트/오답 생성의 기본 문제 수. 늘려도 한 세트 안에서는 이득이 없다.
    QUIZ_GEN_CONCURRENCY: int = 5
    # /quiz/generate/prompt 에서 프롬프트에 개수 언급이 없을 때 만들 문제 수.
    #
    # 5 -> 3 으로 내렸다 (2026-09-05 파드 검증). 이 경로는 LLM 을
    # 1(분석) + N(생성) + 1(검증) 번 호출하므로 N 이 그대로 응답 시간이 된다.
    # 파드 실측에서 5개가 84.9초였고, RunPod HTTP 프록시(Cloudflare)가 100초에서
    # 524 로 끊는 것이 같은 검증에서 확인됐다. 여유가 15초뿐이라 생성 길이가
    # 조금만 길어져도 넘어간다 (같은 작업이 41.7초 / 81.1초로 흔들리는 것을 관측).
    # 3 으로 내리면 약 60초가 되어 마진이 40초로 벌어진다.
    #
    # 설정으로 뺀 이유: 발표 당일에 느리면 이미지 재빌드(15분) 없이
    # RunPod 템플릿 env 만 고쳐 대응할 수 있어야 한다.
    PROMPT_QUIZ_DEFAULT_COUNT: int = 3

    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    # 임베딩 실행 장치. 로컬(4060 Ti 8GB)은 VRAM 을 LLM 이 다 쓰므로 cpu,
    # 서버(H100)는 vLLM 이 0.88 만 잡고 남는 자리에 올릴 수 있어 cuda.
    # bge-m3 는 CPU 에서 1회 50~200ms 인데, 중복 검사가 문항 수의 제곱으로
    # 호출되므로(bank.py) 서버에서는 이게 실질 병목이 된다.
    EMBEDDING_DEVICE: str = "cpu"
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