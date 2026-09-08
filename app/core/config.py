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
    # 채팅(/chat/stream, /chat/evaluate) 전용 사고 강도.
    #
    # 퀴즈와 분리한 이유: 두 작업의 성격이 다르고, 실측 결과도 반대로 나왔다.
    #   퀴즈  - 정해진 JSON 틀 안에서 용어를 정확히 인출해야 한다.
    #           사고를 끄면 판관비/판매비와관리비 중복 나열 같은 조어 오류가 돌아온다.
    #           -> medium 유지 (claude/stockmate-quiz-latency.md)
    #   채팅  - 주어진 자료를 설명·종합한다. 사고를 꺼도 품질 저하가 관찰되지 않았다.
    #           -> none
    #
    # 채팅에서 사고를 끄는 이득이 큰 이유는 **스트리밍**이기 때문이다. 사고 토큰
    # 구간에는 vLLM 의 qwen3 파서가 content 를 빈 문자열로 보내므로, 사용자는
    # 그 시간 동안 아무것도 못 본다. 2026-09-05 실측(OpenRouter, 같은 질문):
    #
    #   effort   첫 글자까지   빈 청크 비율
    #   medium   6.1초         19%
    #   low      3.3초         33%
    #   none     1.4초          1%
    #
    # 빈 문자열로 총 시간이 줄지는 않지만, 사용자가 체감하는 "멈춰 있는 시간"이
    # 4배 이상 줄어든다. 빈 청크가 사라지는 것은 덤이다.
    #
    # 빈 문자열이면 LLM_REASONING_EFFORT 를 그대로 쓴다.
    CHAT_REASONING_EFFORT: str = "none"
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
    # 한 번의 LLM 호출로 만들 문제 수의 상한.
    #
    # 문제를 하나씩 병렬로 만들면 서로를 못 봐서 중복이 심하다
    # (2026-09-08 측정: 같은 주제 3문제에서 중복 발동률 77%, 30세트 중 11건은
    #  유사도 0.95 이상). 한 번에 쓰게 하면 모델이 나란히 놓고 쓰므로 사라진다.
    #
    # 무한정 키울 수 없는 이유는 출력 토큰이다. 한 문제당 한국어 해설까지
    # 400~700 토큰이고 사고 토큰도 얹히므로 LLM_MAX_TOKENS(4096)에 닿는다.
    # 3 이면 기본 개수(PROMPT_QUIZ_DEFAULT_COUNT=3)가 한 번에 처리된다.
    # 그보다 많이 요청하면 여러 덩어리로 나눠 병렬 호출한다.
    QUIZ_BATCH_MAX: int = 3
    # 묶음 생성에서 문제 하나가 추가될 때마다 늘려줄 출력 토큰.
    #
    # 2026-09-08 실측 사고: 3개 묶음이 LLM_MAX_TOKENS(4096)에 닿아 JSON 이 중간에
    # 잘렸고, 3회 재시도를 전부 소진해 500 이 났다 (topic=부채비율, 3회 연속).
    # 운이 아니라 구조적이다 - 사고 토큰은 호출당 한 번이지만 본문은 개수에 비례한다.
    #
    # 그래서 묶음 호출만 예산을 늘린다. LLM_MAX_TOKENS 를 전역으로 올리면
    # 채팅 응답 길이까지 같이 늘어난다(build_llm 이 그 값을 공유한다).
    # 한국어 객관식 1문항(문제+선택지4+해설)이 대략 600~800 토큰이라 900 을 둔다.
    QUIZ_BATCH_TOKENS_PER_ITEM: int = 900
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
    # 세트 전체를 LLM 1회로 검사하는 단계(quality.validate_questions)를 어떻게 돌릴지.
    #
    #   blocking   - 검사하고 불합격 문제를 재생성한 뒤 응답한다 (원래 동작)
    #   background - 먼저 응답하고, 검사는 뒤에서 돌려 불합격만 로그에 남긴다 (기본)
    #   off        - 아예 하지 않는다
    #
    # 2026-09-08 계측: 프롬프트 퀴즈 3문제 81.5초 중 **이 단계가 24.7초(30%)**였다.
    # 그런데 그날 실행한 모든 케이스(파드 부하 20건 포함)에서 "전체 합격"이었다.
    # 이 검사는 9b/35B-A3B 시절의 "정답/해설 모순"을 잡으려고 만든 것인데,
    # 27B 로 바꾸면서 그 결함 유형이 사라졌다는 것이 모델 선정 문서 실험 3의 결론이다.
    # 결정론적 검사(한국어 가드·선택지 번호·주제 이탈)는 LLM 없이 그대로 돈다.
    #
    # 그래도 표본이 하루치라 완전히 빼지 않고 background 를 기본으로 뒀다.
    # 지연은 사라지면서 "무엇이 걸렸는지" 로그는 계속 쌓인다. 며칠 보고 결정할 것.
    #
    # ⚠️ background 는 **체감 지연만** 줄인다. GPU 작업량은 그대로다.
    # 동시 사용자 처리량이 문제라면 off 로 두어야 실제로 일이 줄어든다.
    QUIZ_VALIDATE_MODE: str = "background"
    # 프롬프트 분석 단계(자유 프롬프트 -> 주제/개수/유형/카테고리) 전용 사고 강도.
    #
    # 이 단계는 추론이 아니라 구조화된 추출 작업이다. 채팅과 같은 이유로 사고를
    # 낮출 여지가 크다 (계측: 분석 한 번에 19.8초 = 전체의 24%).
    # 잘못 뽑아도 _validate_plan_item 이 카탈로그에 없는 코드를 걸러내고
    # _enforce_question_type 이 유형을 바로잡는 안전망이 있다.
    #
    # 빈 문자열이면 LLM_REASONING_EFFORT 를 그대로 쓴다.
    ANALYZE_REASONING_EFFORT: str = "none"

    # ── 하트비트 응답 (Cloudflare 100초 벽 우회) ──
    #
    # RunPod HTTP 프록시는 **응답 첫 바이트까지 100초**를 기다린 뒤 524 로 끊는다.
    # 켜 두면 오래 걸리는 응답에 공백을 주기적으로 흘려보내 그 벽을 없앤다.
    # JSON 값 앞의 공백은 문법상 무시되므로 백엔드 코드는 그대로 둬도 된다.
    # 자세한 배경과 상태 코드 제약은 app/core/keepalive.py 문서 참고.
    #
    # 끄고 싶어질 상황: 중간 프록시가 application/json 을 버퍼링해 하트비트가
    # 소용없거나, 오히려 문제를 만들 때. 그때 이미지 재빌드 없이 템플릿 env 로
    # 되돌릴 수 있어야 해서 설정으로 뺐다.
    RESPONSE_KEEPALIVE_ENABLED: bool = True
    # 첫 박동까지의 대기. 이 시간 안에 끝나면 응답은 예전과 완전히 동일하고,
    # 예외도 정상적인 422/500 상태 코드로 나간다.
    #
    # 30초인 이유: 실제로 잡히는 예외(PromptTopicError = 주제 판별 실패)는
    # 분석 단계에서 나므로 약 4초면 결정된다. 30초면 그 여유를 충분히 벌면서
    # 100초 벽까지 70초를 남긴다. 첫 박동이 늦을수록 상태 코드가 살고,
    # 이를수록 벽에서 멀어지는 맞교환이다.
    RESPONSE_KEEPALIVE_DELAY_SEC: float = 30.0
    # 박동 간격. Cloudflare 의 무응답 허용치보다 넉넉히 짧아야 한다.
    RESPONSE_KEEPALIVE_INTERVAL_SEC: float = 10.0

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