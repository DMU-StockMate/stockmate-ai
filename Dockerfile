# StockMate AI — 단일 파드 서빙 이미지
#
# vLLM + Qdrant + FastAPI 를 한 컨테이너에서 띄운다.
# 목적: 베타/발표 당일에 파드 하나만 올리면 공개 URL 이 나오게 하는 것.
# 로컬 PC 는 꺼져 있어도 된다.
#
#   빌드 : docker build -t <user>/stockmate-ai:<tag> .
#   실행 : RunPod 파드 (Network Volume 을 /workspace 에 마운트)
#
# 베이스는 2026-08-30 리허설에서 실제로 검증한 vllm/vllm-openai 다이제스트를
# 고정한다. :latest 를 쓰면 발표 당일에 다른 빌드가 내려와 검증이 무효가 된다.
FROM vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14

ARG QDRANT_VERSION=1.18.2

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Qdrant — 정적 musl 바이너리. 도커 인 도커가 안 되므로 바이너리로 넣는다.
# ---------------------------------------------------------------------------
RUN curl -fsSL -o /tmp/qdrant.tar.gz \
      "https://github.com/qdrant/qdrant/releases/download/v${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-musl.tar.gz" \
 && tar -xzf /tmp/qdrant.tar.gz -C /usr/local/bin qdrant \
 && rm /tmp/qdrant.tar.gz \
 && chmod +x /usr/local/bin/qdrant \
 && qdrant --version

COPY docker/qdrant-config.yaml /opt/qdrant/config/config.yaml

# ---------------------------------------------------------------------------
# 앱 의존성
#
# 베이스 이미지의 torch / transformers / vllm 버전을 constraint 로 못박아
# pip 가 이들을 갈아엎지 못하게 한다. 갈아엎어야만 설치되는 의존성이 있다면
# 여기서 빌드가 깨진다 — 시간당 $3.29 짜리 파드가 아니라 빌드에서 잡는 게 맞다.
# ---------------------------------------------------------------------------
COPY requirements-server.txt /tmp/requirements-server.txt
RUN python3 -c "import importlib.metadata as m; print('\n'.join(f'{p}=={m.version(p)}' for p in ('torch','transformers','tokenizers','numpy','vllm')))" \
      > /tmp/constraints.txt \
 && echo '--- pinned ---' && cat /tmp/constraints.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/requirements-server.txt

# ---------------------------------------------------------------------------
# 앱
# ---------------------------------------------------------------------------
WORKDIR /app
COPY app/ /app/app/
COPY scripts/ /app/scripts/
COPY docker/entrypoint.sh /usr/local/bin/stockmate-entrypoint
RUN chmod +x /usr/local/bin/stockmate-entrypoint

# 빌드 게이트: vLLM 과 앱이 같은 인터프리터에서 함께 살아있는지 확인한다.
RUN python3 -c "import vllm; print('vllm', vllm.__version__)" \
 && python3 -c "import fastapi, langchain, qdrant_client, sentence_transformers; print('app deps ok')" \
 && python3 -c "import app.main; print('app.main import ok')"

# ---------------------------------------------------------------------------
# 런타임 기본값.
#
# **튜닝 값(LLM_*, VLLM_*, EMBEDDING_DEVICE)의 단일 출처는 여기다.**
# 로컬 .env 에서 값을 바꿔 좋아졌다면 여기도 같이 고치고 다시 빌드해야
# 서버에 반영된다. RunPod 템플릿 env 는 비밀값만 담는 것을 원칙으로 한다
# (템플릿에 같은 키를 넣으면 이 값을 덮어쓰므로 드리프트의 원인이 된다).
#
# 비밀값(API 키)은 넣지 않는다 — 이미지가 public 이다. 파드 env 로 주입한다.
# ---------------------------------------------------------------------------
ENV HF_HOME=/workspace/hf \
    QDRANT_STORAGE=/workspace/qdrant_storage \
    APP_PORT=8080 \
    \
    LLM_BACKEND=openai \
    LLM_BASE_URL=http://127.0.0.1:8000/v1 \
    LLM_MODEL=qwen3.8-27b \
    LLM_DISABLE_THINKING=False \
    LLM_REASONING_EFFORT=medium \
    CHAT_REASONING_EFFORT=none \
    LLM_MAX_TOKENS=4096 \
    \
    QUIZ_GEN_CONCURRENCY=5 \
    QUIZ_DUP_CHECK_PAST=false \
    PROMPT_QUIZ_DEFAULT_COUNT=3 \
    \
    QDRANT_HOST=127.0.0.1 \
    QDRANT_PORT=6333 \
    EMBEDDING_DEVICE=cuda \
    \
    VLLM_MODEL=Qwen/Qwen3.8-27B-FP8 \
    VLLM_SERVED_NAME=qwen3.8-27b \
    VLLM_MAX_MODEL_LEN=16384 \
    VLLM_GPU_MEM_UTIL=0.88 \
    VLLM_MAX_NUM_SEQS=32 \
    VLLM_REASONING_EFFORT=medium

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/stockmate-entrypoint"]
