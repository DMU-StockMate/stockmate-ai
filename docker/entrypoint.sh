#!/usr/bin/env bash
# StockMate AI 파드 기동 스크립트 — Qdrant → vLLM → FastAPI 순서로 띄운다.
#
# 순서가 중요하다. FastAPI 는 기동 시 DART 종목코드를 받고 첫 요청에서
# 임베딩 모델을 올리는데, 그 전에 Qdrant 와 vLLM 이 떠 있어야 한다.
# 셋 중 하나라도 죽으면 컨테이너 전체를 내려서 RunPod 로그에 드러나게 한다
# (조용히 반쪽만 살아있는 상태가 가장 디버깅하기 나쁘다).
set -uo pipefail

log() { echo "[entrypoint $(date -u +%H:%M:%S)] $*"; }

: "${HF_HOME:=/workspace/hf}"
: "${QDRANT_STORAGE:=/workspace/qdrant_storage}"
: "${APP_PORT:=8080}"
: "${VLLM_PORT:=8000}"
: "${VLLM_API_KEY:=}"
# 앱이 vLLM 에 붙을 때 쓰는 키는 vLLM 에 준 키와 같아야 한다.
: "${LLM_API_KEY:=${VLLM_API_KEY}}"
export LLM_API_KEY

mkdir -p "${HF_HOME}" "${QDRANT_STORAGE}"

PIDS=()
cleanup() {
  log "종료 신호 — 자식 프로세스 정리"
  for pid in "${PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

die() { log "치명적: $*"; exit 1; }

wait_for() {  # wait_for <이름> <URL> <최대초>
  local name="$1" url="$2" limit="$3" waited=0
  log "${name} 대기 중 (최대 ${limit}s)..."
  while ! curl -fsS -o /dev/null --max-time 5 "${url}" 2>/dev/null; do
    sleep 3; waited=$((waited + 3))
    if [ "${waited}" -ge "${limit}" ]; then
      die "${name} 가 ${limit}s 안에 준비되지 않았다 (${url})"
    fi
    if [ $((waited % 60)) -eq 0 ]; then log "  ${name} ... ${waited}s"; fi
  done
  log "${name} 준비 완료 (${waited}s)"
}

# --------------------------------------------------------------------------
# 1. Qdrant
# --------------------------------------------------------------------------
log "Qdrant 시작 (storage=${QDRANT_STORAGE})"
(
  cd /opt/qdrant || exit 1
  QDRANT__STORAGE__STORAGE_PATH="${QDRANT_STORAGE}" \
  QDRANT__SERVICE__HTTP_PORT="${QDRANT_PORT:-6333}" \
    exec qdrant 2>&1 | sed -u 's/^/[qdrant] /'
) &
PIDS+=($!)
wait_for "Qdrant" "http://127.0.0.1:${QDRANT_PORT:-6333}/readyz" 120

# --------------------------------------------------------------------------
# 2. vLLM
#    2026-08-30 리허설에서 검증한 인자 그대로.
#    JSON 값은 exec 형식이라 셸을 안 거치므로 따옴표를 덧붙이지 않는다.
# --------------------------------------------------------------------------
VLLM_ARGS=(
  "${VLLM_MODEL}"
  --served-model-name "${VLLM_SERVED_NAME}"
  --host 127.0.0.1
  --port "${VLLM_PORT}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}"
  --kv-cache-dtype fp8
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
  --reasoning-parser qwen3
  --default-chat-template-kwargs "{\"reasoning_effort\":\"${VLLM_REASONING_EFFORT}\"}"
)
[ -n "${VLLM_API_KEY}" ] && VLLM_ARGS+=(--api-key "${VLLM_API_KEY}")

log "vLLM 시작: ${VLLM_MODEL} (gpu-mem-util=${VLLM_GPU_MEM_UTIL})"
( exec vllm serve "${VLLM_ARGS[@]}" 2>&1 | sed -u 's/^/[vllm] /' ) &
PIDS+=($!)
# 가중치가 볼륨에 있으면 5~6분, 처음 받으면 그 이상. 넉넉히 준다.
wait_for "vLLM" "http://127.0.0.1:${VLLM_PORT}/health" 1500

# --------------------------------------------------------------------------
# 3. FastAPI — 이게 포그라운드다. 죽으면 컨테이너가 죽는다.
# --------------------------------------------------------------------------
# 워커는 1개로 둔다. 늘리면 bge-m3 임베딩 모델이 워커 수만큼 GPU 에 중복
# 적재되어 vLLM 이 쓸 VRAM 을 갉아먹는다. 앱은 async 라 1워커로 충분하다.
log "FastAPI 시작 (:${APP_PORT}, embedding=${EMBEDDING_DEVICE})"
exec uvicorn app.main:app \
  --host 0.0.0.0 --port "${APP_PORT}" \
  --workers 1 --no-access-log 2>&1 | sed -u 's/^/[app] /'
