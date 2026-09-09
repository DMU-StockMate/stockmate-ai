# RunPod H100 서빙 런북 — Qwen3.8-27B FP8

작성 2026-08-30. 리허설 전 초안 (§7 결과란은 리허설 후 채운다).
가격은 2026-08-30 runpod.io/pricing 재확인 값.

**베타 당일에는 §8 만 보면 된다.** §1~§6 은 리허설용 상세 절차다.

---

## 0. 비용 (실행 전 확인)

| 항목 | 단가 | 리허설(1.5h) | 베타+발표(30h) |
|---|---|---|---|
| H100 80GB PCIe | $2.89/h | $4.34 | $86.70 |
| H100 80GB **SXM** (PCIe 재고 없을 때) | $3.29/h | $4.94 | $98.70 |
| Network Volume 80GB (Standard) | $0.07/GB/월 = **$5.60/월** | $5.60 (월 단위 청구) | 위와 동일 계정 청구 |
| Container Disk 30GB | $0.10/GB/월 ≈ $0.004/h | $0.01 | $0.12 |
| **합계 (PCIe)** | | **약 $10 (1.4만원)** | **약 $98 (13.5만원)** |
| **합계 (SXM)** | | **약 $11 (1.5만원)** | **약 $110 (15.2만원)** |

**PCIe / SXM 어느 쪽이든 무방하다.** 둘 다 80GB 라 FP8 27B 가 올라가는지 여부는
바뀌지 않는다. SXM 이 오히려 상위 모델이다 — HBM3 3.35TB/s vs PCIe 의 HBM2e 2.0TB/s.
디코딩은 메모리 대역폭에 묶이므로 SXM 쪽이 토큰 생성이 더 빠르다.
시급 14% 더 내고 대역폭 1.6배를 받는 셈. 예산 상한 19만원 안에도 들어간다.
**재고 있는 쪽을 잡으면 된다.** 명령·설정은 한 글자도 바뀌지 않는다.
(SXM 은 NVLink 가 강점인데 1장 서빙에는 안 쓰인다. 그 부분만 값을 못 건진다.)

주의할 점 두 가지.

1. **Network Volume 은 파드를 지워도 계속 과금된다.** 월 단위 $5.60 이
   발표일까지 계속 나간다. 8/30 생성 → 발표일까지 두 달이면 $11.2.
   그래도 재다운로드 30분 × H100 시급을 아끼는 편이 싸다.
2. **파드를 Stop 만 해도 Volume Disk 는 계속 과금된다.** 그래서 파드 자체 디스크에
   가중치를 두면 안 되고 Network Volume 을 쓴다 (모델 선정 문서 "서버 구성안" 참고).

리허설을 1시간으로 잡았지만 **실측은 75~90분**을 예상하는 게 맞다.
가중치 28GB 다운로드와 재기동 검증(§5) 때문이다. 그래도 $5 안쪽 차이다.

---

## 1. 사전 준비 (과금 없음, 로컬에서)

- Hugging Face 토큰 발급 (read 권한). `Qwen/Qwen3.8-27B-FP8` 은 Apache 2.0 이라
  gated 는 아니지만, 토큰이 있으면 다운로드 rate limit 를 안 맞는다.
- RunPod 계정에 결제수단 등록 + **크레딧 $25 정도 선충전**.
  잔액 0 이 되면 파드가 예고 없이 종료되고 Network Volume 도 위험해진다.
- vLLM 접근 키를 하나 정한다 (예: `sk-stockmate-<랜덤>`).
  프록시 URL 은 추측 가능한 주소라 `--api-key` 없이 띄우면 공개 서버가 된다.

---

## 2. 리전 결정 → Network Volume 생성

**RunPod MCP 로 2026-08-30 실측 완료. 아래는 조사 결과가 아니라 확정값이다.**

### 2-1. GPU 재고 실측

| GPU | VRAM | Secure 시급 | 전체 재고 |
|---|---|---|---|
| H100 PCIe | 80GB | $2.89 | **NONE — 전 데이터센터 재고 0** |
| **H100 SXM** | 80GB | **$3.29** | **HIGH** |
| H100 NVL | 94GB | $3.19 | LOW |

H100 PCIe 는 일시적 품절이 아니라 데이터센터 목록 자체가 비어 있다.
**SXM 으로 간다.** (성능은 오히려 SXM 이 위 — §0 참고)

### 2-2. 리전 — H100 SXM × Network Volume 교집합

**함정: SXM 이 있는 데이터센터라고 Network Volume 을 만들 수 있는 게 아니다.**
둘을 교차한 결과, 쓸 수 있는 곳은 셋뿐이다.

| 데이터센터 | SXM 재고 | Volume 종류 | 판정 |
|---|---|---|---|
| **AP-JP-1** (일본) | LOW | STANDARD | ✅ **1순위** |
| US-NE-1 | LOW | STANDARD | ✅ 2순위 |
| EUR-IS-3 | LOW | STANDARD | ✅ 3순위 |
| EU-FR-1 / EUR-NO-2 / US-CA-2 | LOW | HIGH_PERFORMANCE 만 | △ $0.14/GB 로 2배. 이득 없음 |
| **AP-IN-1** | **MEDIUM (최다)** | **없음** | ❌ **재고가 제일 많은데 볼륨을 못 만든다** |
| CA-MTL-1 / US-GA-2 / US-MO-1 | LOW | 없음 | ❌ |

AP-IN-1 이 이 런북이 존재하는 이유다. Deploy 화면만 보고 골랐으면
재고 제일 많은 AP-IN-1 을 잡았을 텐데, 거기엔 Network Volume 이 아예 없어서
목표 2번(가중치 유지)이 통째로 불가능하다.

**AP-JP-1 로 간다.** STANDARD 볼륨이 되고, 한국에서 가장 가깝다.
(RTT 이득은 생성 15초짜리 작업에서 체감 차이가 크진 않다. 다만 같은 $3.29 에
공짜로 얻는 이득이라 안 챙길 이유가 없다.)

### 2-3. 재고 리스크 — 베타 당일 대비

**SXM 은 세 후보 전부 LOW 다.** 발표 당일 AP-JP-1 이 비어 있을 수 있다.

- 당일 §8 첫 단계에서 재고부터 확인한다 (MCP 로 조회하면 10초).
- AP-JP-1 이 막히면 → US-NE-1 → EUR-IS-3 순으로 내려간다.
- **다만 볼륨은 리전 고정이라, 다른 리전으로 가면 28GB 재다운로드다** (10~20분).
- 보험을 원하면 US-NE-1 에도 볼륨을 하나 더 만든다. 월 $5.60 추가.
  졸업작품 발표 당일 20분을 날리는 것보다 싸다면 만들 것. **리허설 단계에서는 불필요.**

### 2-4. 볼륨 생성

Storage 페이지(console.runpod.io/user/storage) → `New Network Volume`

| 항목 | 값 |
|---|---|
| Datacenter | **AP-JP-1** |
| Name | `stockmate-weights` |
| Size | **80 GB** (나중에 늘릴 수는 있어도 줄일 수 없다) |
| Tier | **Standard** — $0.07/GB/월 = **$5.60/월** |

→ **여기서부터 월 $5.60 과금 시작.** 파드를 지워도 계속 나간다.

### 2-5. 생성 완료 (2026-08-30)

```
Volume ID    htdeoj8rmu
Name         stockmate-weights
Data Center  AP-JP-1
Size         80 GB
Type         STANDARD  ($5.60/월)
```

**이 볼륨은 이미 존재한다. 베타 당일에 다시 만들 필요 없다.**
파드 생성 시 이걸 선택하기만 하면 된다.

## 3. 템플릿과 파드 배포

### 3-1. 템플릿 — 생성 완료 (2026-08-30)

**다시 만들 필요 없다.** Deploy 화면에서 고르기만 하면 된다.

```
Template ID    beonudgm8k
Name           stockmate-vllm
Image          vllm/vllm-openai:latest
Container Disk 30 GB
Mount Path     /workspace
Ports          8000/http, 8000/tcp
Env            HF_HOME=/workspace/hf
               HF_TOKEN=<발급본. 문서에 적지 않는다>
               VLLM_API_KEY=sk-stockmate-7f3a91c4e2
Args           Qwen/Qwen3.8-27B-FP8 --served-model-name qwen3.8-27b
               --host 0.0.0.0 --port 8000 --max-model-len 16384
               --gpu-memory-utilization 0.92 --kv-cache-dtype fp8
               --max-num-seqs 32 --reasoning-parser qwen3
               --default-chat-template-kwargs '{"reasoning_effort":"medium"}'
```

두 가지가 실측으로 확정됐다.

1. **ENTRYPOINT 는 `["vllm","serve"]`** — Args 는 모델 ID 부터 시작한다.
   `vllm serve` 를 앞에 다시 붙이면 `usage: vllm serve [model_tag]` 로 죽는다.
2. **JSON 값은 반드시 작은따옴표로 감싼다.** RunPod 는 args 문자열을 shlex 로
   쪼개면서 큰따옴표를 제거한다. 감싸지 않으면
   `{"reasoning_effort":"medium"}` 이 `{reasoning_effort:medium}` 으로 도착해
   `invalid loads value` 로 죽는다. 1차 시도가 여기서 실패했다.

템플릿 기본값으로 `startJupyter` / `startSsh` 가 켜져 있는데, 이 이미지에는
jupyter 가 없어 무시된다. 문제되면 UI 에서 끈다.

### 3-2. 파드 배포 — 이 단계만 UI 에서 (MCP 불가)

**MCP 의 `create-pod` 로는 Network Volume 을 붙일 수 없다.** 확인한 곳:
`create-pod` / `update-pod` / `create-template` / `update-template` 넷 다
`volumeInGb` + `volumeMountPath` 만 받고, 기존 볼륨을 가리키는 필드가 없다.
(원본 REST v2 는 `mounts.network[].volumeId` 로 받지만 MCP 가 투영하지 않았다.
템플릿 응답의 `mounts` 도 읽기 전용으로만 보인다.)

`volumeInGb` 로 대신하면 **안 된다** — 그건 파드 Volume Disk 를 새로 만드는
것이라 정지 시 $0.20/GB/월, 파드를 지우면 가중치도 사라진다.

Pods → Deploy 에서 드롭다운 세 개:

1. **Network Volume** = `stockmate-weights` (AP-JP-1) ← **먼저 고른다**
2. **GPU** = `H100 SXM` × 1, Secure Cloud
3. **Pod Template** = `stockmate-vllm`

나머지(디스크·포트·env·시작명령)는 템플릿이 채운다. Deploy 를 누르면
**$3.29/h 과금 시작.**

파드가 뜬 뒤부터는 전부 MCP 로 처리 가능하다 —
`get-pod`(프록시 URL), `stream-pod-logs`(기동 감시), `stop-pod` / `delete-pod`,
`get-billing`(실지출).

## 4. 검증 1 — 모델이 H100 한 장에 올라가는가

파드 Logs 탭을 본다. 첫 기동은 이미지 pull + 28GB 다운로드로 **10~20분**.

성공 신호:
```
INFO ... Starting vLLM API server on http://0.0.0.0:8000
INFO ... GPU KV cache size: ... tokens
```

`GPU KV cache size` 숫자를 적어둘 것. 이게 동시성 여유의 실측치다.

실패하면 흔한 원인:
- `CUDA out of memory` → `--gpu-memory-utilization 0.90` 으로 낮추거나
  `--max-model-len 8192` 로 줄인다.
- `ValueError: ... rope_scaling` / 모델 인식 실패 → vLLM 이미지가 낡음.
  `vllm/vllm-openai:latest` 대신 명시 태그(0.17 이상)로 다시 배포.

**스모크 테스트** (로컬 PowerShell):
```powershell
$POD="<POD_ID>"; $KEY="sk-stockmate-..."
curl.exe -s "https://$POD-8000.proxy.runpod.net/v1/models" -H "Authorization: Bearer $KEY"
```

---

## 5. 검증 2 — 가중치가 남는가 → **통과 (2026-08-30 실측)**

파드를 Terminate 하고 같은 볼륨으로 재배포한 결과:

```
weight_utils.py:858  Filesystem type for checkpoints: FUSE. Checkpoint size: 28.75 GiB
default_loader.py    Loading weights took 54.54 seconds
```

`FUSE` = 마운트된 Network Volume. **다운로드 단계가 통째로 사라지고 볼륨에서 55초에 읽었다.**
`HF_HOME=/workspace/hf` 가 제대로 동작한다.

### 그런데 총 기동 시간은 줄지 않았다 — 이게 중요하다

| 단계 | 1차 기동 (13:21:27→13:28:36) | 2차 기동 (13:46:13→13:56:23) |
|---|---|---|
| 이미지 pull | 캐시 히트 (2초) | **3분 20초** (호스트가 바뀜) |
| 가중치 | 다운로드 (약 1분) | 볼륨에서 55초 |
| 엔진 초기화 | 242.9s | 234.8s |
| **합계** | **7분 09초** | **10분 10초** |

세 가지를 알 수 있다.

1. **볼륨이 아끼는 건 다운로드뿐이다.** 엔진 초기화 4분은 매번 그대로 든다
   (torch.compile 32초 + DeepGEMM 워밍업 2분 11초 + CUDA 그래프 캡처).
   그 캐시는 `/root/.cache/vllm` 즉 **컨테이너 디스크**에 있어서 파드와 함께 사라진다.
2. **이미지 pull 은 호스트 운이다.** 같은 리전이라도 다른 물리 머신에 배정되면
   20GB대 이미지를 다시 받는다. 3분 20초.
3. 따라서 **베타 당일에는 7분이 아니라 10~12분을 잡아야 한다.**
   서비스 오픈 15분 전에 Deploy 를 누르는 것이 안전하다.

### 더 줄이고 싶다면 (미검증)

템플릿 env 에 `VLLM_CACHE_ROOT=/workspace/vllm_cache` 를 추가하면
torch.compile / DeepGEMM / FlashInfer 캐시도 볼륨에 남아 엔진 초기화가 짧아질 수 있다.
**이번 리허설에서 검증하지 않았다.** 당일에 처음 켜보지 말고, 쓸 거라면
미리 한 번 파드를 띄워 확인할 것. 지금 설정으로도 10~12분이면 충분하다면 건드리지 말 것.

## 6. 검증 3 — FastAPI 가 붙는가

코드 변경 없이 환경변수만 바꾼다 (`.env` 는 건드리지 않는다 —
pydantic-settings 는 셸 환경변수가 `.env` 보다 우선).

```powershell
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$env:LLM_BACKEND="openai"
$env:LLM_BASE_URL="https://<POD_ID>-8000.proxy.runpod.net/v1"
$env:LLM_API_KEY="sk-stockmate-..."
$env:LLM_MODEL="qwen3.8-27b"
$env:LLM_DISABLE_THINKING="False"
$env:LLM_REASONING_EFFORT="medium"
$env:LLM_MAX_TOKENS="2048"
$env:QUIZ_BANK_ENABLED="false"

uv run python scripts/compare_models.py --only quiz
```

**여기서 반드시 확인할 것 하나.** `app/core/llm.py` 는 지금
`reasoning_effort` 를 요청 **본문 최상위**에 넣는다:

```python
extra_body["reasoning_effort"] = settings.LLM_REASONING_EFFORT
```

그런데 Qwen3.8 모델 카드는 `chat_template_kwargs` **안에** 넣으라고 한다.
vLLM 버전에 따라 최상위 필드가 채팅 템플릿까지 전달되지 않을 수 있다.
그러면 기본값 `xhigh` 로 돌아가 응답이 미친 듯이 느려진다.

판정법: 응답 시간이 케이스당 60초를 넘거나 `<think>` 블록이 길게 나오면
전달이 안 된 것이다. 두 가지 대응 중 하나:

- **권장**: §3 의 `--default-chat-template-kwargs` 로 서버에서 못박는다.
  코드를 안 건드려도 되고 베타 당일 실수 여지가 없다.
- 대안: `llm.py` 를 고쳐 `extra_body["chat_template_kwargs"]["reasoning_effort"]`
  로 중첩시킨다.

또한 `LLM_DISABLE_THINKING=True` 이면 `chat_template_kwargs={"enable_thinking": False}`
가 함께 나가는데, Qwen3.8 템플릿이 이 인자를 모를 수 있다.
**3.8 서버에 붙일 때는 `False` 로 둔다** (로컬 3.6 에서는 계속 `True`).

---

## 7. 검증 4 — 동시 20명

`scripts/loadtest_llm.py` (이번에 추가).

```powershell
uv run python scripts/loadtest_llm.py `
  --base-url "https://<POD_ID>-8000.proxy.runpod.net/v1" `
  --model qwen3.8-27b `
  --api-key "sk-stockmate-..." `
  --concurrency 20 --rounds 2
```

합격선 (베타 기준):
- 성공 20/20, HTTP 5xx·타임아웃 0
- p95 지연 40초 이내
- 라운드 2가 라운드 1보다 빠를 것 (prefix caching 이 도는지 확인)

넘치면 조정 순서: `--max-num-seqs` 를 48로 올린다 → 그래도 안 되면
`LLM_MAX_TOKENS` 를 낮춘다 → 그래도 안 되면 문제 은행 사전 생성으로
당일 실시간 생성량 자체를 줄인다 (모델 선정 문서 "남은 과제" 3번).

### 리허설 실측 결과 — 2026-08-30 (pod `xo8ilj9xju86jc`, H100 SXM, AP-JP-1)

| 항목 | 결과 |
|---|---|
| 첫 기동 시간 (28GB 다운로드 포함) | **7분 09초** (생성 13:21:27 → startup complete 13:28:36) |
| 엔진 초기화 (프로파일·컴파일·워밍업) | 242.9s (torch.compile 32.6s, DeepGEMM 워밍업 2분 13초) |
| **재기동 시간 (Volume 캐시 히트)** | **10분 10초** — §5 참고. 다운로드는 사라졌지만 이미지 pull 이 새로 붙었다 |
| **GPU KV cache size** | **807,634 tokens — 16k 요청 기준 동시 49.29개** |
| 가중치 + non-torch | 29.17 GiB |
| KV 캐시 메모리 | 40.14 GiB (여유 5.8 GiB 더 있음) |
| 단건 지연 (245토큰) | **3.5초 ≈ 70 tok/s** |
| `reasoning_effort=medium` 적용 | **확인** — reasoning_tokens 120 (xhigh 였으면 수천) |
| 동시 20 성공률 (TCP 직결) | **20/20** |
| 동시 20 성공률 (HTTPS 프록시) | **20/20 × 2라운드** |
| 동시 20 지연 | p50/p95/max 전부 **15.0초** (연속 배칭) |
| 동시 20 집계 처리량 | **992 tok/s** (TCP) / **1,029 tok/s** (프록시 2라운드) |
| prefix caching 동작 | **확인** — 프록시 2라운드가 19.1s → 14.4s 로 단축 |
| API 키 인증 | **확인** — 키 없이 요청 시 401 |
| `compare_models --only quiz` | **12/12 통과, 166.6초** (평균 13.9s/케이스) |
| 생성 품질 | 35B 결함 3건(ROE 당기가손익 / 주가총액 / 영업이익 선택지) **전부 정확** |
| OX 정답 분포 | O 4 / X 2 (35B 는 6문항 전부 O) |
| Start Command 형태 | 모델ID 부터 (ENTRYPOINT 가 `vllm serve`) |
| JSON 인자 따옴표 | **작은따옴표 필수** |
| vLLM 버전 | 0.28.0 |
| 실제 리전 / GPU | AP-JP-1 / NVIDIA H100 80GB HBM3 (SXM) |

**판정: 네 가지 목표 전부 통과.**

1. ✅ Qwen3.8-27B FP8 이 H100 80GB 한 장에 올라간다 (가중치 28.5GiB, KV 캐시 40GiB)
2. ✅ Network Volume 에 가중치가 남아 재다운로드가 없다 (FUSE 에서 55초 로드)
3. ✅ FastAPI 가 `LLM_BASE_URL` 만 바꿔서 붙는다 (코드 변경 0, 12/12 통과)
4. ✅ 동시 20명을 견딘다 (p95 14.4초, 여유 49x)

**여유가 크다는 점이 중요하다.** 동시 49개까지 받는데 20명만 쓰고,
p95 가 15초라 Cloudflare 100초 제한과도 멀다. 베타 규모에서 병목은 GPU 가 아니다.

### 성능 튜닝 여지 (필요해지면)

- `--gpu-memory-utilization 0.9413` 으로 올리면 KV 캐시를 45.4GiB 까지 쓸 수 있다.
  (vLLM 이 CUDA 그래프 프로파일링 때문에 0.92 를 실효 0.8987 로 쓴다고 로그에 알려준다.)
  **지금은 불필요** — 49x 동시성이면 충분하다.
- `--max-num-seqs 32` 도 여유. 20명이면 건드릴 이유 없다.

## 8. 베타 당일 절차 — 이것만 보면 된다 (v2, 전체 스택 단일 파드)

리허설로 전부 검증된 절차다. 볼륨·템플릿·이미지는 **이미 만들어져 있다.**

```
Network Volume  stockmate-weights          htdeoj8rmu   (AP-JP-1, 80GB STANDARD)
Pod Template    stockmate-fullstack        c3bqwiolqi
이미지           dohun1214/stockmate-ai:0.1.2
GPU             NVIDIA H100 80GB HBM3 (SXM)  $3.29/h
공개 URL         https://<POD_ID>-8080.proxy.runpod.net
서비스 인증      X-API-Key: <.env 의 AI_API_KEY>   (/health, /docs, /openapi.json 은 예외)
```

폴백: 템플릿 `stockmate-vllm`(`beonudgm8k`)은 vLLM 만 띄우는 v1 구성이다. 지우지 말 것.

### T-20분 — 재고 확인

H100 SXM 이 **AP-JP-1** 에 있는지 본다. 없으면 US-NE-1 → EUR-IS-3 순이지만
**볼륨은 AP-JP-1 전용이라 다른 리전은 가중치 재다운로드**로 기동이 길어진다.

### T-15분 — 파드 생성

RunPod 콘솔에서 드롭다운 세 개(Network Volume → GPU → Template)를 고르거나,
REST 로 한 번에 만든다. **MCP `create-pod` 는 Network Volume 을 붙이지 못하므로
반드시 콘솔이나 아래 REST 를 쓸 것.**

```powershell
$key = ((Get-Content .env | Select-String '^RUNPOD_API_KEY=').Line -replace '^RUNPOD_API_KEY=','')
$body = @{
  name='stockmate-beta'; templateId='c3bqwiolqi'
  gpu=@{ id='NVIDIA H100 80GB HBM3'; count=1 }
  dataCenterIds=@('AP-JP-1'); cloud='SECURE'
  mounts=@{ network=@(@{ volumeId='htdeoj8rmu'; path='/workspace' }) }
} | ConvertTo-Json -Depth 6
(Invoke-RestMethod 'https://api.runpod.io/v2/pods' -Method Post `
   -ContentType 'application/json' -Headers @{Authorization="Bearer $key"} -Body $body).id
```

→ **$3.29/h 과금 시작.**

### T-15 ~ T-7분 — 기동 대기 (약 7분)

로그에서 이 순서가 나오면 정상이다.

```
[entrypoint] Qdrant 준비 완료
[vllm]       Loading weights took ~55 seconds     ← 볼륨 히트. 다운로드가 돌면 리전이 틀렸다
[vllm]       GPU KV cache size: 807,634 tokens
[entrypoint] vLLM 준비 완료 (~350s)
[app]        DART 종목 코드 로드 완료: 3958개
[app]        임베딩 모델 워밍업 완료 (~18s, device=cuda)
[app]        Uvicorn running on http://0.0.0.0:8080
```

**`임베딩 모델 워밍업 완료` 를 보고 나서 서비스를 열 것.** 이게 뜨기 전에 첫 요청이
들어가면 그 사용자가 18초를 더 기다린다.

### T-5분 — 스모크

```powershell
$u='https://<POD_ID>-8080.proxy.runpod.net'
Invoke-RestMethod "$u/health"        # 인증 불필요 -> {"status":"ok"}
$h=@{'X-API-Key'='<AI_API_KEY>'}
Invoke-RestMethod "$u/quiz/generate" -Method Post -ContentType 'application/json; charset=utf-8' -Headers $h `
  -Body ([Text.Encoding]::UTF8.GetBytes((@{user=@{user_id=1;investment_level='초급'};quiz_type='OX';topic='PER';count=1}|ConvertTo-Json -Depth 5)))
```

→ NestJS 팀에 `$u` 와 `AI_API_KEY` 를 전달하고 오픈.

### 종료 — 즉시 삭제

```powershell
Invoke-RestMethod "https://api.runpod.io/v2/pods/<POD_ID>" -Method Delete -Headers @{Authorization="Bearer $key"}
```

**Stop 이 아니라 삭제.** 가중치와 Qdrant 데이터는 볼륨에 남는다.
켜둔 채 잊는 것이 가장 비싼 실수다 (하루 방치 = $79).

### 증상별 대응

| 증상 | 원인 | 대응 |
|---|---|---|
| 502 / 404 | 포트 미노출 | 템플릿 ports 가 `8080/http` 인지 |
| 401 | API 키 누락 | `X-API-Key` 헤더. `/health` 는 예외 |
| `usage: vllm serve [model_tag]` | Args 에 `vllm serve` 중복 | v2 이미지는 Args 를 안 쓴다. 템플릿 args 를 비울 것 |
| `invalid loads value` | JSON 인자 큰따옴표가 먹힘 | 작은따옴표로 감쌀 것 |
| **`JSON 객체를 찾지 못함` 500** | **사고 토큰이 출력 예산을 먹어 본문 잘림** | **`LLM_MAX_TOKENS` 를 올릴 것 (현재 4096)** |
| 다운로드가 다시 돎 | 리전 불일치 또는 `HF_HOME` 누락 | 볼륨이 AP-JP-1 인지, env 확인 |
| Start Command 수정 필요 | — | 파드에서는 불가. 템플릿 고치고 **파드 재생성** |

## 8-0. 재배포 체크리스트 — 새 대화에서도 이대로 하면 된다

로컬에서 코드를 고친 뒤 서버에 반영하는 전체 절차다. **이 문서만 보고 실행 가능해야 한다.**
값들은 저장소의 `.env`(앱 설정)와 `.env.runpod`(RunPod 자격증명·ID)에 있다.

### 0. 이 절차가 필요한 때

앱 코드가 이미지에 구워져 있으므로 **로컬 코드 변경은 재빌드 없이는 서버에 반영되지 않는다.**
다만 매번 할 필요는 없다 — 일상 개발은 로컬 FastAPI + OpenRouter 로 하고,
배포는 베타 직전 / 피드백 반영 후 / 발표 전 동결 정도로 두세 번이면 충분하다.

### 개발 중에 이 두 가지만 지키면 배포가 순조롭다

기능 개발은 다른 세션에서 하더라도, 아래 두 가지는 그때 같이 해두어야 한다.
빠뜨리면 배포 시점에 발견되고 재빌드로 15분씩 날아간다.

| 개발 중 한 일 | 같이 해야 할 일 |
|---|---|
| 파이썬 패키지 추가 (`uv add ...`) | **`requirements-server.txt` 에도 추가** |
| `.env` 에서 튜닝 값 변경 | **`Dockerfile` ENV + `.env.example` 에도 반영** |
| 설정 항목 추가 (`config.py`) | 위와 같음 — 네 곳(`config.py` / `.env` / `Dockerfile` / `.env.example`) |
| API 키 재발급 | RunPod 템플릿 `c3bqwiolqi` 의 env 도 갱신 |

패키지 누락은 Dockerfile 끝의 `import app.main` 게이트가 빌드에서 잡아준다
(파드가 아니라 빌드에서 실패하는 것이 설계 의도다).

**튜닝 값 불일치는 이제 스크립트가 잡아준다.** 커밋 전에 한 번 돌릴 것:

```powershell
uv run python scripts/check_env_parity.py
```

Dockerfile ENV ↔ `.env` ↔ `.env.example` 를 대조하고, 같은 키가 두 번 정의된
경우(뒤엣것이 이겨 조용히 어긋난다)도 잡는다. 실제로 이걸로 `.env.example` 이
Ollama 시절 값으로 남아 있고 `LLM_MODEL` 이 중복 정의된 것을 찾았다.

**로컬과 서버가 달라야 정상인 항목은 네 개뿐이다** (스크립트의 화이트리스트):

| 항목 | 로컬 | 서버 |
|---|---|---|
| `LLM_BACKEND` / `LLM_BASE_URL` / `LLM_MODEL` | OpenRouter, `qwen/qwen3.8-27b` | vLLM, `qwen3.8-27b` |
| `EMBEDDING_DEVICE` | `cpu` | `cuda` |
| `QDRANT_HOST` | `localhost` | `127.0.0.1` |
| `APP_PORT` | `8000` | `8080` |

나머지(퀴즈 튜닝·하트비트·사고 강도)는 **전부 같은 값이어야 한다.**

### 일상 개발 환경 (참고)

```powershell
docker start qdrant                     # 컨테이너 이름 qdrant, 재시작 정책 없음
uv run python scripts/check_env_parity.py   # 서버 설정과 어긋났는지 먼저 확인
uv run uvicorn app.main:app --reload
```

백엔드가 다른 기기에서 붙는다면 `--host 0.0.0.0 --port 8000` 을 붙인다.
그때는 `.env` 의 `AI_API_KEY` 를 **서버와 같은 값으로 채울 것** — 비워 두면
인증이 꺼져서 401 경로를 로컬에서 못 잡는다(실제로 이것 때문에 베타에서
백엔드 401 을 늦게 발견했다).

LLM 은 `.env` 가 OpenRouter(`qwen/qwen3.8-27b`)를 보고 있다. 유휴 비용 0,
퀴즈 1건 약 $0.006. 서버와 같은 모델이라 품질 판단에 그대로 쓸 수 있다.

**로컬에서 재현되는 것 / 안 되는 것**

| | 로컬 | 비고 |
|---|---|---|
| 하트비트 응답 | ✅ 그대로 동작 | 30초 넘기면 공백 박동 |
| 인증(401) | ✅ `AI_API_KEY` 채우면 동일 | |
| SSE 빈 토큰 필터 | ✅ | |
| **프록시 100초 제한** | ❌ 안 나타남 | Cloudflare 가 없다 |
| **gzip 압축** | ❌ 안 나타남 | 계측 시 함정(§9) |
| 동시성/처리량 | ❌ 다름 | OpenRouter는 변동폭이 2배까지 난다 |

**`--reload` 함정**: 리로더를 죽여도 자식 프로세스가 포트 8000 과 로그 파일을
붙들고 있어 재시작이 조용히 실패한다. `taskkill /F /T /PID <리로더PID>` 로
트리째 죽일 것. 오래 띄워둘 때는 `--reload` 없이 쓰는 편이 낫다.

### 1. 버전 정하기

**태그를 절대 덮어쓰지 말 것.** 항상 올린다.
현재까지: `0.1.0` → `0.1.1` → `0.1.2` → `0.1.3` → `0.1.4` → `0.1.5` → `0.1.6` → **`0.1.7`**.
템플릿 `c3bqwiolqi` 는 `0.1.7` 을 가리킨다.

### 2. 튜닝 값 동기화 (빠뜨리기 쉬움)

로컬 `.env` 에서 `LLM_REASONING_EFFORT`, `LLM_MAX_TOKENS`, `EMBEDDING_DEVICE` 등을
바꿔서 좋아졌다면, **`Dockerfile` 의 ENV 블록에도 같은 값을 반영**한 뒤 빌드한다.
Dockerfile 이 이 값들의 단일 출처다. 안 맞추면 "로컬에선 되는데 서버에선 안 되는"
가장 찾기 어려운 버그가 된다.

`.env` 의 비밀값(API 키)은 RunPod 템플릿 env 에 따로 들어 있다. 키를 재발급했다면
템플릿 env 도 갱신할 것.

### 3. 빌드 · 푸시

```powershell
cd C:\Users\dhp13\Desktop\stockmate-ai
docker build -t dohun1214/stockmate-ai:<새버전> .
docker push dohun1214/stockmate-ai:<새버전>
```

빌드 게이트 3개가 로그에 보여야 한다: `vllm 0.28.0` / `app deps ok` / `app.main import ok`.
베이스 캐시가 있으면 빌드 3분, 푸시 2~5분(앱 레이어만 올라감).

### 4. 템플릿 갱신

RunPod 템플릿 `c3bqwiolqi`(`stockmate-fullstack`)의 이미지를 새 태그로 바꾼다.
MCP `update-template` 또는 콘솔. **`args` 는 비워둔 채로 유지한다** (이미지 ENTRYPOINT 가 처리).

### 5. 파드 생성

```powershell
$key = ((Get-Content .env.runpod | Select-String '^RUNPOD_API_KEY=').Line -replace '^RUNPOD_API_KEY=','')
$body = @{
  name='stockmate'; templateId='c3bqwiolqi'
  gpu=@{ id='NVIDIA H100 80GB HBM3'; count=1 }
  dataCenterIds=@('AP-JP-1'); cloud='SECURE'
  mounts=@{ network=@(@{ volumeId='htdeoj8rmu'; path='/workspace' }) }
} | ConvertTo-Json -Depth 6
(Invoke-RestMethod 'https://api.runpod.io/v2/pods' -Method Post `
   -ContentType 'application/json' -Headers @{Authorization="Bearer $key"} -Body $body).id
```

**MCP `create-pod` 는 쓰지 말 것** — Network Volume 을 붙이지 못한다.

### 6. 검증 (약 7분 뒤)

로그에서 순서대로 확인:
`Qdrant 준비 완료` → `Loading weights took ~55s` → `vLLM 준비 완료` →
`DART 종목 코드 로드 완료` → `임베딩 모델 워밍업 완료 (~18s, device=cuda)` → `Uvicorn running`

스모크:
```powershell
$u='https://<POD_ID>-8080.proxy.runpod.net'
Invoke-RestMethod "$u/health"
```

### 7. 끝나면 삭제

```powershell
Invoke-RestMethod "https://api.runpod.io/v2/pods/<POD_ID>" -Method Delete -Headers @{Authorization="Bearer $key"}
```

### 알아둘 것

- 로컬에서 git 명령을 쓸 때 Cowork VM(`device_bash`)은 `.git/index.lock` 을 지우지
  못한다. **git 은 Windows PowerShell 에서 실행할 것.**
- 커밋 메시지에 한글이 있으면 BOM 없는 UTF-8 파일로 쓰고 `git commit -F` 를 쓴다.
  PowerShell 5.1 은 BOM 없는 UTF-8 **스크립트**는 CP949 로 읽으므로, 스크립트 파일은
  반대로 BOM 을 넣어야 한다.

## 8-1. (v1 기록) 앱이 로컬에 남아 있던 구성 — 지금은 §8-2 로 대체됨

**H100 파드에 올라간 것은 LLM 하나뿐이다.** 이 프로젝트의 코드나 데이터는
서버에 전혀 올라가지 않았다. 파드는 Hugging Face 에서 Qwen3.8-27B-FP8 을
직접 받아서 vLLM 으로 서빙만 한다.

```
[내 PC (Windows)]                              [RunPod H100 / AP-JP-1]
  FastAPI (app/)                                 vLLM
  Qdrant        localhost:6333                     └ Qwen3.8-27B-FP8
  chroma_storage 18MB (로컬 디스크)      HTTPS
  bge-m3 임베딩 (CPU)              ───────────────▶  :8000/v1
  퀴즈/RAG 로직
```

리허설에서 `compare_models.py` 도 **내 PC 에서** 돌았고, 파드로는 HTTP 요청만 갔다.

### 그래서 베타 당일에 남는 문제

**LLM 은 해결됐지만 앱 자체는 여전히 내 PC 에서 돈다.** 베타 사용자 20명이
쓰려면 내 PC 가 외부에서 접근 가능해야 하고, 그동안 계속 켜져 있어야 한다.

확인해야 할 것:

1. **외부 노출 경로** — 포트포워딩 또는 터널(ngrok/cloudflared).
   노출하는 순간 `AI_API_KEY` 를 반드시 채울 것 (config.py 주석에도 적혀 있음).
   비워두면 인증 없는 공개 API 가 된다.
2. **내 PC 가 병목이 될 수 있다.** bge-m3 임베딩이 CPU 에서 1회 50~200ms 이고
   중복 검사는 문항 수의 제곱으로 호출된다(`bank.py`). H100 은 놀고 내 PC 가
   막히는 그림이 나올 수 있다. **이번 리허설은 LLM 만 측정했지 이 경로는 재지 않았다.**
3. **Qdrant / chroma_storage 도 로컬**이다. PC 가 꺼지면 전부 멈춘다.

→ **이 문제는 §8-2 (v2) 로 해결됐다.** 앱까지 파드에 올려서 포트포워딩·터널·PC
   상시 가동이 모두 불필요해졌다. 아래 내용은 왜 v2 로 갔는지에 대한 기록이다.

### 코드를 고치면 무엇을 다시 해야 하나

| 무엇을 고쳤나 | 파드에 필요한 조치 |
|---|---|
| `app/` 아래 파이썬 코드, 프롬프트, RAG 데이터 | **없음.** 로컬 FastAPI 만 재시작 |
| `.env` / 환경변수 | **없음.** 로컬만 |
| vLLM 플래그, 모델, 포트, env | 템플릿 수정 → **파드 재생성** (파드에서는 수정 불가) |

파드는 모델만 들고 있으므로, 프로젝트를 아무리 고쳐도 파드는 그대로 둬도 된다.

## 8-2. 전체 스택 단일 파드 (v2 구성) — **검증 완료**

§8-1 에서 정리한 "앱은 여전히 내 PC" 문제를 없애기 위해 vLLM + Qdrant + FastAPI 를
한 컨테이너에 넣었다. 베타 당일에 파드 하나만 띄우면 공개 URL 이 나온다.

```
[H100 파드 / AP-JP-1]                     공개 URL
  FastAPI   :8080  (0.0.0.0)  ────────▶  https://<POD_ID>-8080.proxy.runpod.net
    ├── vLLM    :8000  (127.0.0.1)          ← NestJS 팀이 호출하는 주소
    └── Qdrant  :6333  (127.0.0.1)
  Network Volume /workspace
    ├── hf/              LLM 가중치 + bge-m3
    └── qdrant_storage/  벡터 데이터
```

포트포워딩·터널·PC 상시 가동이 전부 불필요해진다.
vLLM 과 Qdrant 는 127.0.0.1 바인딩이라 외부로 새지 않는다.

### 만든 것

| 파일 | 역할 |
|---|---|
| `Dockerfile` | `FROM vllm/vllm-openai@sha256:61fc8a89…` (리허설 검증본 고정) + Qdrant 바이너리 + 앱 |
| `docker/entrypoint.sh` | Qdrant → vLLM → FastAPI 순차 기동. 각 단계 health 폴링, 실패 시 컨테이너 종료 |
| `docker/qdrant-config.yaml` | 127.0.0.1 바인딩, storage 는 `/workspace/qdrant_storage` |
| `requirements-server.txt` | 앱 의존성 (torch/transformers/vllm 은 베이스 것을 쓰므로 제외) |
| `.dockerignore` | `.env`, `.venv`, `chroma_storage`, `compare_out` 등 제외 |

코드 변경은 3파일 6줄뿐이다. `EMBEDDING_DEVICE` 설정을 추가해 로컬은 `cpu`,
서버 이미지는 `cuda` 로 뜬다. 죽은 Chroma 의존성 2개를 제거했다.

### 빌드 · 푸시

```powershell
cd C:\Users\dhp13\Desktop\stockmate-ai
docker login -u dohun1214
docker build -t dohun1214/stockmate-ai:0.1.0 .
docker push dohun1214/stockmate-ai:0.1.0
```

**태그를 덮어쓰지 말고 버전을 올릴 것** (`0.1.1`, `0.1.2` …).
같은 태그를 덮어쓰면 발표 당일에 검증하지 않은 이미지가 내려올 수 있다.

2026-08-30 빌드 결과:
- 이미지 **29.9GB**, digest `sha256:063071520118a4f08e99318cc4ed10081a32711b4c099c0171a404b8c37782b6`
- 빌드 게이트 통과: `vllm 0.28.0` / `app deps ok` / `app.main import ok`
- 고정된 버전: torch 2.13.0+cu130, transformers 5.15.1, tokenizers 0.22.2, numpy 2.2.6, vllm 0.28.0
- 베이스 pull 15분, 빌드 3분, 푸시 12분 (회선 약 9MB/s)

### 왜 constraint 를 걸었나

`sentence-transformers` 를 설치하면 pip 가 베이스 이미지의 `transformers` 나
`torch` 를 갈아엎어 vLLM 을 망가뜨릴 수 있다. 그래서 빌드 시점에 베이스의
버전을 뽑아 constraint 로 못박았다. 갈아엎어야만 설치되는 의존성이 있으면
**빌드가 실패한다** — 시간당 $3.29 짜리 파드가 아니라 빌드에서 잡는 게 맞다.

### 템플릿 (v2)

| 필드 | 값 | 비고 |
|---|---|---|
| Name | `stockmate-fullstack` | **생성 완료: `c3bqwiolqi`** |
| Image | `dohun1214/stockmate-ai:0.1.0` | |
| **Container Disk** | **60 GB** | 이미지가 29.9GB 라 30GB 로는 안 된다 |
| Volume Mount Path | `/workspace` | |
| Ports | `8080/http` | **8000 은 열지 않는다** — vLLM 은 내부 전용 |
| Start Command | **비움** | 이미지 ENTRYPOINT 가 처리한다 |

env — 비밀값은 **템플릿 env 에 직접** 넣었다 (이미지가 public 이라 이미지에는 안 넣는다).
RunPod Secrets 참조(`{{ RUNPOD_SECRET_이름 }}`)로 바꾸면 한 단계 더 안전해진다:

```
HF_TOKEN            <값>
VLLM_API_KEY        sk-stockmate-7f3a91c4e2
AI_API_KEY          {{ RUNPOD_SECRET_AI_API_KEY }}
DART_API_KEY        {{ RUNPOD_SECRET_DART_API_KEY }}
NAVER_CLIENT_ID     {{ RUNPOD_SECRET_NAVER_CLIENT_ID }}
NAVER_CLIENT_SECRET {{ RUNPOD_SECRET_NAVER_CLIENT_SECRET }}
KIS_APP_KEY         {{ RUNPOD_SECRET_KIS_APP_KEY }}
KIS_APP_SECRET      {{ RUNPOD_SECRET_KIS_APP_SECRET }}
KIS_ACCOUNT         {{ RUNPOD_SECRET_KIS_ACCOUNT }}
```

나머지(`LLM_*`, `QDRANT_*`, `EMBEDDING_DEVICE=cuda`, `VLLM_*`)는 이미지 ENV 기본값에
들어 있어 템플릿에 다시 쓸 필요가 없다. `gpu-memory-utilization` 은 임베딩 자리를
만들려고 0.92 → **0.88** 로 낮췄다 (KV 캐시 여유가 49배였으므로 손해 없음).

### E2E 검증 결과 — 2026-08-30 통과

| 항목 | 결과 |
|---|---|
| 3프로세스 기동 (Qdrant→vLLM→FastAPI) | ✅ 총 **6분 33초** (이미지 pull 30초, vLLM 348s, 워밍업 18.3s) |
| 공개 HTTPS + 인증 | ✅ `/health` 공개, 그 외 401 차단 |
| 퀴즈 생성 OX / 객관식 | ✅ |
| Qdrant 중복 검사 + **GPU 임베딩** | ✅ `device=cuda`, 유사도 0.386 / 0.673 / 0.779 정상 판정 |
| RAG `/chat/evaluate` | ✅ 35.7s — 뉴스 10 + DART 100 + 재무제표 1 인제스트, **KIS 실시간 시세 연동** |
| 일반 질문 `/chat/evaluate` | ✅ 14.4s (mode=general) |
| **동시 20명 퀴즈 생성** | ✅ **20/20**, 전체 57.3s, p50 38.5s / **p95 41.5s** |
| count=4 장시간 요청 | ✅ 98.9s 정상 완료 |
| 가중치 볼륨 재사용 | ✅ 재다운로드 없음 |
| 총 비용 (하루 전체) | **$3.35** |

### 검증 중 발견해 고친 것

1. **첫 요청이 임베딩 로딩 18초를 뒤집어씀** → `app/main.py` lifespan 에 워밍업 추가.
2. **`객관식 문제 생성 실패: 응답에서 JSON 객체를 찾지 못함` 이 3번 중 1번 발생.**
   `reasoning_effort=medium` 의 사고 토큰이 출력 예산(2048)을 먹어 JSON 본문이 잘렸다.
   `config.py` 주석에 이미 적혀 있던 현상이 서버 환경에서 재현된 것.
   → 템플릿 env 에 **`LLM_MAX_TOKENS=4096`**. 같은 조합 **6/6 성공**으로 확인.
   (평균 52.3s, 최대 73s)
3. **재시도 실패 사유가 어디에도 안 남음.** 라우터가 `ValueError` 를 `HTTPException`
   으로 바꿔 던져 전역 핸들러를 타지 않고, 재시도 루프는 `except: continue` 로
   사유를 버렸다. → `generator.py` 두 루프에 시도별 WARNING + 소진 시 ERROR 추가.

### 틀렸던 가설 — 그리고 그 정정 (2026-09-05/08)

리허설 중반에 "RunPod 프록시가 80초에서 끊는다"고 판단했다가, `count=4` 요청이
**98.9초에 정상 완료**하는 것을 보고 "제한은 확인되지 않았다"로 기록했었다.

**그 기록이 틀렸다.** 98.9초는 100초를 넘겨서 통과한 게 아니라 **안 넘겨서**
통과한 것이다. 2026-09-05 파드에서 동시 20명 프롬프트 퀴즈가 100초를 넘기자
**20건 전부 HTTP 524** 로 떨어졌다.

> **RunPod HTTP 프록시(Cloudflare)는 응답 첫 바이트까지 100초를 기다린 뒤
> 524 로 끊는다.** 그때도 서버는 요청을 정상 처리하고 있다 — 클라이언트가 524 를
> 받은 뒤에도 `퀴즈 뱅크 저장` 이 계속 찍혔다. **생성이 아니라 전달이 실패한다.**

**0.1.5 부터 이 벽은 없앴다.** 응답이 30초를 넘기면 10초마다 공백을 흘려보내
연결을 살려 둔다(`app/core/keepalive.py`). 파드에서 140초 요청 정상 도착,
동시 20명에서 324초 요청까지 524 0건을 확인했다.
80초 부근 실패는 여전히 위 2번(재시도 소진)일 수 있으니 로그를 볼 것.

자세한 경위: `claude/stockmate-ai-improvements.md`

### 남은 개선 (선택)

- `VLLM_CACHE_ROOT=/workspace/vllm_cache` 로 컴파일 캐시를 볼륨에 남기면 기동 단축 가능. **미검증.**
- Qdrant 가 기동 시 경고: `FUSE filesystems may cause data corruption`.
  동작에는 문제없었고 데이터는 자기 복구되는 캐시다. 이상 시 `QDRANT_STORAGE` 를
  컨테이너 디스크로 돌릴 수 있으나, 그러면 파드 삭제 시 퀴즈 뱅크가 사라진다.
- 생성 지연이 count=1 에 평균 52초다. 문제 은행 사전 생성(남은 과제 3번)이
  가장 효과적인 개선이다.

### `/chat/stream` SSE 검증 — 2026-08-30 통과

NestJS 가 실제로 붙는 주 엔드포인트다. Cloudflare 프록시가 SSE 를 버퍼링하면
토큰이 실시간으로 안 흘러 스트리밍이 무의미해지므로 별도로 확인했다.

```
status 200 | content-type: text/event-stream; charset=utf-8
Transfer-Encoding: chunked | CF-RAY: ...-ICN   (인천 엣지)
meta 2,019ms  →  첫 token 3,245ms  →  done 14,841ms
923줄이 12.8초에 걸쳐 분산 도착 — 버퍼링 없음
token 이벤트 984개 중 643개에 내용, 조립 시 1,234자
```

**버퍼링 없음. 그대로 써도 된다.** 빈 `content` 가 섞이는 것은 LangChain
스트리밍의 정상 동작이므로 클라이언트는 빈 문자열을 그냥 무시하면 된다.

계측 시 주의: PowerShell 5.1 은 **BOM 없는 UTF-8 스크립트를 CP949 로 읽는다.**
한글 질문이 깨진 채 전송되어 모델이 엉뚱하게 답하는 것을 앱 버그로 오인했다.
테스트 스크립트는 BOM 을 넣어 저장할 것.

## 8-3. NestJS 백엔드 연동

```env
AI_BASE_URL=https://<POD_ID>-8080.proxy.runpod.net
AI_API_KEY=<.env 의 AI_API_KEY 와 동일한 값>
```

`<POD_ID>` 는 **파드를 만들 때마다 바뀐다. 하드코딩 금지.**

```ts
HttpModule.register({
  baseURL: process.env.AI_BASE_URL,
  headers: { 'X-API-Key': process.env.AI_API_KEY },
  timeout: 180_000,
})
```

### 타임아웃이 이 연동의 최대 함정

| 요청 | 실측 |
|---|---|
| `/quiz/generate` count=1 | 평균 52초, 최대 **73초** |
| `/quiz/generate` count=4 | **98.9초** |
| `/chat/evaluate` (RAG) | 35.7초 |
| `/chat/stream` 완료까지 | 14.8초 |
| 동시 20명 | p95 41.5초 |

axios 기본값은 무제한이지만 `HttpModule` 에 timeout 을 걸었거나 앞단에 nginx 가
있으면 **기본 60초**에서 잘린다. 서버는 답을 만들고 있는데 백엔드가 먼저 끊는다.
**180초 이상**으로 잡을 것.

### 그 밖에

- **500 을 자동 재시도하지 말 것.** `/quiz/generate` 의 500 은 이미 LLM 을 3번
  호출하고 실패한 결과다. 자동 재시도하면 GPU 를 80초 더 태우고, 동시 사용자가
  몰릴 때 눈덩이가 된다. 사용자에게 "다시 시도" 버튼을 주는 쪽이 맞다.
- **인증 예외 경로**: `/health`, `/docs`, `/redoc`, `/openapi.json` 은 키 없이 열린다
  (의도된 설계). 그 외는 401.
- **readiness**: entrypoint 가 vLLM health 를 기다린 뒤 FastAPI 를 띄우므로,
  `/health` 가 200 이면 전체 스택이 준비된 상태다.
- **계약서**: `docs/openapi.json`. 갱신은 `uv run python scripts/export_openapi.py`.
  이번 서버화 작업으로 API 표면은 바뀌지 않았다.

### 자격증명 위치

**모든 키는 `.env` 에 있다** (`.gitignore` 4번째 줄로 제외됨, 추적 안 됨).
RunPod 템플릿 `c3bqwiolqi` 의 env 에도 같은 값이 들어 있다.

```
AI_API_KEY  DART_API_KEY  NAVER_CLIENT_ID/SECRET  KIS_APP_KEY/SECRET  KIS_ACCOUNT
HF_TOKEN    RUNPOD_API_KEY
RUNPOD_NETWORK_VOLUME_ID=htdeoj8rmu   RUNPOD_TEMPLATE_FULLSTACK=c3bqwiolqi
RUNPOD_TEMPLATE_VLLM_ONLY=beonudgm8k  RUNPOD_DATACENTER=AP-JP-1
```

졸업작품이 끝나면 전부 재발급할 것. 특히 KIS 는 실계좌 연동 키다.

## 9. 함정 기록

- **Network Volume 은 리전 고정.** 먼저 H100 재고 리전을 확인하고 만든다.
  Secure Cloud 전용.
- **`HF_HOME=/workspace/hf` 를 빠뜨리면** 가중치가 컨테이너 디스크에 떨어져
  파드를 지울 때 같이 사라진다. 이 런북 전체가 이 한 줄을 위한 것이다.
- **Expose HTTP Ports 에 8000 을 안 적으면** 프록시가 라우팅을 안 한다.
  포트를 열었는데 502 면 이걸 먼저 본다.
- **RunPod HTTP 프록시는 Cloudflare 100초 타임아웃**이 걸린다 (2026-09-05 실측 확인).
  **0.1.5 부터 하트비트 응답으로 무력화했으므로 지금은 신경 쓸 필요가 없다.**
  단, 524 가 다시 보이면 `RESPONSE_KEEPALIVE_ENABLED` 가 꺼졌는지부터 본다.
  (TCP 직결 포트는 HTTPS 가 아니게 되므로 공개 URL 로는 부적절 — 최후 수단)
- **프록시가 응답을 gzip 으로 압축한다.** 프록시 뒤를 계측할 때 raw 바이트를 읽으면
  압축된 것을 보게 된다. 로컬(127.0.0.1)에서는 압축이 없어 이 함정이 안 보인다.
  프록시 경유 측정에는 `scripts/check_keepalive.py` / `check_chat_stream.py` 를 쓸 것.
- **uvicorn access log 를 끄지 말 것.** 한때 entrypoint 가 `--no-access-log` 로
  띄우고 있었는데, "어떤 요청이 몇 초에 어떤 상태 코드로 끝났는지"를 볼 수 없어
  연동 장애 추적이 하루씩 걸렸다. 켜 두면 10분이면 잡힌다.
- **새 모듈에서 `logging.getLogger(__name__)` 을 쓰지 말 것.** 이 프로젝트는
  로거마다 핸들러를 직접 붙이는 구조(`app/core/logger.py`)라 맨 getLogger 는
  핸들러도 레벨도 없어 **INFO 로그가 통째로 사라진다.** 반드시 `setup_logger`.
- **`reasoning_effort` 기본값이 `xhigh`.** 서버 기동 플래그로 못박는 게 가장 안전.
- **파드 Stop 은 GPU 반납이다.** 재고가 빠지면 Start 가 안 된다.
  긴 공백이라면 Terminate 하고 나중에 재배포하는 편이 낫다 (가중치는 남으므로).
- **`vllm/vllm-openai` 의 ENTRYPOINT 가 버전마다 다르다.** 시작 명령이 두 번
  들어가 `unrecognized arguments: vllm serve` 로 죽는다. §3-1 의 B 형태로 해결.
- **PCIe 와 SXM 은 서빙 설정이 같다.** 재고 있는 쪽을 잡으면 된다.
  SXM 이 대역폭 1.6배라 오히려 빠르고, 시급만 $0.40 비싸다.
- **RunPod 는 Start Command 의 큰따옴표를 먹는다.** args 문자열을 shlex 로
  쪼개기 때문. JSON 값을 넘길 때는 `'{"key":"value"}'` 처럼 작은따옴표로 감쌀 것.
- **파드 생성 후에는 Start Command 를 못 고친다.** MCP `update-pod` 에도,
  RunPod UI 에도 args 수정이 없다. 시작 명령이 틀리면 **파드를 지우고 다시 만들어야**
  한다. 그래서 템플릿에 미리 넣어두고 검증된 것만 쓰는 게 중요하다.
- **크레딧 소진 시 예고 없이 종료.** 발표 전날 잔액 확인.
