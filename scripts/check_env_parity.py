"""로컬 설정(.env)과 서버 이미지(Dockerfile ENV)가 어긋났는지 대조한다.

## 왜 필요한가

앱 코드는 이미지에 구워지지만 **설정은 세 곳에 흩어져 있다.**

    app/core/config.py   기본값 (코드)
    .env                 로컬에서 실제로 쓰는 값
    Dockerfile ENV       서버(파드)에서 실제로 쓰는 값
    .env.example         새로 세팅하는 사람이 따라 쓰는 값

런북 §8-0 이 "설정을 추가하면 네 곳을 같이 고쳐라"라고 정해 뒀는데, 사람이
지키는 규칙이라 빠뜨리면 **로컬에서는 되고 서버에서만 다르게 도는** 상태가
된다. 그런 차이는 배포 후에야 드러나고 재빌드 15분이 날아간다.

실제로 이 스크립트를 처음 돌렸을 때 `.env.example` 이 Ollama 시절 값으로
남아 있고 `LLM_MODEL` 이 두 번 정의돼 있는 것을 찾았다.

## 무엇을 보는가

- Dockerfile ENV 와 `.env` 의 **튜닝값이 같은지** (다르면 서버/로컬 동작이 다름)
- `.env.example` 에 빠진 항목이 있는지
- `config.py` 에만 있고 어디에도 안 적힌 설정이 있는지

로컬과 서버가 **달라야 정상인 항목**(LLM 백엔드, 임베딩 장치, 포트 등)은
INTENTIONAL 에 적어 두고 차이를 허용한다. 여기 없는 항목이 다르면 실수다.

    uv run python scripts/check_env_parity.py
    -> 어긋난 게 있으면 exit 1
"""
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 로컬과 서버가 달라야 정상인 항목. 여기 없는데 다르면 동기화 실수다.
INTENTIONAL = {
    "LLM_BACKEND", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY",
    "EMBEDDING_DEVICE", "QDRANT_HOST", "QDRANT_PORT", "QDRANT_STORAGE",
    "APP_PORT", "HF_HOME",
}

# 값 자체를 비교하면 안 되는 것들 (자격증명)
SECRETS = {
    "AI_API_KEY", "LLM_API_KEY", "DART_API_KEY", "NAVER_CLIENT_ID",
    "NAVER_CLIENT_SECRET", "KIS_APP_KEY", "KIS_APP_SECRET", "KIS_ACCOUNT",
    "HF_TOKEN", "RUNPOD_API_KEY", "VLLM_API_KEY",
}

# 앱 설정으로 볼 접두사 (vLLM 기동 인자 등 서버 전용은 제외)
PREFIXES = ("LLM_", "QUIZ_", "PROMPT_", "ANALYZE_", "CHAT_", "RESPONSE_", "EMBEDDING_")


def parse_env(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def parse_dockerfile_env(path: Path) -> dict:
    """ENV 블록(역슬래시로 이어진 여러 줄)을 편다."""
    src = io.open(path, encoding="utf-8").read()
    out = {}
    for match in re.finditer(r"^ENV\s+(.*?)(?=\n(?!\s)|\Z)", src, re.M | re.S):
        block = match.group(1).replace("\\\n", "\n")
        for part in block.split("\n"):
            part = part.strip().rstrip("\\").strip()
            if "=" in part and not part.startswith("#"):
                key, value = part.split("=", 1)
                out[key.strip()] = value.strip()
    return out


def config_fields(path: Path) -> set:
    src = io.open(path, encoding="utf-8").read()
    return set(re.findall(r"^\s{4}([A-Z][A-Z0-9_]*)\s*:", src, re.M))


def duplicated_keys(path: Path) -> list:
    """같은 키를 두 번 정의하면 뒤엣것이 이긴다 - 조용한 사고의 원인."""
    seen, dup = set(), []
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key in seen:
            dup.append(key)
        seen.add(key)
    return dup


def main() -> int:
    docker = parse_dockerfile_env(ROOT / "Dockerfile")
    env = parse_env(ROOT / ".env")
    example = parse_env(ROOT / ".env.example")
    fields = config_fields(ROOT / "app" / "core" / "config.py")

    problems: list[str] = []
    keys = sorted(k for k in docker if k.startswith(PREFIXES))

    print(f"{'변수':<34}{'Dockerfile':<14}{'.env':<14}{'example':<14} 판정")
    print("-" * 92)
    for key in keys:
        d = docker.get(key, "—")
        e = env.get(key, "—")
        x = example.get(key, "—")
        show_d, show_e, show_x = (
            ("[secret]", "[secret]", "[secret]") if key in SECRETS else (d[:13], e[:13], x[:13])
        )

        if key in INTENTIONAL:
            verdict = "의도된 차이" if d != e else "동일"
        elif key in SECRETS:
            verdict = "값 비교 안 함"
        elif e == "—":
            verdict = "❌ .env 에 없음"
            problems.append(f"{key}: .env 에 없다")
        elif d.lower() != e.lower():
            verdict = "❌ 불일치"
            problems.append(f"{key}: Dockerfile={d} / .env={e}")
        elif x == "—":
            verdict = "⚠️ example 누락"
            problems.append(f"{key}: .env.example 에 없다")
        elif x.lower() != d.lower():
            verdict = "⚠️ example 값 다름"
            problems.append(f"{key}: .env.example={x} / Dockerfile={d}")
        else:
            verdict = "✅"
        print(f"{key:<34}{show_d:<14}{show_e:<14}{show_x:<14} {verdict}")

    # 중복 정의 (뒤엣것이 이겨서 조용히 어긋난다)
    for name in (".env", ".env.example"):
        dup = duplicated_keys(ROOT / name)
        if dup:
            problems.append(f"{name}: 같은 키가 두 번 정의됨 - {', '.join(dup)}")

    # config.py 에만 있는 설정 (템플릿 env 로 조정은 되지만 문서화가 안 된 것)
    undocumented = sorted(
        f for f in fields
        if f.startswith(PREFIXES) and f not in docker and f not in SECRETS
    )
    if undocumented:
        print("\n[참고] config.py 에만 있고 Dockerfile ENV 에 없는 설정:")
        print("  " + ", ".join(undocumented))
        print("  pydantic-settings 는 선언 여부와 무관하게 환경변수를 읽으므로")
        print("  파드 템플릿 env 로 조정은 된다. 다만 기본값이 코드에만 있어")
        print("  '서버에서 무슨 값으로 도는지' 를 Dockerfile 만 봐서는 알 수 없다.")

    print("\n" + "=" * 60)
    if problems:
        print(f"어긋난 곳 {len(problems)}건:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("로컬(.env) / 서버(Dockerfile ENV) / 예시(.env.example) 동기화 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
