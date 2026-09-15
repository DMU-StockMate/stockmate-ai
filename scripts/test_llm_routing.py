"""OpenRouter 프로바이더 고정이 의도한 요청에만 실리는지 확인한다.

네트워크 없이 요청 본문만 검사한다.

    uv run python scripts/test_llm_routing.py

배경: OpenRouter 는 모델 이름 하나 뒤에 프로바이더를 여럿 두고 양자화가 제각각이다
(2026-09-15 실측: qwen/qwen3.8-27b 에 16곳, bf16 / fp8 / fp4 / unknown 혼재).
고정하지 않으면 같은 요청 10회가 6개 프로바이더로 흩어진다(그중 fp4 3회).
다만 이 필드는 OpenRouter 전용이므로 파드(vLLM)로 나가면 안 된다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.llm import build_llm

OPENROUTER = "https://openrouter.ai/api/v1"
VLLM = "http://127.0.0.1:8000/v1"


def extra_body(base_url: str, quantizations: str, fallbacks: bool = False) -> dict:
    saved = (settings.LLM_BACKEND, settings.LLM_BASE_URL,
             settings.LLM_OPENROUTER_QUANTIZATIONS,
             settings.LLM_OPENROUTER_ALLOW_FALLBACKS)
    settings.LLM_BACKEND = "openai"
    settings.LLM_BASE_URL = base_url
    settings.LLM_OPENROUTER_QUANTIZATIONS = quantizations
    settings.LLM_OPENROUTER_ALLOW_FALLBACKS = fallbacks
    try:
        llm = build_llm(temperature=0.3)
        return dict(getattr(llm, "extra_body", None) or {})
    finally:
        (settings.LLM_BACKEND, settings.LLM_BASE_URL,
         settings.LLM_OPENROUTER_QUANTIZATIONS,
         settings.LLM_OPENROUTER_ALLOW_FALLBACKS) = saved


CASES = [
    ("OpenRouter 면 fp8 로 고정한다", OPENROUTER, "fp8", False,
     {"quantizations": ["fp8"], "allow_fallbacks": False}),
    ("여러 양자화를 쉼표로 받는다", OPENROUTER, "fp8, bf16", True,
     {"quantizations": ["fp8", "bf16"], "allow_fallbacks": True}),
    ("파드(vLLM)로는 보내지 않는다", VLLM, "fp8", False, None),
    ("빈 값이면 고정하지 않는다", OPENROUTER, "", False, None),
    ("공백만 있어도 고정하지 않는다", OPENROUTER, " , ", False, None),
]


def main() -> int:
    failed = 0
    for name, base_url, quants, fallbacks, expected in CASES:
        got = extra_body(base_url, quants, fallbacks).get("provider")
        if got == expected:
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name}\n        got={got!r}\n        expected={expected!r}")

    # 고정과 무관하게 reasoning_effort 는 계속 실려야 한다
    body = extra_body(OPENROUTER, "fp8")
    if body.get("reasoning_effort"):
        print("  PASS  reasoning_effort 가 함께 실린다")
    else:
        failed += 1
        print(f"  FAIL  reasoning_effort 가 빠졌다: {body!r}")

    print()
    print(f"실패 {failed}건" if failed else "전부 통과")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
