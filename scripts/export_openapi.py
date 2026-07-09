"""FastAPI 앱의 OpenAPI 스키마를 정적 JSON 파일로 내보낸다.

서버를 띄우지 않고도 코드에서 바로 스키마를 생성한다 (app.openapi()는
lifespan 훅을 실행하지 않으므로 DART/Qdrant/Ollama 연결 없이도 동작함).
NestJS 팀이 최신 계약(quiz_context.category, /quiz/generate의 count/questions 등)을
서버 재시작 없이 파일로 확인할 수 있게 하기 위한 용도.

사용법:
    uv run python scripts/export_openapi.py
    -> docs/openapi.json 생성/갱신
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "openapi.json"


def main():
    schema = app.openapi()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"OpenAPI 스키마 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
