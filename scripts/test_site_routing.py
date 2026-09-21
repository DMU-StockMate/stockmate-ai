"""사이트 질문 판별과 채팅 갈래 선택을 확인한다. 네트워크·LLM 없이 돈다.

    uv run python -X utf8 scripts/test_site_routing.py

배경: 사이트 자체에 대한 질문("모의투자 처음에 돈 얼마 줘?")이 일반 갈래로 가서 모델이
다른 서비스 이야기를 지어냈다. 신호어로 사이트 갈래를 고르는데, 신호어가 일반 금융 용어와
겹치면(신용등급, ISA 가입, 계정과목) 개념 질문까지 끌려온다. 그 경계를 여기서 고정한다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.routers.chat as chat
from app.schemas.chat import Message
from app.services.rag.chain import SITE_PROMPT
from app.services.site.guide import is_site_question, load_site_guide

SITE = [
    "모의투자 처음에 돈 얼마 줘?", "시드머니 얼마야?", "퀴즈 맞히면 뭐 받아?",
    "같은 문제를 다시 맞히면 보상을 또 받아?", "랭킹은 어떻게 매겨?", "커뮤니티에 글 어떻게 써?",
    "관심종목 어떻게 추가해?", "내 등급은 어떻게 정해져?", "등급은 어떻게 올라가?",
    "투자성향 다시 검사하고 싶어", "밤 10시에도 모의투자 주문돼?", "비밀번호를 잊어버렸어",
    "회원탈퇴 어떻게 해?", "AI로 나만의 문제 만들 수 있어?", "오답 복습 기능 있어?",
    "스톡메이트는 어떤 사이트야?", "StockMate 뭐 하는 곳이야?", "이 사이트에서 뭐 할 수 있어?",
    "연속 학습 보너스 받으려면?", "주간 상승 랭킹은 뭐야?", "개념정리 메뉴는 어디 있어?",
    "찜한 문제는 어디서 봐?", "알림은 언제 와?", "마이페이지에서 뭐 바꿀 수 있어?",
    "닉네임 바꾸고 싶어", "로그인이 안 돼", "게시글 삭제하면 복구돼?", "댓글 수정 가능해?",
    "좋아요를 누르면 알림 가?", "가상 자금 초기화할 수 있어?", "삼성전자 모의투자로 어떻게 사?",
    "회원가입 어떻게 해?", "계정 삭제하고 싶어", "AI 설명 기록은 어디서 봐?",
]
NOT_SITE = [
    "PER이 뭐야?", "삼성전자 최근 공시 알려줘", "신용등급이 뭐야?", "ISA 계좌 가입 방법 알려줘",
    "계정과목이 뭐야?", "PER이 낮으면 좋아요?", "배당주 투자할 때 뭘 봐야 해?",
    "금리가 오르면 주가는 어떻게 돼?", "SK하이닉스 실적 어때?", "ETF랑 펀드 차이가 뭐야?",
    "손절은 언제 해야 해?", "코스피랑 코스닥 차이", "신용등급 강등되면 주가 떨어져?",
]


def fake_extract(question: str) -> list[str]:
    """퍼지 추출기가 사이트 용어를 종목으로 오인하는 상황을 흉내 낸다."""
    found = []
    if "삼성전자" in question:
        found.append("삼성전자")
    if "SK하이닉스" in question:
        found.append("SK하이닉스")
    if "커뮤니티" in question:
        found.append("엔써커뮤니티")  # 2026-09-22 실측 오인
    return found


def main() -> int:
    failed = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failed
        if ok:
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name} {detail}")

    for q in SITE:
        check(f"사이트: {q}", is_site_question(q))
    for q in NOT_SITE:
        check(f"사이트 아님: {q}", not is_site_question(q))

    chat.extract_tickers = fake_extract
    chat.extract_tickers_from_history = lambda history: []
    u = lambda text: Message(role="user", content=text)
    a = lambda text: Message(role="assistant", content=text)
    routes = [
        ("사이트 신호어가 종목 오인보다 먼저", "커뮤니티에 글 어떻게 써?", [], ("site", [])),
        ("종목 + 모의투자 사용법은 사이트", "삼성전자 모의투자로 어떻게 사?", [], ("site", [])),
        ("종목 질문은 그대로 RAG", "삼성전자 최근 공시 알려줘", [], ("rag", ["삼성전자"])),
        ("개념 질문은 그대로 일반", "PER이 뭐야?", [], ("general", [])),
        ("신호어 없는 후속 질문은 직전 사이트 질문을 따른다", "그럼 그건 어디서 봐?",
         [u("찜한 문제는 어떻게 모아?"), a("하트 버튼...")], ("site", [])),
        ("직전이 일반 질문이면 후속도 일반", "좀 더 쉽게 설명해줘",
         [u("PER이 뭐야?"), a("PER은...")], ("general", [])),
        ("후속 질문에 종목이 있으면 RAG", "SK하이닉스는?",
         [u("랭킹은 어떻게 매겨?"), a("총 자산...")], ("rag", ["SK하이닉스"])),
    ]
    for name, q, history, expected in routes:
        got = chat._route(q, history)
        check(name, got == expected, f"got={got!r} expected={expected!r}")

    guide = load_site_guide()
    for fact in ("3,000만 원", "오전 9시 ~ 오후 3시 30분", "수수료와 세금은 없다", "처음 맞혔을 때"):
        check(f"안내서에 '{fact}'", fact in guide)
    msgs = SITE_PROMPT.format_messages(
        history=[], question="q", today="2026년 09월 22일",
        level_guide="친절하게", site_guide=guide,
    )
    check("사이트 프롬프트에 안내서가 들어간다", "3,000만 원" in msgs[0].content)

    print(f"\n{'OK' if not failed else f'{failed} FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
