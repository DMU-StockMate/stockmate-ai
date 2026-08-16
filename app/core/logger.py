import logging
import sys

_stream_prepared = False


def _prepare_stream() -> None:
    """콘솔 출력을 UTF-8 로 맞춘다 (최초 1회).

    윈도우 기본 콘솔 인코딩은 cp949 라 한자·이모지가 섞이면 인코딩에 실패한다.
    실측 사고: LLM 이 만든 문제에 한자 '賣' 가 들어갔고, 그 문자가 담긴
    예외 메시지를 출력하려다 UnicodeEncodeError 로 배치 스크립트가 통째로 죽었다.

    이 서비스는 LLM 출력을 그대로 로그에 싣기 때문에 어떤 문자가 올지 알 수 없다.
    errors="replace" 까지 걸어 인코딩 실패로는 절대 죽지 않게 한다.
    """
    global _stream_prepared
    if _stream_prepared:
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # 재구성이 불가능한 스트림(리다이렉트 등)은 그대로 둔다
            pass
    _stream_prepared = True


def setup_logger(name: str) -> logging.Logger:
    _prepare_stream()
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)-5s %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
