"""Per-request record of whether each router deployment attempt failed with a
provider-proven "not sent" error (drama contract §9).

Opt-in: nothing is recorded unless a caller opened a tracking scope with
``start()``, so behaviour for every other caller is unchanged. The scope holds a
mutable list, so attempts made in child tasks (retries, fallbacks) are seen.
"""

from contextvars import ContextVar, Token
from typing import List, Optional

MARKER_ATTR = "drama_submission"
NOT_SENT = "not_sent"

_ATTEMPTS: ContextVar[Optional[List[bool]]] = ContextVar("drama_router_attempts", default=None)


def marked_not_sent(error: Optional[BaseException], depth: int = 0) -> bool:
    """True when ``error`` (or its cause chain) carries the not-sent marker."""
    while error is not None and depth < 16:
        if getattr(error, MARKER_ATTR, None) == NOT_SENT:
            return True
        error = error.__cause__ or error.__context__
        depth += 1
    return False


def start() -> Token:
    return _ATTEMPTS.set([])


def stop(token: Token) -> None:
    _ATTEMPTS.reset(token)


def record(error: BaseException) -> None:
    attempts = _ATTEMPTS.get()
    if attempts is not None:
        attempts.append(marked_not_sent(error))


def snapshot() -> Optional[List[bool]]:
    attempts = _ATTEMPTS.get()
    return None if attempts is None else list(attempts)
