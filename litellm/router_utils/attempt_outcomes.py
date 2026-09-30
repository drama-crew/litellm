"""Request-scoped ledger of provider attempt outcomes (drama contract §9).

The only source of truth for "provably not sent". Each provider attempt adds
exactly one entry: ``not_sent`` when the LibTV provider boundary marked the
failure at the point it was raised, ``unknown`` for everything else (timeouts,
5xx, success-then-failure, routing errors, non-LibTV deployments). Exception
chains (``__cause__`` / ``__context__``) are never consulted.

Opt-in: nothing is recorded unless a caller opened a ledger with ``start()``.
The ledger is a mutable list, so attempts made in child tasks are seen.
"""

from contextvars import ContextVar, Token
from typing import List, Optional

NOT_SENT = "not_sent"
SENT = "sent"  # the provider returned a result: success is never evidence of not-sent
UNKNOWN = "unknown"
VERDICT_ATTR = "drama_all_attempts_not_sent"

_LEDGER: ContextVar[Optional[List[str]]] = ContextVar("drama_attempt_ledger", default=None)


def start() -> Token:
    return _LEDGER.set([])


def stop(token: Token) -> None:
    _LEDGER.reset(token)


def note(entry: str) -> None:
    """Provider boundary: record this attempt's outcome."""
    ledger = _LEDGER.get()
    if ledger is not None:
        ledger.append(entry)


def begin_attempt() -> Optional[int]:
    ledger = _LEDGER.get()
    return None if ledger is None else len(ledger)


def succeed_attempt(mark: Optional[int]) -> None:
    """Router: an attempt returned a provider result. Leave exactly one ``sent`` entry."""
    ledger = _LEDGER.get()
    if ledger is None or mark is None:
        return
    del ledger[mark:]
    ledger.append(SENT)


def fail_attempt(mark: Optional[int]) -> None:
    """Router: an attempt raised. Leave exactly one entry for it.

    ``sent`` when the provider had already returned a result inside the attempt,
    ``not_sent`` only when the boundary recorded exactly one ``not_sent`` entry,
    otherwise ``unknown`` (no boundary passed, several entries).
    """
    ledger = _LEDGER.get()
    if ledger is None or mark is None:
        return
    entries = ledger[mark:]
    del ledger[mark:]
    if SENT in entries:
        ledger.append(SENT)
    else:
        ledger.append(NOT_SENT if entries == [NOT_SENT] else UNKNOWN)


def provider_succeeded() -> bool:
    ledger = _LEDGER.get()
    return bool(ledger) and SENT in ledger


def all_not_sent() -> bool:
    ledger = _LEDGER.get()
    return bool(ledger) and all(entry == NOT_SENT for entry in ledger)


def mark_verdict(error: BaseException, verdict: bool) -> None:
    try:
        setattr(error, VERDICT_ATTR, verdict)
    except Exception:  # noqa: BLE001  # immutable exception: the verdict stays fail-closed
        pass


def verdict_of(error: BaseException) -> bool:
    """True only for an explicit verdict set by the metered entry, or a live all-not-sent ledger."""
    verdict = getattr(error, VERDICT_ATTR, None)
    if isinstance(verdict, bool):
        return verdict
    return all_not_sent() and not provider_succeeded()
