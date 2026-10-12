"""Trusted channel for the causyn test-worker-pool flag.

``worker_pool`` must never be readable from request parameters: everything in a request body (top level,
``extra_body``, metadata) merges into the provider's ``optional_params``, so a flag read from there is
public-controlled. Instead the proxy sets this context variable only after its trust check passed (a test-prefix
handler, or a moderation continuation of one), and the provider handler reads it from here and nowhere else.
"""

from __future__ import annotations

import math
import os
from contextvars import ContextVar, Token

TEST_POOL = "test"
TEST_POOL_DIRECT_DEADLINE_ENV = "CAUSYN_TEST_POOL_DIRECT_DEADLINE_SECONDS"
TEST_POOL_DIRECT_DEADLINE_DEFAULT = 900.0
_TRUSTED_POOL: ContextVar[str | None] = ContextVar("causyn_trusted_worker_pool", default=None)


def set_trusted_worker_pool(pool: str | None) -> Token[str | None]:
    return _TRUSTED_POOL.set(pool)


def reset_trusted_worker_pool(token: Token[str | None]) -> None:
    _TRUSTED_POOL.reset(token)


def trusted_worker_pool() -> str | None:
    return _TRUSTED_POOL.get()


def test_pool_direct_deadline_seconds() -> float:
    """Direct-mode execution budget for the test pool (slower lanes); invalid or <= 0 falls back to the default."""
    try:
        value = float(os.getenv(TEST_POOL_DIRECT_DEADLINE_ENV, ""))
    except ValueError:
        return TEST_POOL_DIRECT_DEADLINE_DEFAULT
    return value if math.isfinite(value) and value > 0 else TEST_POOL_DIRECT_DEADLINE_DEFAULT
