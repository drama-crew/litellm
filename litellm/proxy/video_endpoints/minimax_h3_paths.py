from __future__ import annotations

from typing import Literal

IR_PREFIX = "/video/minimax-h3"
DIRECT_PREFIX = IR_PREFIX + "/direct"
PREFIXES = (IR_PREFIX, DIRECT_PREFIX)
# Test worker pool facade: the same API as the production prefixes, but every generation is enqueued on the
# isolated ``*_test`` worker streams (see ``llms/libtv/video_generate.py``) and is admitted against its own cap.
TEST_IR_PREFIX = "/video/minimax-h3-test"
TEST_DIRECT_PREFIX = TEST_IR_PREFIX + "/direct"
TEST_PREFIXES = (TEST_IR_PREFIX, TEST_DIRECT_PREFIX)
ALL_PREFIXES = PREFIXES + TEST_PREFIXES
# Official MiniMax V2 model name; the lower-case spelling is the legacy alias still accepted on input.
PUBLIC_MODEL = "MiniMax-H3"
LEGACY_PUBLIC_MODEL = "minimax-h3"
UNAVAILABLE_PUBLIC_MODEL = "MiniMax-H3-Max"
INTERNAL_MODEL = "causyn-1.1"
Namespace = Literal["minimax-h3", "minimax-h3-direct", "minimax-h3-test", "minimax-h3-direct-test"]
WORKER_POOL_TEST = "test"


def namespace(path: str) -> Namespace | None:
    if path.startswith(DIRECT_PREFIX + "/"):
        return "minimax-h3-direct"
    if path.startswith(IR_PREFIX + "/"):
        return "minimax-h3"
    if path.startswith(TEST_DIRECT_PREFIX + "/"):
        return "minimax-h3-direct-test"
    if path.startswith(TEST_IR_PREFIX + "/"):
        return "minimax-h3-test"
    return None


def is_test_path(path: str) -> bool:
    return path == TEST_IR_PREFIX or path.startswith(TEST_IR_PREFIX + "/")


_NAMESPACE_PREFIX: dict[str, str] = {
    "minimax-h3": IR_PREFIX,
    "minimax-h3-direct": DIRECT_PREFIX,
    "minimax-h3-test": TEST_IR_PREFIX,
    "minimax-h3-direct-test": TEST_DIRECT_PREFIX,
}


def continuation_path(ns: str | None, route: str) -> str | None:
    """The public route an admitted H3 task entered through, or None when it is not an H3 namespace.

    The moderation continuation re-authenticates the owner's stored key; that check must be made against the
    route the task was admitted through (the key's ``allowed_routes`` grant namespaced routes, never the generic
    ``/v1/videos``). ``None`` keeps the caller's generic default (non-H3 tasks, or an older platform that does
    not report the namespace).
    """
    prefix = _NAMESPACE_PREFIX.get(ns or "")
    if prefix is None:
        return None
    if route == "avideo_generation":
        return prefix + "/v2/video_generation"
    if route == "context_ir" and prefix in (IR_PREFIX, TEST_IR_PREFIX):
        return prefix + "/v2/h3_context_ir"
    return None


FILES_SEGMENT = "/v1/causyn_files"


def is_files_path(path: str) -> bool:
    """Direct-upload file routes exist on the production IR and Direct prefixes only (never the test pool)."""
    return any(path == prefix + FILES_SEGMENT or path.startswith(prefix + FILES_SEGMENT + "/") for prefix in PREFIXES)
