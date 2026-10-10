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
PUBLIC_MODEL = "minimax-h3"
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
