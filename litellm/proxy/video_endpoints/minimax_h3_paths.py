from __future__ import annotations

from typing import Literal

IR_PREFIX = "/video/minimax-h3"
DIRECT_PREFIX = IR_PREFIX + "/direct"
PREFIXES = (IR_PREFIX, DIRECT_PREFIX)
PUBLIC_MODEL = "minimax-h3"
INTERNAL_MODEL = "causyn-1.1"
Namespace = Literal["minimax-h3", "minimax-h3-direct"]


def namespace(path: str) -> Namespace | None:
    if path.startswith(DIRECT_PREFIX + "/"):
        return "minimax-h3-direct"
    if path.startswith(IR_PREFIX + "/"):
        return "minimax-h3"
    return None
