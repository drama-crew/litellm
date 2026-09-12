from typing import Protocol


class RedisCommands(Protocol):
    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> object: ...


def keys(counter_key: str, namespace: str) -> tuple[str, str]:
    return namespace + counter_key, namespace + "moderation:counter:" + counter_key
