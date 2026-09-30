import asyncio

import pytest
import pytest_asyncio

from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def drain_logging():
    """Leave the process-wide logging worker unbound from this test's event loop."""
    yield
    await asyncio.sleep(0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5)
    await GLOBAL_LOGGING_WORKER.stop()


@pytest.fixture(autouse=True)
def public_media_dns(monkeypatch):
    """Submit-time media admission resolves hostnames; unit tests must never touch real DNS."""
    from litellm.llms.causyn import public_media_policy

    async def resolve(host: str, port: int) -> list[str]:
        return ["93.184.216.34"]

    monkeypatch.setattr(public_media_policy, "resolve_host", resolve)
