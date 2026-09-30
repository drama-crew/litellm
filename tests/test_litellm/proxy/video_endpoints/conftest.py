import asyncio

import pytest_asyncio

from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER


@pytest_asyncio.fixture(autouse=True, loop_scope="function")
async def drain_logging():
    """Leave the process-wide logging worker unbound from this test's event loop."""
    yield
    await asyncio.sleep(0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5)
    await GLOBAL_LOGGING_WORKER.stop()
