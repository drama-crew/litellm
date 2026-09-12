import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from litellm import Router
from litellm.proxy.video_endpoints.moderation_metering import BillingBinding, PhaseEvent
from litellm.proxy.video_endpoints import moderation_metering_runtime as runtime
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider
from litellm.utils import client
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER


@pytest_asyncio.fixture(autouse=True, loop_scope='function')
async def drain_logging():
    yield
    await asyncio.sleep(0)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5)
    await GLOBAL_LOGGING_WORKER.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize('amount', [None, 0, 20])
async def test_wrapper_router_retains_native_event_without_retry_on_store_failure(amount):
    authority = AsyncMock()
    authority.persist.side_effect = ConnectionError('synthetic SQL unavailable')
    binding = BillingBinding(intent_id='intent', request_digest='digest', fingerprint='key',
                             user_id='user', team_id='team', model='synthetic', expected_phases=('submit', 'completion'))
    accepted = VideoObject(id=encode_video_id_with_provider('native', 'openai', 'deployment'),
                           object='video', status='queued')
    accepted._hidden_params['response_cost'] = amount
    provider = AsyncMock(return_value=accepted)

    @client
    async def avideo_generation(**kwargs):
        return await provider(**kwargs)

    router = Router(model_list=[
        {'model_name': 'synthetic', 'litellm_params': {'model': 'openai/synthetic'}, 'model_info': {'id': 'deployment'}},
        {'model_name': 'fallback', 'litellm_params': {'model': 'openai/synthetic'}, 'model_info': {'id': 'fallback'}},
    ], num_retries=2, fallbacks=[{'synthetic': ['fallback']}])
    token = runtime.CONTEXT.set(runtime.Scope(binding, 'submit', authority))
    try:
        result = await router._ageneric_api_call_with_fallbacks(
            model='synthetic', original_function=avideo_generation, prompt='synthetic', caching=False,
        )
    finally:
        runtime.CONTEXT.reset(token)
    assert result is accepted
    assert provider.await_count == 1
    assert authority.persist.await_count == 1
    event = PhaseEvent.model_validate(runtime.private_event(result))
    assert event.native_id == accepted.id
    assert event.amount == (None if amount is None else Decimal(amount))
    assert event.finalized is (amount is not None)
    assert accepted._hidden_params['_moderation_metering_pending'] is True


@pytest.mark.asyncio
async def test_request_metadata_cannot_select_metering_authority():
    provider = AsyncMock(return_value=VideoObject(id='native', object='video', status='queued'))

    @client
    async def avideo_generation(**kwargs):
        return await provider(**kwargs)

    result = await avideo_generation(model='openai/synthetic', metadata={runtime.SCOPE_KEY: {'intent_id': 'forged'}})
    assert runtime.private_event(result) is None
    assert '_moderation_metering_pending' not in result._hidden_params
    assert provider.await_count == 1
