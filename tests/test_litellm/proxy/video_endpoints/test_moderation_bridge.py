import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.video_endpoints import endpoints
from litellm.types.videos.main import VideoObject


@pytest.mark.asyncio
@pytest.mark.parametrize('path,payload', [
    ('/v1/videos', {'model': 'hailuo-h3', 'prompt': 'synthetic'}),
    ('/videos', {'model': 'hailuo-h3', 'prompt': 'synthetic'}),
    ('/videos/mod_video_source/remix', {'model': 'hailuo-h3', 'prompt': 'synthetic'}),
    ('/v1/videos/mod_video_source/remix', {'model': 'hailuo-h3', 'prompt': 'synthetic'}),
    ('/videos/edits', {'model': 'hailuo-h3', 'prompt': 'synthetic', 'video': {'id': 'mod_video_source'}}),
    ('/v1/videos/edits', {'model': 'hailuo-h3', 'prompt': 'synthetic', 'video': {'id': 'mod_video_source'}}),
    ('/videos/extensions', {'model': 'hailuo-h3', 'prompt': 'synthetic', 'video': {'id': 'mod_video_source'}}),
    ('/v1/videos/extensions', {'model': 'hailuo-h3', 'prompt': 'synthetic', 'video': {'id': 'mod_video_source'}}),
])
async def test_pending_input_never_submits_provider(path, payload, monkeypatch):
    monkeypatch.setenv('DRAMA_MODERATION_PLATFORM_URL', 'http://platform.test')
    monkeypatch.setenv('DRAMA_MODERATION_SERVICE_TOKEN', 'test-token')
    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key='a' * 64, user_id='owner', team_id='owner',
        metadata={'type': 'personal_openapi', 'openapi_key_id': 'key-id'},
    )
    async def platform_request(request):
        assert request.headers['authorization'] == 'Bearer test-token'
        data = json.loads(request.content)
        assert data['principal']['user_id'] == 'owner'
        return httpx.Response(200, json={
            'id': 'mod_video_stable', 'state': 'input_moderation', 'model': 'hailuo-h3',
            'moderation_status': 'pending', 'created_at': 1, 'policy_source': 'model', 'output': None,
        })
    app.state.moderation_transport = httpx.MockTransport(platform_request)
    provider = AsyncMock(return_value=VideoObject(id='native', object='video', status='queued'))
    with patch.object(ProxyBaseLLMRequestProcessing, 'base_process_llm_request', provider):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post(path, json=payload)
    assert provider.call_count == 0
    assert response.status_code == 200, response.text
    assert response.json()['id'] == 'mod_video_stable'
    assert response.json()['moderation_status'] == 'pending'


@pytest.mark.asyncio
@pytest.mark.parametrize('route,payload', [
    ('avideo_remix', {'source_video_id': 'video_encoded'}),
    ('avideo_edit', {'video': {'id': 'video_encoded'}}),
    ('avideo_extension', {'video': {'id': 'video_encoded'}}),
])
async def test_resume_reuses_existing_source_adapter(route, payload):
    from starlette.requests import Request

    from litellm.proxy.video_endpoints.moderation_execution import execution_request, invoke
    from litellm.types.videos.utils import encode_video_id_with_provider

    native = encode_video_id_with_provider('provider-native', 'openai', 'deployment')
    data = {'model': 'hailuo-h3', **({'source_video_id': native} if route == 'avideo_remix' else {'video': {'id': native}})}
    app = FastAPI()
    request = execution_request(Request({'type': 'http', 'method': 'POST', 'path': '/', 'headers': [], 'app': app}), data, '/v1/videos')
    seen = []
    async def process(processor, **kwargs):
        seen.append(processor.data)
        return VideoObject(id='accepted', object='video', status='queued')
    with patch.object(ProxyBaseLLMRequestProcessing, 'base_process_llm_request', process):
        result = await invoke(request, UserAPIKeyAuth(), data, route)
    assert result.id == 'accepted'
    assert seen[0]['video_id'] == native
    assert seen[0]['custom_llm_provider'] == 'openai'
    assert seen[0]['num_retries'] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['/v1/videos/characters', '/videos/characters'])
async def test_character_pending_preserves_character_protocol(monkeypatch, path):
    from litellm.proxy.video_endpoints import moderation_bridge as bridge
    monkeypatch.setenv('DRAMA_MODERATION_PLATFORM_URL', 'http://platform.test')
    monkeypatch.setenv('DRAMA_MODERATION_SERVICE_TOKEN', 'test-token')
    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key='a' * 64, user_id='owner', team_id='owner', metadata={'openapi_key_id': 'key-id'})
    async def control(request):
        body = json.loads(request.content)
        assert body['model'] == 'hailuo-h3'
        assert body['payload']['video'].startswith('oss://')
        return httpx.Response(200, json={'id': 'mod_video_character', 'model': 'hailuo-h3', 'state': 'input_moderation',
            'moderation_status': 'pending', 'policy_source': 'model', 'created_at': 1, 'parameters': {'name': 'Synthetic'}})
    app.state.moderation_transport = httpx.MockTransport(control)
    with patch.object(bridge, 'upload_media', AsyncMock(return_value='oss://private/input')), patch.object(
        ProxyBaseLLMRequestProcessing, 'base_process_llm_request', AsyncMock()) as provider:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post(path, data={'name': 'Synthetic', 'target_model_names': 'hailuo-h3'},
                                         files={'video': ('synthetic.mp4', b'synthetic', 'video/mp4')})
    assert provider.await_count == 0
    assert response.status_code == 200, response.text
    assert response.json()['object'] == 'character'
    assert response.json()['name'] == 'Synthetic'
    assert response.json()['moderation_status'] == 'pending'


@pytest.mark.asyncio
async def test_real_auth_gate_defers_budget_until_bound_admission(monkeypatch):
    import importlib

    from starlette.requests import Request

    from litellm.proxy import proxy_server
    from litellm.proxy.video_endpoints import moderation_bridge as bridge
    gate = importlib.import_module('litellm.proxy.auth.user_api_key_auth')
    monkeypatch.setenv('DRAMA_MODERATION_PLATFORM_URL', 'http://platform.test')
    monkeypatch.setattr(proxy_server, 'master_key', 'synthetic-master')
    monkeypatch.setattr(proxy_server, 'user_custom_auth', None)
    for name in ('get_team_object', 'get_user_object', 'get_global_proxy_spend', 'get_end_user_object'):
        monkeypatch.setattr(gate, name, AsyncMock(return_value=None))
    common = AsyncMock(return_value=True)
    reserve = AsyncMock()
    monkeypatch.setattr(gate, 'common_checks', common)
    monkeypatch.setattr(gate, '_reserve_budget_after_common_checks', reserve)
    auth = UserAPIKeyAuth(api_key='a' * 64, user_id='owner', team_id='owner', end_user_id='end',
                         metadata={'openapi_key_id': 'key-id'})
    request = Request({'type': 'http', 'method': 'POST', 'path': '/v1/videos', 'headers': []})
    await gate._run_centralized_common_checks(auth, request, {'model': 'hailuo-h3'}, '/v1/videos')
    assert common.await_count == 1 and reserve.await_count == 0
    request.scope['moderation_admission'] = 'client-forged-true'
    await gate._run_centralized_common_checks(auth, request, {'model': 'hailuo-h3'}, '/v1/videos')
    assert reserve.await_count == 0
    request.scope['moderation_admission'] = bridge.ADMITTED
    await gate._run_centralized_common_checks(auth, request, {'model': 'hailuo-h3'}, '/v1/videos')
    assert reserve.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('path,code', [
    ('/v1/videos/mod_video_stable', 200), ('/videos/mod_video_stable', 200),
    ('/v1/videos/mod_video_stable/content', 409), ('/videos/mod_video_stable/content', 409),
    ('/v1/videos', 200), ('/videos', 200),
    ('/v1/videos/characters/mod_video_stable', 200), ('/videos/characters/mod_video_stable', 200),
    ('/v2/query/video_generation/mod_video_stable', 200), ('/v2/query/video_generation', 200),
])
async def test_all_public_read_surfaces_hide_pending_output(path, code, monkeypatch):
    from litellm.proxy.video_endpoints import context_ir_endpoints, minimax_h3_endpoints
    monkeypatch.setenv('DRAMA_MODERATION_PLATFORM_URL', 'http://platform.test')
    monkeypatch.setenv('DRAMA_MODERATION_SERVICE_TOKEN', 'test-token')
    app = FastAPI()
    app.include_router(endpoints.router)
    app.include_router(minimax_h3_endpoints.router)
    app.include_router(context_ir_endpoints.router)
    app.dependency_overrides[context_ir_endpoints.context_ir_service] = lambda: None
    app.dependency_overrides[minimax_h3_endpoints.context_ir_service_for_request] = lambda: None
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key='a' * 64, user_id='owner', team_id='owner', metadata={'openapi_key_id': 'key-id'})
    view = {'id': 'mod_video_stable', 'model': 'hailuo-h3', 'state': 'output_moderation', 'moderation_status': 'pending',
            'policy_source': 'model', 'created_at': 1, 'output': {'url': 'https://private.invalid/leak'}, 'parameters': {'name': 'Synthetic'}}
    async def control(request):
        return httpx.Response(200, json={'items': [view]} if request.url.path.endswith('/list') else view)
    app.state.moderation_transport = httpx.MockTransport(control)
    provider = AsyncMock()
    with patch.object(ProxyBaseLLMRequestProcessing, 'base_process_llm_request', provider):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.get(path)
    assert response.status_code == code, response.text
    assert 'private.invalid' not in response.text
    assert provider.await_count == 0


@pytest.mark.asyncio
async def test_approved_content_uses_oss_302_and_no_media_proxy(monkeypatch):
    from starlette.requests import Request

    from litellm.proxy.video_endpoints import moderation_bridge as bridge
    monkeypatch.setenv('DRAMA_MODERATION_PLATFORM_URL', 'http://platform.test')
    view = {'id': 'mod_video_stable', 'model': 'hailuo-h3', 'state': 'completed', 'moderation_status': 'approved',
            'policy_source': 'model', 'created_at': 1, 'output': {'url': 'https://private-bucket.example/signed'}}
    monkeypatch.setattr(bridge, 'platform', AsyncMock(return_value=view))
    auth = UserAPIKeyAuth(api_key='a' * 64, user_id='owner', team_id='owner', metadata={'openapi_key_id': 'key'})
    response = await bridge.download(Request({'type': 'http', 'method': 'GET', 'path': '/', 'headers': []}), auth, 'mod_video_stable')
    assert response.status_code == 302
    assert response.headers['location'] == view['output']['url']


@pytest.mark.asyncio
async def test_legacy_source_requires_exact_key_owner_before_provider_read(monkeypatch):
    from starlette.requests import Request

    from litellm.proxy.video_endpoints import moderation_bridge as bridge
    monkeypatch.setattr(bridge, 'platform', AsyncMock(return_value={'bound': False}))
    class DB:
        async def query_raw(self, sql, owner, task):
            assert 'owner=$1 AND task_id=$2' in sql
            return [{'id': 'old-log'}] if owner == 'a' * 64 and task == 'old-native' else []
    monkeypatch.setattr(endpoints.openapi_log_capture, 'database', lambda: DB())
    result = VideoObject(id='old-native', object='video', model='hailuo-h3', status='completed')
    result._hidden_params = {'url': 'https://synthetic.invalid/legacy'}
    reader = AsyncMock(return_value=result)
    monkeypatch.setattr(endpoints, 'video_status', reader)
    request = Request({'type': 'http', 'method': 'POST', 'path': '/', 'headers': [], 'app': FastAPI()})
    wrong = UserAPIKeyAuth(api_key='b' * 64, user_id='owner', team_id='owner', metadata={'openapi_key_id': 'other'})
    with pytest.raises(Exception) as denied:
        await bridge.source_descriptor(request, wrong, {'video': {'id': 'old-native'}})
    assert denied.value.status_code == 404 and reader.await_count == 0
    auth = wrong.model_copy(update={'api_key': 'a' * 64})
    description = await bridge.source_descriptor(request, auth, {'video': {'id': 'old-native'}})
    assert description == {'requested_id': 'old-native', 'native_id': 'old-native', 'model': 'hailuo-h3', 'url': 'https://synthetic.invalid/legacy'}
    assert reader.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['/v2/video_generation', '/v2/h3_context_ir'])
async def test_v2_creation_precedes_rewrite_and_context_ir_budget(path, monkeypatch):
    from types import SimpleNamespace

    from litellm.proxy.video_endpoints import context_ir_endpoints as ir
    from litellm.proxy.video_endpoints import minimax_h3_endpoints as h3
    monkeypatch.setenv('DRAMA_MODERATION_PLATFORM_URL', 'http://platform.test')
    monkeypatch.setenv('DRAMA_MODERATION_SERVICE_TOKEN', 'test-token')
    monkeypatch.setenv('LITELLM_VIDEO_ID_SECRET', 'synthetic-test-secret')
    app = FastAPI()
    app.include_router(h3.router)
    app.include_router(ir.router)
    service = SimpleNamespace(create=AsyncMock())
    app.dependency_overrides[ir.context_ir_service] = lambda: service
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key='a' * 64, user_id='owner', team_id='owner', metadata={'openapi_key_id': 'key-id'})
    async def control(request):
        payload = json.loads(request.content)
        assert payload['model'] == ('causyn-1.1' if path.endswith('h3_context_ir') else 'hailuo-h3')
        return httpx.Response(200, json={'id': 'mod_video_v2', 'model': payload['model'], 'state': 'input_moderation',
            'moderation_status': 'pending', 'policy_source': 'model', 'created_at': 1})
    app.state.moderation_transport = httpx.MockTransport(control)
    with patch.object(ir, 'require_durable_budget') as budget, patch.object(ProxyBaseLLMRequestProcessing, 'base_process_llm_request', AsyncMock()) as provider:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            response = await client.post(path, json={'model': 'MiniMax-H3', 'content': [{'type': 'text', 'text': 'synthetic'},
                {'type': 'image_url', 'image_url': {'url': 'https://synthetic.invalid/input'}, 'role': 'first_frame'}], 'duration': 8, **({'resolution': '768P'} if path.endswith('video_generation') else {})})
    assert response.status_code == 200, response.text
    assert response.json()['task_id'] == 'mod_video_v2'
    assert budget.call_count == provider.await_count == service.create.await_count == 0


@pytest.mark.asyncio
async def test_stable_context_ir_cancel_preserves_protocol_before_provider(monkeypatch):
    from litellm.proxy.video_endpoints import context_ir_endpoints as ir

    monkeypatch.setenv('DRAMA_MODERATION_PLATFORM_URL', 'http://platform.test')
    monkeypatch.setenv('DRAMA_MODERATION_SERVICE_TOKEN', 'synthetic')
    app = FastAPI()
    app.include_router(ir.router)
    service = AsyncMock()
    app.dependency_overrides[ir.context_ir_service] = lambda: service
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key='a' * 64, user_id='owner', team_id='owner', metadata={'openapi_key_id': 'key'})
    async def control(request):
        assert request.url.path == '/internal/moderation/intents/mod_video_stable/cancel'
        assert json.loads(request.content)['principal']['user_id'] == 'owner'
        return httpx.Response(200, json={'action': 'cancelled'})
    app.state.moderation_transport = httpx.MockTransport(control)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        response = await client.delete('/v2/video_generation/mod_video_stable')
    assert response.status_code == 200, response.text
    assert response.json() == {'task_id': 'mod_video_stable', 'action': 'cancelled', 'status': 'cancelled'}
    service.cancel_or_delete.assert_not_called()
