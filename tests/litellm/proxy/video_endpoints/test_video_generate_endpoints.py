import json

import pytest

VALID = {
    "task_id": "8f3c1d2e-0000-4000-8000-000000000001",
    "model": "drama-video-1",
    "deadline_ts": 4102444800.0,
    "request": {
        "prompt": "a cinematic drone shot over mountains at sunrise",
        "duration_seconds": 5,
        "resolution": "720p",
        "ratio": "16:9",
        "references": [
            {
                "role": "first_frame",
                "media_type": "image",
                "url": "https://source.example/refs/first-frame.png",
            }
        ],
    },
    "staging_upload": {
        "url": "https://target.example/staging/video-tasks/8f3c1d2e.mp4",
        "key": "staging/video-tasks/8f3c1d2e.mp4",
        "content_type": "video/mp4",
        "expires_at": 4102445000.0,
    },
}

RESULT = {
    "staging_key": "staging/video-tasks/8f3c1d2e.mp4",
    "bytes": 12345678,
    "content_type": "video/mp4",
    "duration_seconds": 5.0,
    "width": 1280,
    "height": 720,
}


@pytest.mark.asyncio
async def test_post_returns_202_with_task_id(client_admin, fake_redis):
    r = await client_admin.post('/v1/libtv/video-generate', json=VALID)
    assert r.status_code == 202
    body = r.json()
    assert body['ok'] is True and body['task_id']


@pytest.mark.asyncio
async def test_post_writes_status_nx_before_xadd(client_admin, fake_redis):
    """写序不可颠倒：先 SET status NX 再 XADD。
    反过来的话，一次极快的 claim 会在 routes/worker_runner.py:254-263
    找不到 task_id/status 键 → 静默 xack → 任务凭空消失、generation 永远 RUNNING。"""
    await client_admin.post('/v1/libtv/video-generate', json=VALID)
    ops = [op for op in fake_redis.calls if op[0] in ('set', 'xadd')]
    assert [op[0] for op in ops] == ['set', 'xadd']
    assert ops[0][2].get('nx') is True          # 必须带 nx=True
    assert ops[0][2].get('ex')                  # 且带 TTL


@pytest.mark.asyncio
async def test_duplicate_task_id_returns_202_without_xadd(client_admin, fake_redis):
    """SET NX 是唯一的去重点（cas.py:83 允许从 claimed 再认领，CAS 层不提供幂等）。"""
    fake_redis.set_nx_result = False
    r = await client_admin.post('/v1/libtv/video-generate', json=VALID)
    assert r.status_code == 202
    assert not [op for op in fake_redis.calls if op[0] == 'xadd']


@pytest.mark.asyncio
async def test_non_admin_is_403(client_user):
    r = await client_user.post('/v1/libtv/video-generate', json=VALID)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_reference_url_outside_allowlist_is_422_invalid_url(client_admin):
    bad = dict(VALID)
    bad['references'] = [{'role': 'first_frame', 'media_type': 'image',
                          'url': 'https://evil.com/a.png'}]
    r = await client_admin.post('/v1/libtv/video-generate', json=bad)
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'invalid_url'


@pytest.mark.asyncio
async def test_nested_reference_url_outside_allowlist_is_422_invalid_url(client_admin):
    """上一个用例把 references 放在**顶层**，而 spec §1.1 的真实形状是嵌在
    request 里的。顶层那条只是防御性兜底，真正会被生产流量走到的是这条嵌套
    路径——它必须独立有测试，否则删掉 _iter_reference_urls 里的嵌套来源，
    白名单对真实请求形状形同虚设而测试全绿。"""
    bad = {**VALID, 'request': {**VALID['request'],
                                'references': [{'role': 'first_frame',
                                                'media_type': 'image',
                                                'url': 'https://evil.com/a.png'}]}}
    r = await client_admin.post('/v1/libtv/video-generate', json=bad)
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'invalid_url'


@pytest.mark.asyncio
async def test_empty_allowlist_is_503_not_422(client_admin, monkeypatch):
    """fail-closed 的配置缺失是服务端错误，不是用户输入错误
    （沿用 validated_transfer._parse_hosts 的既有语义）。"""
    monkeypatch.setenv('LIBTV_VIDEO_GENERATE_SOURCE_HOSTS', '')
    r = await client_admin.post('/v1/libtv/video-generate', json=VALID)
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_no_live_worker_is_422_no_worker_available(client_admin, fake_redis):
    fake_redis.alive = []
    r = await client_admin.post('/v1/libtv/video-generate', json=VALID)
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'no_worker_available'


@pytest.mark.asyncio
async def test_worker_alive_but_full_is_422_no_capacity(client_admin, fake_redis):
    fake_redis.alive = ['vw-1']
    fake_redis.capacity = {'vw-1': 0}
    r = await client_admin.post('/v1/libtv/video-generate', json=VALID)
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'no_capacity_available'


@pytest.mark.asyncio
async def test_extra_field_is_422_invalid_params(client_admin):
    r = await client_admin.post('/v1/libtv/video-generate', json={**VALID, 'nope': 1})
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'invalid_params'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'status,expected',
    [('queued', 'queued'), ('claimed', 'claimed'), ('done', 'succeeded'),
     ('failed', 'failed'), ('cancelled', 'failed')],
)
async def test_get_status_mapping(client_admin, fake_redis, status, expected):
    fake_redis.status['t1'] = status
    fake_redis.result['t1'] = json.dumps({'ok': status == 'done', 'result': RESULT})
    r = await client_admin.get('/v1/libtv/video-generate/t1')
    assert r.json()['status'] == expected


@pytest.mark.asyncio
async def test_get_cancelled_carries_cancelled_error_code(client_admin, fake_redis):
    fake_redis.status['t1'] = 'cancelled'
    r = await client_admin.get('/v1/libtv/video-generate/t1')
    assert r.json()['error']['code'] == 'cancelled'


@pytest.mark.asyncio
async def test_get_unknown_task_is_404(client_admin, fake_redis):
    r = await client_admin.get('/v1/libtv/video-generate/nope')
    assert r.status_code == 404
    assert r.json()['error']['code'] == 'unknown_task'


@pytest.mark.asyncio
async def test_get_done_with_expired_result_is_result_expired(client_admin, fake_redis):
    """status_key TTL 24h、result_key TTL 1h —— 这个窗口真实存在，必须有确定行为。"""
    fake_redis.status['t1'] = 'done'
    fake_redis.result.pop('t1', None)
    body = (await client_admin.get('/v1/libtv/video-generate/t1')).json()
    assert body['status'] == 'failed'
    assert body['error']['code'] == 'result_expired'


@pytest.mark.asyncio
async def test_get_malformed_result_is_rejected(client_admin, fake_redis):
    fake_redis.status['t1'] = 'done'
    fake_redis.result['t1'] = json.dumps({'ok': True, 'result': {'staging_key': 'k'}})  # 缺字段
    body = (await client_admin.get('/v1/libtv/video-generate/t1')).json()
    assert body['status'] == 'failed'
    assert body['error']['code'] == 'malformed_result'


@pytest.mark.asyncio
async def test_get_splits_flat_worker_error_into_structured(client_admin, fake_redis):
    fake_redis.status['t1'] = 'failed'
    fake_redis.result['t1'] = json.dumps(
        {'ok': False, 'error': 'ark_failed: upstream exploded', 'error_kind': 'transient'}
    )
    err = (await client_admin.get('/v1/libtv/video-generate/t1')).json()['error']
    assert err == {'code': 'ark_failed', 'message': 'upstream exploded', 'kind': 'transient'}


@pytest.mark.asyncio
async def test_get_unsplittable_error_falls_back_to_unknown(client_admin, fake_redis):
    fake_redis.status['t1'] = 'failed'
    fake_redis.result['t1'] = json.dumps({'ok': False, 'error': 'boom', 'error_kind': 'transient'})
    err = (await client_admin.get('/v1/libtv/video-generate/t1')).json()['error']
    assert err['code'] == 'unknown' and err['message'] == 'boom'


@pytest.mark.asyncio
async def test_get_succeeded_returns_result_verbatim(client_admin, fake_redis):
    fake_redis.status['t1'] = 'done'
    fake_redis.result['t1'] = json.dumps({'ok': True, 'result': RESULT})
    assert (await client_admin.get('/v1/libtv/video-generate/t1')).json()['result'] == RESULT
