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
async def test_top_level_references_key_is_422_invalid_params(client_admin):
    """历史上这条用例把 references 放在顶层，靠 _iter_reference_urls 里一段
    「不是文档化形状」的兜底分支才被拦成 invalid_url（见该函数 docstring 承认
    这不是 spec §1.1 的真实请求形状）。F1 把 _validate_shape 换到 _validate_urls
    前面之后，顶层 references 会先撞上 _ALLOWED_TOP_LEVEL_KEYS 的封闭键集，变成
    invalid_params（未知字段），根本走不到 URL 校验——这是预期行为变化，不是回归：
    真实的嵌套路径由 test_nested_reference_url_outside_allowlist_is_422_invalid_url
    覆盖。F11 会删掉那段已不可达的顶层兜底代码。"""
    bad = dict(VALID)
    bad['references'] = [{'role': 'first_frame', 'media_type': 'image',
                          'url': 'https://evil.com/a.png'}]
    r = await client_admin.post('/v1/libtv/video-generate', json=bad)
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'invalid_params'


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


# ---------------------------------------------------------------------------
# F1: non-dict body must be a deterministic 422, never an uncaught 500.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize('body', [[], 'hi', None, 5, True])
async def test_non_dict_body_is_422_invalid_params_not_500(client_admin, body):
    """F1: payload 不是 dict 时（[]/'hi'/None/5/True），旧代码先跑
    _validate_urls -> _iter_reference_urls 立刻 payload.get("request") 炸
    AttributeError；endpoints.py 只捉 orjson.JSONDecodeError/VideoGenerateError，
    AttributeError 直接以未处理异常逃逸（ASGITransport 默认
    raise_app_exceptions=True，这里表现为 client.post 本身抛出，而不是拿到一个
    500 response）。design §9.4 把非 4xx 判定当 TransientTransferError 重试，会对
    一个确定性坏输入无限重试。修复：_validate_shape 先跑，必须是确定性
    422 invalid_params，且完全不触碰 redis（这条用例故意不给 fake_redis）。"""
    r = await client_admin.post('/v1/libtv/video-generate', json=body)
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'invalid_params'


# ---------------------------------------------------------------------------
# F2: XADD failure after a successful SET NX must roll back + surface 503.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_xadd_failure_after_set_nx_rolls_back_status_key_and_is_503(client_admin, fake_redis):
    """F2: SET NX 成功后 XADD 失败必须回滚——否则第一次请求 500，平台重试时第二次
    请求的 SET NX 因 key 已存在返回 None，代码直接 202 却从未 XADD 过，任务卡在
    queued 直到 24h status TTL 过期后变 404 unknown_task。正确行为：尽力删掉刚写
    的 status key（吞掉删除失败，不掩盖原始异常），然后 raise VideoGenerateError
    让调用方明确拿到 503 + 可识别 code，这样平台重试时 SET NX 才能重新成功、真正
    XADD。参照 validated_transfer.py:477-484 的「ambiguous」回滚模式。"""
    fake_redis.xadd_exception = RuntimeError('redis stream unavailable')
    r = await client_admin.post('/v1/libtv/video-generate', json=VALID)
    assert r.status_code == 503
    assert r.json()['error']['code'] == 'enqueue_ambiguous'
    assert fake_redis.deleted_keys == [f'worker:task:status:{VALID["task_id"]}']


@pytest.mark.asyncio
async def test_xadd_failure_with_failing_rollback_still_surfaces_enqueue_ambiguous(client_admin, fake_redis):
    """回滚删除本身也失败时，不能让 DELETE 的异常掩盖原始的 enqueue 失败——调用方
    仍然必须看到 503 enqueue_ambiguous，而不是一个不相关的 500/异常。"""
    fake_redis.xadd_exception = RuntimeError('redis stream unavailable')
    fake_redis.delete_exception = RuntimeError('redis totally down')
    r = await client_admin.post('/v1/libtv/video-generate', json=VALID)
    assert r.status_code == 503
    assert r.json()['error']['code'] == 'enqueue_ambiguous'


# ---------------------------------------------------------------------------
# F5: an unconfigured redis client must be a loud 503, never a silent
# AttributeError-turned-500 (POST) or a completely uncaught crash (GET).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_without_redis_configured_is_503_not_500(client_admin, monkeypatch):
    """F5: _get_redis_client() 两个 env 都没配时静默返回 None，POST 里
    redis.zrangebyscore 撞上 None -> AttributeError 逃逸成未处理异常（对应探针
    "redis URL 两个 env 都没配 -> _get_redis_client() 返回 None"）。design §8.1
    承诺配置错误要有响亮的 503（空 allowlist 已经这样做了）；今天的不一致是
    "allowlist 缺配 503，redis 缺配 500→无限透传重试"。这条用例故意不用
    fake_redis，直接跑真实的 _get_redis_client()。"""
    monkeypatch.delenv('LIBTV_VIDEO_GENERATE_REDIS_URL', raising=False)
    monkeypatch.delenv('MEDIA_TRANSFER_REDIS_URL', raising=False)
    r = await client_admin.post('/v1/libtv/video-generate', json=VALID)
    assert r.status_code == 503
    assert r.json()['error']['code'] == 'misconfigured'


@pytest.mark.asyncio
async def test_get_without_redis_configured_is_503_not_500(client_admin, monkeypatch):
    """GET 路径原本连 try/except 都没有，比 POST 更糟——AttributeError 直接从
    handler 里逃逸。"""
    monkeypatch.delenv('LIBTV_VIDEO_GENERATE_REDIS_URL', raising=False)
    monkeypatch.delenv('MEDIA_TRANSFER_REDIS_URL', raising=False)
    r = await client_admin.get('/v1/libtv/video-generate/t1')
    assert r.status_code == 503
    assert r.json()['error']['code'] == 'misconfigured'


# ---------------------------------------------------------------------------
# F6: fail-closed URL validation -- a malformed reference/staging_upload must
# be a deterministic 422, never silently skipped past URL/allowlist checking.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reference_item_not_dict_is_422_invalid_url(client_admin):
    """F6: 旧代码 `isinstance(item, dict) and isinstance(item.get("url"), str)`
    条件为假时直接跳过该 item，既不 yield 它的 url 也不报错——一个非 dict 的
    reference（比如整条写成字符串）会被无声无息地放行，白名单形同虚设。"""
    bad = {**VALID, 'request': {**VALID['request'], 'references': ['not-a-dict']}}
    r = await client_admin.post('/v1/libtv/video-generate', json=bad)
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'invalid_url'


@pytest.mark.asyncio
async def test_reference_missing_url_is_422_invalid_url(client_admin):
    """F6: reference 是 dict 但没有 url 键——旧代码同样静默跳过，不报错。"""
    bad = {**VALID, 'request': {**VALID['request'],
                                'references': [{'role': 'first_frame', 'media_type': 'image'}]}}
    r = await client_admin.post('/v1/libtv/video-generate', json=bad)
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'invalid_url'


@pytest.mark.asyncio
async def test_reference_url_non_string_is_422_invalid_url(client_admin):
    """F6: url 键存在但不是字符串（比如数字）——旧代码的 isinstance(...,str) 门槛
    只用来决定「要不要 yield 去给 _check_url 校验」，不是 dict/字符串就直接跳过，
    _check_url 自己的类型检查根本没机会跑到，等于没校验。"""
    bad = {**VALID, 'request': {**VALID['request'],
                                'references': [{'role': 'first_frame', 'media_type': 'image', 'url': 123}]}}
    r = await client_admin.post('/v1/libtv/video-generate', json=bad)
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'invalid_url'


@pytest.mark.asyncio
async def test_staging_upload_missing_url_is_422_invalid_url(client_admin):
    """F6: staging_upload 是 dict 但没有 url 键——旧代码
    `isinstance(staging_upload, dict) and "url" in staging_upload` 为假时
    直接跳过整个 _check_url 调用，请求带着一个没有 url 的 staging_upload
    静默放行，下游 worker 完成整段 GPU 计算后却没有回传地址。"""
    bad = {**VALID, 'staging_upload': {'key': VALID['staging_upload']['key'],
                                        'content_type': 'video/mp4',
                                        'expires_at': VALID['staging_upload']['expires_at']}}
    r = await client_admin.post('/v1/libtv/video-generate', json=bad)
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'invalid_url'


@pytest.mark.asyncio
async def test_staging_upload_url_non_string_is_422_invalid_url(client_admin):
    """F6: staging_upload.url 存在但不是字符串——旧代码的 "url" in staging_upload
    只检查键是否存在，不检查值类型，非字符串的 url 会被当成「有 url」直接传给
    _check_url，但更隐蔽的是键缺失时连这条路径都不会走到（见上一条用例）。这条
    用例锁定「键存在但类型错」也必须被拒绝，不能悄悄放过。"""
    bad = {**VALID, 'staging_upload': {**VALID['staging_upload'], 'url': 123}}
    r = await client_admin.post('/v1/libtv/video-generate', json=bad)
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'invalid_url'


# ---------------------------------------------------------------------------
# F7: closed key sets for request / references[i] / staging_upload -- an
# unrecognized field nested inside any of these three objects must be
# rejected the same way an unrecognized top-level field already is.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_extra_field_is_422_invalid_params(client_admin):
    """F7: 顶层已有封闭键集校验（_ALLOWED_TOP_LEVEL_KEYS），但 request 对象内部
    今天没有——多塞一个未知字段（比如 callback_url）会被原样转发进 envelope，
    塞进 Redis Stream 交给 worker，而不是在入口就被拒绝。"""
    bad = {**VALID, 'request': {**VALID['request'], 'callback_url': 'https://evil.com/cb'}}
    r = await client_admin.post('/v1/libtv/video-generate', json=bad)
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'invalid_params'


@pytest.mark.asyncio
async def test_reference_extra_field_is_422_invalid_params(client_admin):
    """F7: reference item（spec §1.1: role/media_type/url）今天也没有封闭键集
    校验，可以夹带任意额外字段。"""
    bad = {**VALID, 'request': {**VALID['request'], 'references': [
        {'role': 'first_frame', 'media_type': 'image',
         'url': VALID['request']['references'][0]['url'], 'checksum': 'deadbeef'},
    ]}}
    r = await client_admin.post('/v1/libtv/video-generate', json=bad)
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'invalid_params'


@pytest.mark.asyncio
async def test_staging_upload_extra_field_is_422_invalid_params(client_admin):
    """F7: staging_upload（spec §1.1: url/key/content_type/expires_at）同样
    今天没有封闭键集校验。"""
    bad = {**VALID, 'staging_upload': {**VALID['staging_upload'], 'region': 'us-east-1'}}
    r = await client_admin.post('/v1/libtv/video-generate', json=bad)
    assert r.status_code == 422
    assert r.json()['error']['code'] == 'invalid_params'


# ---------------------------------------------------------------------------
# F3: capacity admission must degrade to alive-only when no *alive* worker
# has ever reported free_slots -- spec §3.5: "`free_slots` 缺省时不写容量
# 键；requester 侧对没有容量信息的任务类型跳过第 2 项检查（退化为只判活）"。
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_capacity_hash_totally_empty_degrades_to_alive_only(client_admin, fake_redis):
    """F3: 该任务类型从来没有任何 worker 上报过 free_slots（老 worker 二进制，
    或心跳还没来得及带上容量字段）——容量哈希整个是空的。旧实现
    `sum(capacity_by_worker.get(name, 0) for name in alive)` 对每个 alive
    worker 取不到 hash 记录时用 0 兜底，于是永远算出 total_free=0，把『完全没有
    容量信息』误判成『容量确实是 0』，无差别拒绝这个任务类型的所有请求——
    即使实际上什么都不知道。spec §3.5 明确要求这种情况『退化为只判活』，
    不是当成 0 空槽拒绝。"""
    fake_redis.alive = ['vw-old-binary']
    fake_redis.capacity = {}
    r = await client_admin.post('/v1/libtv/video-generate', json=VALID)
    assert r.status_code == 202


@pytest.mark.asyncio
async def test_capacity_only_known_alive_workers_count_towards_total(client_admin, fake_redis):
    """F3 的更精确形状：不是『哈希整体是否为空』这么粗的判断，而是『alive 且在
    容量哈希里有记录』的 worker 集合（known）是否为空。这条用例里容量哈希非空，
    但唯一的记录属于一个已经不在 alive zset 里的陈旧 worker——当前唯一 alive
    的 worker 完全没有容量记录，known 集合仍然应该是空，必须退化为只判活，
    而不是因为『hgetall 返回了非空 dict』就走正常求和路径。"""
    fake_redis.alive = ['vw-new-unreported']
    fake_redis.capacity = {'vw-stale-not-alive': 5}
    r = await client_admin.post('/v1/libtv/video-generate', json=VALID)
    assert r.status_code == 202


# ---------------------------------------------------------------------------
# F8: _valid_result must type-check (not just presence-check) each of the 4
# existing required result fields. Deliberately does NOT add any new
# required field -- rejecting a result this late discards a full (GPU-
# minutes) completed generation.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_result_with_string_bytes_is_malformed_result(client_admin, fake_redis):
    """F8: 旧实现只检查 `"bytes" in result`（字段是否存在），不检查类型——
    worker 上报 "bytes": "12345678"（字符串而非数字）会被当成合法结果直接
    放行给平台侧，污染下游消费者对该字段类型的假设。"""
    bad_result = {**RESULT, 'bytes': str(RESULT['bytes'])}
    fake_redis.status['t1'] = 'done'
    fake_redis.result['t1'] = json.dumps({'ok': True, 'result': bad_result})
    body = (await client_admin.get('/v1/libtv/video-generate/t1')).json()
    assert body['status'] == 'failed'
    assert body['error']['code'] == 'malformed_result'


@pytest.mark.asyncio
async def test_get_result_with_bool_duration_seconds_is_malformed_result(client_admin, fake_redis):
    """F8: Python 里 bool 是 int 的子类型——如果类型检查天真地写成
    isinstance(x, (int, float))，duration_seconds=True 会被误判成合法数字。
    必须显式排除 bool。"""
    bad_result = {**RESULT, 'duration_seconds': True}
    fake_redis.status['t1'] = 'done'
    fake_redis.result['t1'] = json.dumps({'ok': True, 'result': bad_result})
    body = (await client_admin.get('/v1/libtv/video-generate/t1')).json()
    assert body['status'] == 'failed'
    assert body['error']['code'] == 'malformed_result'


# ---------------------------------------------------------------------------
# F10: every error-path response body must share one {"error": {"code": str,
# "message": str}} shape. 403 was the sole holdout (a bare string).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_admin_post_403_body_matches_error_shape(client_user):
    r = await client_user.post('/v1/libtv/video-generate', json=VALID)
    assert r.status_code == 403
    assert r.json() == {'error': {'code': 'forbidden', 'message': 'proxy-admin authorization is required'}}


@pytest.mark.asyncio
async def test_non_admin_get_403_body_matches_error_shape(client_user):
    r = await client_user.get('/v1/libtv/video-generate/t1')
    assert r.status_code == 403
    assert r.json() == {'error': {'code': 'forbidden', 'message': 'proxy-admin authorization is required'}}


@pytest.mark.asyncio
async def test_all_error_bodies_share_error_code_message_shape(client_admin, client_user, monkeypatch):
    """F10 回归锁定：不是只测 403 本身，而是把有代表性的几条错误路径
    （403 / 422 invalid_params / 503 misconfigured）放在一起断言它们的响应体
    都是同一个 {"error": {"code": str, "message": str}} 形状。403 曾经是唯一
    的例外（裸字符串 {"error": "..."}），但此前从没有一条测试把『所有』错误
    路径的形状放在一起锁定过——以后新增错误路径时很容易不知不觉引入第二种
    形状而不被任何测试发现。故意不请求 fake_redis fixture：它会整体替换掉
    _get_redis_client，导致最后一条『redis 未配置』的 503 用例测不到目标路径。
    """
    responses = [
        await client_user.post('/v1/libtv/video-generate', json=VALID),  # 403
        await client_user.get('/v1/libtv/video-generate/t1'),  # 403
        await client_admin.post('/v1/libtv/video-generate', json=[]),  # 422 invalid_params
    ]
    monkeypatch.delenv('LIBTV_VIDEO_GENERATE_REDIS_URL', raising=False)
    monkeypatch.delenv('MEDIA_TRANSFER_REDIS_URL', raising=False)
    responses.append(await client_admin.get('/v1/libtv/video-generate/t1'))  # 503 misconfigured
    for r in responses:
        assert r.status_code >= 400
        body = r.json()
        assert isinstance(body['error'], dict), body
        assert isinstance(body['error']['code'], str)
        assert isinstance(body['error']['message'], str)
