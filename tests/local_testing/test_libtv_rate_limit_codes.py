"""libtv 的业务限流码必须映射成 HTTP 429。

2026-09-01 causyn.cn：`third_asset/create` 返回

    code=10026 msg=当前素材检测提交较频繁，请控制在每分钟 15 个以内

`_check` 只把 1200000136 判 429，其余一律 502 → `_raise_normalized_libtv_error`
抛 BadGatewayError → 平台 `user_error.py` 的 429 分支落空 → 用户看到
"生成失败，请稍后重试"，看不出该等一等，于是原样重试连撞十次。
"""

import json

import pytest

from litellm.llms.libtv.client import LibTVClient
from litellm.llms.libtv.common import LibTVError


class _Resp:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self._payload = payload
        self.headers = {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _client():
    return LibTVClient(token="tok", webid="wid")


def _raise(payload):
    with pytest.raises(LibTVError) as excinfo:
        _client()._check(_Resp(payload), "third_asset/create")
    return excinfo.value


def test_the_production_rate_limit_code_becomes_429():
    err = _raise(
        {
            "code": 10026,
            "msg": "当前素材检测提交较频繁，请控制在每分钟 15 个以内，给系统留一点处理缓冲时间～",
        }
    )
    assert err.status_code == 429


def test_the_previously_known_rate_limit_code_still_becomes_429():
    assert _raise({"code": 1200000136, "msg": "busy"}).status_code == 429


def test_a_string_code_is_matched_too():
    # 上游偶有把 code 序列化成字符串的情况；一个纯粹的表示差异不该改变判定。
    assert _raise({"code": "10026", "msg": "较频繁"}).status_code == 429


def test_an_unrelated_business_code_stays_502():
    # 只认已知的限流码。把 5xx/其它 4xx 说成"排队已满、稍后重试"是一句自信的
    # 假话，会让用户对着一个永远不会好的错误反复重试。
    assert _raise({"code": 40001, "msg": "参数错误"}).status_code == 502


def test_the_message_still_carries_the_code_for_the_logs():
    err = _raise({"code": 10026, "msg": "当前素材检测提交较频繁"})
    assert "code=10026" in err.message
    assert "当前素材检测提交较频繁" in err.message


def test_a_transport_level_status_is_untouched():
    with pytest.raises(LibTVError) as excinfo:
        _client()._check(_Resp({"code": 0}, status_code=503), "third_asset/create")
    assert excinfo.value.status_code == 503
