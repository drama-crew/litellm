import json
from dataclasses import asdict

import httpx
import pytest

from litellm.llms.libtv.account_balance import probe_account_balance
from litellm.llms.libtv.account_health import LibTVAccount, LibTVAccountHealthProber


@pytest.mark.asyncio
@pytest.mark.parametrize("usable", [0, 1500, 9000, 50520])
async def test_balance_preserves_generic_and_restricted_credits_separately(usable):
    def handler(request):
        assert request.method == "GET"
        assert str(request.url) == "https://api2.liblib.art/api/www/member/account"
        assert request.headers["token"] == "private-token"
        assert request.headers["webid"] == "private-webid"
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {"attr": {"usablePower": usable, "exPowerSummary": {"usablePower": 99999}}},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        sample = asdict(
            await probe_account_balance(client, token="private-token", webid="private-webid", now=lambda: 1000)
        )
    assert sample == {
        "status": "healthy",
        "reason": None,
        "usable_credits": usable,
        "restricted_credits": 99999,
        "received_at": 1000,
    }
    assert "private" not in json.dumps(sample)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"code": 401, "msg": "private-token private-webid"},
        {"code": 0, "data": {"attr": {}}},
        {"code": 0, "data": {"attr": {"usablePower": True}}},
        {"code": 0, "data": {"attr": {"usablePower": -1}}},
        {"code": 0, "data": {"attr": {"usablePower": "nan"}}},
        {"code": 0, "data": None},
    ],
)
async def test_invalid_balance_is_unknown_and_never_zero(payload):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
        sample = asdict(await probe_account_balance(client, token="private-token", webid="private-webid"))
    assert sample["status"] == "unknown"
    assert sample["usable_credits"] is None
    assert "private" not in json.dumps(sample)


@pytest.mark.asyncio
async def test_balance_fault_does_not_turn_valid_credentials_unhealthy():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"code": 0, "data": {"uuid": "user"}})
        raise httpx.ConnectError("private-token private-webid", request=request)

    prober = LibTVAccountHealthProber(
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    sample = await prober.probe(LibTVAccount("account-1", "private-token", "private-webid", "key"))
    assert sample["status"] == "healthy"
    assert sample["credits"]["status"] == "unknown"
    assert sample["credits"]["usable_credits"] is None
    assert "private" not in json.dumps(sample)


@pytest.mark.asyncio
async def test_expired_credentials_skip_the_balance_request():
    calls = []

    def handler(request):
        calls.append(request.method)
        return httpx.Response(200, json={"code": 401, "msg": "No login"})

    prober = LibTVAccountHealthProber(
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    sample = await prober.probe(LibTVAccount("account-1", "private-token", "private-webid", "key"))
    assert sample["status"] == "unhealthy"
    assert sample["credits"] is None
    assert calls == ["POST"]
