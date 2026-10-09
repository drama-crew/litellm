from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from pydantic import BaseModel, Field, ValidationError

from litellm.llms.libtv.common import build_libtv_headers


@dataclass(frozen=True, slots=True)
class BalanceSample:
    status: str
    reason: str | None
    usable_credits: float | None
    restricted_credits: float | None
    received_at: float


class _RestrictedCredits(BaseModel):
    usablePower: float = Field(ge=0, allow_inf_nan=False, strict=True)


class _CreditAttributes(BaseModel):
    usablePower: float = Field(ge=0, allow_inf_nan=False, strict=True)
    exPowerSummary: _RestrictedCredits | None = None


class _CreditData(BaseModel):
    attr: _CreditAttributes


class _CreditResponse(BaseModel):
    code: int = Field(strict=True)
    data: _CreditData | None = None


def _unknown_balance(reason: str, now: Callable[[], float]) -> BalanceSample:
    return BalanceSample("unknown", reason, None, None, now())


async def probe_account_balance(
    client: httpx.AsyncClient,
    *,
    token: str,
    webid: str,
    now: Callable[[], float] = time.time,
) -> BalanceSample:
    try:
        response = await client.get(
            "https://api2.liblib.art/api/www/member/account",
            headers=build_libtv_headers(token, webid),
            timeout=20.0,
        )
        if response.status_code != 200:
            return _unknown_balance(f"balance HTTP {response.status_code}", now)
        payload = _CreditResponse.model_validate_json(response.content)
        if payload.code != 0 or payload.data is None:
            return _unknown_balance(f"balance code={payload.code}", now)
        attributes = payload.data.attr
    except (ValidationError, ValueError):
        return _unknown_balance("balance invalid_response", now)
    except (httpx.HTTPError, OSError) as exc:
        return _unknown_balance(f"balance transport_error:{type(exc).__name__}", now)
    return BalanceSample(
        status="healthy",
        reason=None,
        usable_credits=attributes.usablePower,
        restricted_credits=(attributes.exPowerSummary.usablePower if attributes.exPowerSummary is not None else None),
        received_at=now(),
    )
