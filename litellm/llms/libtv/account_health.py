"""Liveness probing for every libtv account the router is configured with.

Why (2026-09-01 causyn.cn): seedance-2.5 runs on a two-account libtv pool. One
of the two tokens had gone dead -- every request through it logged
`getUserInfo code=401 No login` -- and nothing noticed, because the causyn
health monitor watches video_generate worker heartbeats and the samples those
workers publish, neither of which says anything about a vendor credential. The
pool looked "up" the whole time: the surviving account absorbed the traffic
until the day it hit a rate limit, and then all ten submits failed at once.

Shape follows the existing worker health path exactly -- whoever holds the
credential probes and publishes a sample into Redis, and the monitor only ever
reads samples. litellm already holds every libtv token, so probing here means
nothing has to be copied into the monitor's Secret (two copies of a credential
is one rotation away from a permanent false alarm).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from litellm.llms.libtv.common import LIBTV_PASSPORT_BASE, LibTVError, build_libtv_headers
from litellm.llms.libtv.persistence import account_key as derive_account_key

logger = logging.getLogger(__name__)

ACCOUNT_HEALTH_PREFIX = "libtv:account-health"
ACCOUNT_HEALTH_SEEN_KEY = f"{ACCOUNT_HEALTH_PREFIX}:seen"

STATUS_HEALTHY = "healthy"
STATUS_UNHEALTHY = "unhealthy"
STATUS_UNKNOWN = "unknown"

DEFAULT_PROBE_INTERVAL_SECONDS = 60.0
PROBE_TIMEOUT_SECONDS = 20.0

# The sample must outlive one interval (so a slow cycle is not read as a gap)
# but expire well inside the monitor's staleness window, so a prober that dies
# reads as "missing" rather than staying healthy forever.
SAMPLE_TTL_MULTIPLIER = 3

PROBE_NAME = "getUserInfo"


def account_sample_key(account_key: str) -> str:
    return f"{ACCOUNT_HEALTH_PREFIX}:sample:{account_key}"


def _claim_key(account_key: str) -> str:
    return f"{ACCOUNT_HEALTH_PREFIX}:claim:{account_key}"


def probe_interval_seconds() -> float:
    raw = os.getenv("LIBTV_ACCOUNT_HEALTH_INTERVAL_SECONDS")
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_PROBE_INTERVAL_SECONDS
    return value if value > 0 else DEFAULT_PROBE_INTERVAL_SECONDS


def prober_enabled() -> bool:
    raw = (os.getenv("LIBTV_ACCOUNT_HEALTH_ENABLED") or "").strip().lower()
    # Default ON: a probe that has to be switched on is a probe nobody switches
    # on. It is read-only and one request per account per minute.
    return raw not in {"0", "false", "no", "off"}


def _resolve_credential(value: object) -> str:
    if not isinstance(value, str) or not value:
        return ""
    if value.startswith("os.environ/"):
        return os.getenv(value.removeprefix("os.environ/"), "")
    return value


@dataclass(frozen=True)
class LibTVAccount:
    label: str
    token: str
    webid: str
    account_key: str


def discover_libtv_accounts(llm_router: Any) -> list[LibTVAccount]:
    """Every distinct libtv credential the router can route to.

    Discovery beats configuration here: a new pool member added in
    litellm-config.real.yaml is monitored the moment it can serve traffic, with
    no second place to remember to update. Deduped by account_key, because one
    account commonly backs several model groups (seedance-2.5 and
    seedance-2.0-mini share account-1) and probing it twice buys nothing.

    Credentials that do not resolve are skipped rather than reported unhealthy:
    an unresolvable `os.environ/` reference is a config bug, not a dead vendor
    account, and the two want different people looking at them.
    """
    get_model_list = getattr(llm_router, "get_model_list", None)
    if not callable(get_model_list):
        return []
    try:
        deployments = get_model_list() or []
    except Exception:  # noqa: BLE001  # discovery must never take the startup hook down
        logger.warning("libtv account health: could not read the router model list", exc_info=True)
        return []
    accounts: list[LibTVAccount] = []
    seen: set[str] = set()
    for deployment in deployments:
        params = (deployment or {}).get("litellm_params") or {}
        model_info = (deployment or {}).get("model_info") or {}
        raw_model = str(params.get("model") or "")
        prefix, _, remainder = raw_model.partition("/")
        provider = params.get("custom_llm_provider") or (prefix if remainder else None)
        if provider != "libtv":
            continue
        token = _resolve_credential(params.get("api_key"))
        webid = _resolve_credential(params.get("webid"))
        if not token or not webid:
            continue
        key = derive_account_key(token)
        if key in seen:
            continue
        seen.add(key)
        label = model_info.get("id") if isinstance(model_info.get("id"), str) else None
        accounts.append(
            LibTVAccount(label=label or key, token=token, webid=webid, account_key=key)
        )
    return accounts


def _default_http_client_factory() -> Any:
    import httpx

    return httpx.AsyncClient(timeout=PROBE_TIMEOUT_SECONDS)


class LibTVAccountHealthProber:
    """Probes each libtv account and publishes a sample the monitor can read."""

    def __init__(
        self,
        redis_client: Any = None,
        *,
        router: Any = None,
        http_client_factory: Optional[Callable[[], Any]] = None,
        interval_seconds: float | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._redis = redis_client
        self._router = router
        # Resolved at call time, not bound as a default, so the module-level
        # factory stays substitutable (tests must never dial passport.liblib.art).
        self._client_factory = http_client_factory
        self._interval = interval_seconds if interval_seconds is not None else probe_interval_seconds()
        self._now = now
        self._task: Optional[asyncio.Task] = None

    # ---------- probing ----------

    async def probe(self, account: LibTVAccount) -> dict[str, Any]:
        """One read-only `getUserInfo` call, rendered as a health sample.

        Three outcomes, deliberately distinct:
        healthy   -- libtv answered with a user uuid;
        unhealthy -- libtv answered and rejected us (the 401 "No login" that a
                     dead session token produces);
        unknown   -- the probe itself could not run (transport fault). Calling
                     that "unhealthy" would page someone about our own network.

        The sample never carries the token: it is written to a shared Redis and
        read by another service, so it must stay safe to log.
        """
        checked_at = self._now()
        status = STATUS_UNKNOWN
        reason: str | None = None
        client = None
        try:
            factory = self._client_factory or _default_http_client_factory
            client = factory()
            response = await client.post(
                f"{LIBTV_PASSPORT_BASE}/api/www/user/{PROBE_NAME}",
                json={},
                headers=build_libtv_headers(account.token, account.webid),
                timeout=PROBE_TIMEOUT_SECONDS,
            )
            status, reason = self._read_response(response)
        except LibTVError as exc:
            status, reason = STATUS_UNKNOWN, f"probe error: {exc.status_code}"
        except Exception as exc:  # noqa: BLE001  # a probe fault is "unknown", never a silent success
            status, reason = STATUS_UNKNOWN, f"probe error: {type(exc).__name__}: {exc}"
        finally:
            await self._close(client)
        reason = self._redact(reason, account.token)
        return {
            "status": status,
            "reason": reason,
            "label": account.label,
            "account_key": account.account_key,
            "probe": PROBE_NAME,
            "checked_at": checked_at,
            "received_at": self._now(),
        }

    def _read_response(self, response: Any) -> tuple[str, str | None]:
        status_code = getattr(response, "status_code", None)
        if status_code != 200:
            return STATUS_UNHEALTHY, f"HTTP {status_code}"
        try:
            payload = response.json()
        except Exception:  # noqa: BLE001  # a body we cannot parse tells us nothing about the account
            return STATUS_UNKNOWN, "probe error: unparseable body"
        if not isinstance(payload, dict):
            return STATUS_UNKNOWN, "probe error: unexpected body shape"
        code = payload.get("code")
        if code not in (0, None):
            return STATUS_UNHEALTHY, f"code={code} msg={payload.get('msg')}"
        uuid_value = (payload.get("data") or {}).get("uuid")
        if not uuid_value:
            return STATUS_UNHEALTHY, "getUserInfo returned no uuid"
        return STATUS_HEALTHY, None

    @staticmethod
    def _redact(reason: str | None, token: str) -> str | None:
        if not reason or not token:
            return reason
        return reason.replace(token, "<token>")

    @staticmethod
    async def _close(client: Any) -> None:
        if client is None:
            return
        closer = getattr(client, "aclose", None) or getattr(client, "close", None)
        if not callable(closer):
            return
        try:
            result = closer()
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # noqa: BLE001  # a failed close must not mask the probe result
            logger.warning("libtv account health: probe client close failed", exc_info=True)

    # ---------- publishing ----------

    async def probe_and_publish(self, account: LibTVAccount) -> bool:
        """Probe unless another worker already claimed this account's slot.

        Returns whether this call actually probed. The claim is a plain SET NX
        with the interval as its TTL: several uvicorn workers run this loop, and
        without it each of them would probe every account every cycle.
        """
        if not await self._claim(account):
            return False
        sample = await self.probe(account)
        await self._publish(account, sample)
        return True

    async def _claim(self, account: LibTVAccount) -> bool:
        if self._redis is None:
            return True
        try:
            claimed = await self._redis.set(
                _claim_key(account.account_key), "1", ex=max(1, int(self._interval)), nx=True
            )
        except Exception:  # noqa: BLE001  # if redis cannot arbitrate, probing twice is far better than not at all
            logger.warning("libtv account health: claim failed, probing anyway", exc_info=True)
            return True
        return bool(claimed)

    async def _publish(self, account: LibTVAccount, sample: dict[str, Any]) -> None:
        if self._redis is None:
            return
        ttl = max(1, int(self._interval * SAMPLE_TTL_MULTIPLIER))
        try:
            await self._redis.set(account_sample_key(account.account_key), json.dumps(sample), ex=ttl)
            await self._redis.zadd(ACCOUNT_HEALTH_SEEN_KEY, {account.account_key: sample["received_at"]})
        except Exception:  # noqa: BLE001  # publishing is best effort; the monitor reads the miss as stale
            logger.warning("libtv account health: failed to publish sample", exc_info=True)

    # ---------- loop ----------

    async def run_once(self) -> None:
        for account in discover_libtv_accounts(self._router):
            try:
                await self.probe_and_publish(account)
            except Exception:  # noqa: BLE001  # one bad account must not stop the rest of the sweep
                logger.warning(
                    "libtv account health: probe cycle failed for %s", account.label, exc_info=True
                )

    async def _loop(self) -> None:
        while True:
            await self.run_once()
            await asyncio.sleep(self._interval)

    async def start(self) -> "LibTVAccountHealthProber":
        self._task = asyncio.create_task(self._loop())
        return self

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001  # shutdown path
            pass


async def start_libtv_account_health_prober(llm_router: Any) -> Optional[LibTVAccountHealthProber]:
    """Startup hook. None when disabled or when there is no libtv account to watch."""
    if not prober_enabled():
        return None
    accounts = discover_libtv_accounts(llm_router)
    if not accounts:
        return None
    from litellm.llms.libtv.transfer import get_transfer_redis

    redis_client = get_transfer_redis(
        os.getenv("LIBTV_ACCOUNT_HEALTH_REDIS_URL") or os.getenv("LIBTV_VIDEO_GENERATE_REDIS_URL")
    )
    if redis_client is None:
        # Without Redis the sample has nowhere to go and the monitor would read
        # every account as missing -- worse than not claiming to watch at all.
        logger.warning("libtv account health: no redis configured, prober not started")
        return None
    prober = LibTVAccountHealthProber(redis_client, router=llm_router)
    logger.info(
        "libtv account health: probing %d account(s) every %.0fs", len(accounts), prober._interval
    )
    return await prober.start()


__all__ = [
    "ACCOUNT_HEALTH_PREFIX",
    "ACCOUNT_HEALTH_SEEN_KEY",
    "LibTVAccount",
    "LibTVAccountHealthProber",
    "STATUS_HEALTHY",
    "STATUS_UNHEALTHY",
    "STATUS_UNKNOWN",
    "account_sample_key",
    "discover_libtv_accounts",
    "probe_interval_seconds",
    "prober_enabled",
    "start_libtv_account_health_prober",
]
