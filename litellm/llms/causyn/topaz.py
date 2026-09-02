from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, replace
from typing import Literal, Protocol

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.libtv.client import LibTVClient
from litellm.llms.libtv.common import LibTVError
from litellm.llms.libtv.image_upscale import ProviderRejected, ProviderTransportError
from litellm.llms.libtv.transfer import STATUS_TTL_SECONDS
from litellm.llms.libtv.video_generate import VideoGenerateError

TOPAZ_MODEL = "topaz-video-upscaler"
TOPAZ_VENDOR = "topazlabs"
TOPAZ_RESOLUTION = "2K"
TOPAZ_CREATE_PARAMS = {"resolution": TOPAZ_RESOLUTION, "specifiedModel": "prob-4", "slowmo": "1"}
TOPAZ_LEASE_SECONDS = 900
MAX_ATTEMPTS = 4
# Placeholder kept for the case where the provider raises with no text at all.
# `_optional_state_string` rejects an empty string, so writing "" would produce
# a state that cannot be re-loaded and would strand the task.
GENERIC_FAILURE_MESSAGE = "provider_failure"
# The reason is provider-authored and unbounded; the state is a Redis value
# re-serialized on every attempt, so cap it. Long enough for a real upstream
# body, short enough to keep the state small.
MAX_FAILURE_MESSAGE_CHARS = 500
TOPAZ_STATE_PREFIX = "causyn:topaz:"
logger = logging.getLogger(__name__)
TopazPhase = Literal["preparing", "creating", "submitted", "completed", "retryable_failure", "indeterminate", "failed"]
_TOPAZ_STATE_FIELDS = frozenset(
    {
        "target_resolution",
        "attempt_index",
        "confirmed_submission_count",
        "provider_task_id",
        "account_ref",
        "phase",
        "updated_at",
        "lease_until",
        "failure_message",
    }
)
_TOPAZ_PHASES: dict[str, TopazPhase] = {
    "preparing": "preparing",
    "creating": "creating",
    "submitted": "submitted",
    "completed": "completed",
    "retryable_failure": "retryable_failure",
    "indeterminate": "indeterminate",
    "failed": "failed",
}
_TOPAZ_NO_TASK_PHASES = frozenset({"preparing", "creating", "retryable_failure", "failed", "indeterminate"})
_TOPAZ_ACTIVE_PHASES = frozenset({"preparing", "creating"})
_TOPAZ_TERMINAL_FAILURE_PHASES = frozenset({"retryable_failure", "failed"})


_TOPAZ_SOURCE_ERRORS = (LibTVError, VideoGenerateError, httpx.HTTPError, OSError, RuntimeError, ValueError)
_TOPAZ_CLIENT_ERRORS = _TOPAZ_SOURCE_ERRORS + (ProviderRejected, ProviderTransportError)
_TOPAZ_STATE_ADAPTER = TypeAdapter(dict[str, object])


def _decode_state(raw: str | bytes) -> dict[str, object]:
    try:
        return _TOPAZ_STATE_ADAPTER.validate_json(raw)
    except ValidationError as error:
        raise ValueError("invalid Topaz state") from error


def _validate_state_shape(value: dict[str, object]) -> None:
    if frozenset(value) != _TOPAZ_STATE_FIELDS:
        raise ValueError("invalid Topaz state fields")


def _validate_state_counts(value: dict[str, object]) -> tuple[Literal["2k"], int, int]:
    target_resolution = value["target_resolution"]
    if not isinstance(target_resolution, str):
        raise TypeError("invalid Topaz target resolution")
    if target_resolution != "2k":
        raise ValueError("invalid Topaz target resolution")

    attempt_index = value["attempt_index"]
    if isinstance(attempt_index, bool) or not isinstance(attempt_index, int):
        raise TypeError("invalid Topaz attempt index")
    if not 1 <= attempt_index <= MAX_ATTEMPTS:
        raise ValueError("invalid Topaz attempt index")

    confirmed_submission_count = value["confirmed_submission_count"]
    if isinstance(confirmed_submission_count, bool) or not isinstance(confirmed_submission_count, int):
        raise TypeError("invalid Topaz submission count")
    if confirmed_submission_count < 0 or confirmed_submission_count > attempt_index:
        raise ValueError("invalid Topaz submission count")
    return "2k", attempt_index, confirmed_submission_count


def _validate_state_identity(value: dict[str, object]) -> tuple[str, str | None, TopazPhase, str | None]:
    account_ref = value["account_ref"]
    if not isinstance(account_ref, str):
        raise TypeError("invalid Topaz account reference")
    if not account_ref:
        raise ValueError("invalid Topaz account reference")

    raw_phase = value["phase"]
    if not isinstance(raw_phase, str):
        raise TypeError("invalid Topaz phase")
    phase = _TOPAZ_PHASES.get(raw_phase)
    if phase is None:
        raise ValueError("invalid Topaz phase")

    provider_task_id = _optional_state_string(value["provider_task_id"], "provider task id")
    failure_message = _optional_state_string(value["failure_message"], "failure message")
    return account_ref, provider_task_id, phase, failure_message


def _bounded_failure_message(message: str) -> str:
    """Provider text, trimmed and capped; blank falls back to the placeholder."""
    text = (message or "").strip()
    if not text:
        return GENERIC_FAILURE_MESSAGE
    if len(text) <= MAX_FAILURE_MESSAGE_CHARS:
        return text
    return text[: MAX_FAILURE_MESSAGE_CHARS - 1] + "\u2026"


def _optional_state_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"invalid Topaz {field}")
    if not value:
        raise ValueError(f"invalid Topaz {field}")
    return value


def _state_timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("invalid Topaz timestamp")
    if not math.isfinite(value):
        raise ValueError("invalid Topaz timestamp")
    return float(value)


def _validate_state_timestamps(value: dict[str, object]) -> tuple[float, float]:
    return _state_timestamp(value["updated_at"]), _state_timestamp(value["lease_until"])


def _validate_state_phase(
    phase: TopazPhase,
    provider_task_id: str | None,
    failure_message: str | None,
    confirmed_submission_count: int,
    attempt_index: int,
    lease_until: float,
) -> None:
    if provider_task_id is not None and confirmed_submission_count < 1:
        raise ValueError("invalid Topaz provider submission count")
    if provider_task_id is None and phase in _TOPAZ_NO_TASK_PHASES and confirmed_submission_count >= attempt_index:
        raise ValueError("invalid Topaz unconfirmed attempt count")

    if phase in _TOPAZ_ACTIVE_PHASES:
        _validate_active_state(provider_task_id, failure_message, lease_until)
        return
    if phase in {"submitted", "completed"}:
        _validate_submitted_state(phase, provider_task_id, failure_message, lease_until)
        return
    if phase in _TOPAZ_TERMINAL_FAILURE_PHASES:
        _validate_failed_state(failure_message, lease_until)
        return
    if lease_until != 0:
        raise ValueError("invalid Topaz indeterminate lease")


def _validate_active_state(provider_task_id: str | None, failure_message: str | None, lease_until: float) -> None:
    if provider_task_id is not None or failure_message is not None:
        raise ValueError("invalid Topaz active state")
    if lease_until <= 0:
        raise ValueError("invalid Topaz active lease")


def _validate_submitted_state(
    phase: TopazPhase, provider_task_id: str | None, failure_message: str | None, lease_until: float
) -> None:
    if provider_task_id is None or failure_message is not None:
        raise ValueError(f"invalid Topaz {phase} state")
    if lease_until != 0:
        raise ValueError(f"invalid Topaz {phase} lease")


def _validate_failed_state(failure_message: str | None, lease_until: float) -> None:
    if lease_until != 0 or failure_message is None:
        raise ValueError("invalid Topaz failed state")


class TopazClient(Protocol):
    async def aensure_libtv_url(
        self,
        kind: str,
        url: str,
        data: bytes | None,
        default_name: str,
        *,
        allow_cache: bool = True,
    ) -> str: ...

    async def acreate(
        self,
        model_key: str,
        vendor: str,
        task_type: str,
        params: dict[str, object],
        project_name: str,
        *,
        allow_cached_project_retry: bool,
        paid_submission: bool,
    ) -> dict[str, object]: ...

    async def apoll_once(self, task_id: str, task_type: str) -> dict[str, object]: ...

    async def afetch_content(self, url: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class TopazAccount:
    account_ref: str
    token: str
    webid: str


class TopazAccountPool:
    def __init__(self, accounts: tuple[TopazAccount, ...]):
        if not accounts:
            raise ValueError("at least one Topaz account is required")
        self.accounts = accounts

    @classmethod
    def from_environment(cls) -> TopazAccountPool:
        configured = tuple(
            account
            for account in (
                TopazAccount("account-1", os.getenv("LIBTV_TOKEN", "").strip(), os.getenv("LIBTV_WEBID", "").strip()),
                TopazAccount(
                    "account-2", os.getenv("LIBTV_TOKEN_2", "").strip(), os.getenv("LIBTV_WEBID_2", "").strip()
                ),
            )
            if account.token and account.webid
        )
        return cls(configured)

    def for_attempt(self, attempt_index: int) -> TopazAccount:
        return self.accounts[(attempt_index - 1) % len(self.accounts)]

    def by_ref(self, account_ref: str) -> TopazAccount:
        for account in self.accounts:
            if account.account_ref == account_ref:
                return account
        raise ValueError("Topaz account reference is unavailable")


@dataclass(frozen=True, slots=True)
class TopazState:
    target_resolution: Literal["2k"]
    attempt_index: int
    confirmed_submission_count: int
    provider_task_id: str | None
    account_ref: str
    phase: TopazPhase
    updated_at: float
    lease_until: float
    failure_message: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | bytes) -> TopazState:
        value = _decode_state(raw)
        _validate_state_shape(value)
        target_resolution, attempt_index, confirmed_submission_count = _validate_state_counts(value)
        account_ref, provider_task_id, phase, failure_message = _validate_state_identity(value)
        updated_at, lease_until = _validate_state_timestamps(value)
        _validate_state_phase(
            phase,
            provider_task_id,
            failure_message,
            confirmed_submission_count,
            attempt_index,
            lease_until,
        )
        return cls(
            target_resolution=target_resolution,
            attempt_index=attempt_index,
            confirmed_submission_count=confirmed_submission_count,
            provider_task_id=provider_task_id,
            account_ref=account_ref,
            phase=phase,
            updated_at=updated_at,
            lease_until=lease_until,
            failure_message=failure_message,
        )


@dataclass(frozen=True, slots=True)
class TopazAdvance:
    status: Literal["in_progress", "completed", "failed"]
    provider_task_id: str | None = None
    result_url: str | None = None
    attempt_index: int = 0


class TopazIndeterminateError(Exception):
    status_code = 503


class TopazStateCorruptError(TopazIndeterminateError):
    pass


class TopazRedis(Protocol):
    async def get(self, key: str) -> str | bytes | None: ...

    async def set(self, key: str, value: str, *, ex: int, nx: bool = False) -> object: ...

    async def eval(self, script: str, numkeys: int, *args: object) -> object: ...


class TopazStateStore:
    def __init__(self, redis: TopazRedis, *, ttl: int, now: Callable[[], float]):
        self.redis = redis
        self.ttl = ttl
        self.now = now

    @staticmethod
    def key(task_id: str) -> str:
        return f"{TOPAZ_STATE_PREFIX}{task_id}"

    async def load(self, task_id: str) -> TopazState | None:
        value = await self.redis.get(self.key(task_id))
        if value is None:
            return None
        try:
            return TopazState.from_json(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning(
                "causyn Topaz state is corrupt",
                extra={"causyn_task_id": task_id, "state_error": "invalid_state"},
            )
            raise TopazStateCorruptError("Topaz state is corrupt") from None

    async def create(self, task_id: str, state: TopazState) -> bool:
        result = await self.redis.set(self.key(task_id), state.to_json(), ex=self.ttl, nx=True)
        return result in (True, 1, "OK", b"OK")

    async def compare_set(self, task_id: str, expected: TopazState, state: TopazState) -> bool:
        script = (
            "local current=redis.call('GET',KEYS[1]); "
            "if not current then return 0 end; "
            "local old=cjson.decode(current); "
            "if old.phase~=ARGV[1] or old.attempt_index~=tonumber(ARGV[2]) "
            "or old.account_ref~=ARGV[3] or old.updated_at~=tonumber(ARGV[4]) then return 0 end; "
            "redis.call('SET',KEYS[1],ARGV[5],'EX',ARGV[6]); return 1"
        )
        result = await self.redis.eval(
            script,
            1,
            self.key(task_id),
            expected.phase,
            str(expected.attempt_index),
            expected.account_ref,
            repr(expected.updated_at),
            state.to_json(),
            str(self.ttl),
        )
        return result in (1, "1", b"1", True)


async def _reload_or_raise(store: TopazStateStore, task_id: str) -> TopazState:
    state = await store.load(task_id)
    if state is None:
        logger.warning(
            "causyn Topaz state disappeared during transition",
            extra={"causyn_task_id": task_id, "state_error": "missing_state"},
        )
        raise TopazStateCorruptError("Topaz state disappeared during transition")
    return state


def _indeterminate(task_id: str, state: TopazState, message: str) -> TopazIndeterminateError:
    logger.warning(
        "causyn Topaz state is indeterminate",
        extra={"causyn_task_id": task_id, "phase": state.phase, "attempt_index": state.attempt_index},
    )
    return TopazIndeterminateError(message)


async def _after_cas_failure(store: TopazStateStore, task_id: str) -> TopazAdvance:
    state = await _reload_or_raise(store, task_id)
    if state.phase == "indeterminate":
        raise _indeterminate(task_id, state, "Topaz submission state is indeterminate")
    if state.phase == "completed":
        return TopazAdvance(
            "completed",
            provider_task_id=state.provider_task_id,
            attempt_index=state.attempt_index,
        )
    if state.phase == "failed":
        return TopazAdvance("failed", attempt_index=state.attempt_index)
    return TopazAdvance(
        "in_progress",
        provider_task_id=state.provider_task_id,
        attempt_index=state.attempt_index,
    )


async def _mark_indeterminate(store: TopazStateStore, task_id: str, state: TopazState, message: str) -> TopazAdvance:
    indeterminate = replace(state, phase="indeterminate", updated_at=store.now(), lease_until=0)
    if await store.compare_set(task_id, state, indeterminate):
        raise _indeterminate(task_id, indeterminate, message)
    return await _after_cas_failure(store, task_id)


SourceURLFactory = Callable[[str], Awaitable[str]]
SourceValidator = Callable[[str], None]
ClientFactory = Callable[[TopazAccount], TopazClient]
ContentGetter = Callable[[TopazClient, str], Awaitable[bytes]]


def _topaz_params(source_url: str) -> dict[str, object]:
    return {
        "prompt": "",
        "count": 1,
        "textList": [],
        "imageList": [],
        "videoList": [source_url],
        "audioList": [],
        **TOPAZ_CREATE_PARAMS,
    }


class TopazVideoAdapter:
    def __init__(
        self,
        account_pool: TopazAccountPool,
        *,
        client_factory: ClientFactory,
        redis_factory: Callable[[], TopazRedis],
        now: Callable[[], float] = time.time,
        state_ttl: int = STATUS_TTL_SECONDS,
        lease_seconds: int = TOPAZ_LEASE_SECONDS,
        content_getter: ContentGetter | None = None,
    ) -> None:
        self.account_pool = account_pool
        self.client_factory = client_factory
        self.redis_factory = redis_factory
        self.now = now
        self.state_ttl = state_ttl
        self.lease_seconds = lease_seconds
        self.content_getter = content_getter
        self._clients: dict[str, TopazClient] = {}

    @classmethod
    def from_environment(
        cls,
        *,
        redis_factory: Callable[[], TopazRedis],
        now: Callable[[], float] = time.time,
        state_ttl: int = STATUS_TTL_SECONDS,
    ) -> TopazVideoAdapter:
        pool = TopazAccountPool.from_environment()

        def client_factory(account: TopazAccount) -> TopazClient:
            return LibTVClient(token=account.token, webid=account.webid, async_client=AsyncHTTPHandler())

        async def content_getter(client: TopazClient, url: str) -> bytes:
            return await client.afetch_content(url)

        return cls(
            pool,
            client_factory=client_factory,
            redis_factory=redis_factory,
            now=now,
            state_ttl=state_ttl,
            content_getter=content_getter,
        )

    def _store(self) -> TopazStateStore:
        return TopazStateStore(self.redis_factory(), ttl=self.state_ttl, now=self.now)

    def _client_for(self, account_ref: str) -> TopazClient:
        client = self._clients.get(account_ref)
        if client is not None:
            return client
        account = self.account_pool.by_ref(account_ref)
        client = self.client_factory(account)
        self._clients[account_ref] = client
        return client

    def _initial(self, attempt_index: int = 1) -> TopazState:
        now = self.now()
        return TopazState(
            target_resolution="2k",
            attempt_index=attempt_index,
            confirmed_submission_count=0,
            provider_task_id=None,
            account_ref=self.account_pool.for_attempt(attempt_index).account_ref,
            phase="preparing",
            updated_at=now,
            lease_until=now + self.lease_seconds,
        )

    async def _claim_initial(self, store: TopazStateStore, task_id: str) -> tuple[TopazState, bool]:
        loaded = await store.load(task_id)
        if loaded is not None:
            if loaded.phase != "preparing" or loaded.lease_until >= self.now():
                return loaded, False
            claimed = replace(loaded, updated_at=self.now(), lease_until=self.now() + self.lease_seconds)
            if await store.compare_set(task_id, loaded, claimed):
                return claimed, True
            return await _reload_or_raise(store, task_id), False
        state = self._initial()
        if await store.create(task_id, state):
            return state, True
        loaded = await store.load(task_id)
        if loaded is None:
            raise TopazStateCorruptError("Topaz state disappeared after create race")
        return loaded, False

    async def _claim_next(self, store: TopazStateStore, task_id: str, state: TopazState) -> tuple[TopazState, bool]:
        if state.phase != "retryable_failure" or state.attempt_index >= MAX_ATTEMPTS:
            return state, False
        now = self.now()
        next_state = TopazState(
            target_resolution="2k",
            attempt_index=state.attempt_index + 1,
            confirmed_submission_count=state.confirmed_submission_count,
            provider_task_id=None,
            account_ref=self.account_pool.for_attempt(state.attempt_index + 1).account_ref,
            phase="preparing",
            updated_at=now,
            lease_until=now + self.lease_seconds,
        )
        if await store.compare_set(task_id, state, next_state):
            return next_state, True
        return await _reload_or_raise(store, task_id), False

    async def _retryable(self, store: TopazStateStore, task_id: str, state: TopazState, message: str) -> TopazAdvance:
        # `message` is the provider's own text. It used to be accepted as
        # `_message` and dropped on the floor in favour of the placeholder, and
        # nothing else on this path logged it -- the only warnings in this
        # module are about a corrupt/missing state machine. A 2026-09-01
        # production 2K failure therefore ended up with three stacked layers of
        # generic text (this state's "provider_failure", the handler's "causyn
        # 2K processing failed", and the platform's user message) and the cause
        # nowhere at all.
        #
        # Keep it INTERNAL: it goes into the Redis state and the log only.
        # `TopazAdvance` deliberately does not carry it, and the caller-visible
        # strings stay vendor-free -- see 6214f3e1 ("hide 2K provider details in
        # errors"), which is the constraint this must not undo.
        failure_message = _bounded_failure_message(message)
        logger.warning(
            "causyn Topaz attempt failed",
            extra={
                "causyn_task_id": task_id,
                "attempt_index": state.attempt_index,
                "account_ref": state.account_ref,
                "confirmed_submission_count": state.confirmed_submission_count,
                "failure_message": failure_message,
            },
        )
        if state.attempt_index >= MAX_ATTEMPTS:
            failed = replace(
                state, phase="failed", updated_at=self.now(), lease_until=0, failure_message=failure_message
            )
            if await store.compare_set(task_id, state, failed):
                return TopazAdvance("failed", attempt_index=state.attempt_index)
            return await _after_cas_failure(store, task_id)
        retryable = replace(
            state, phase="retryable_failure", updated_at=self.now(), lease_until=0, failure_message=failure_message
        )
        if await store.compare_set(task_id, state, retryable):
            return TopazAdvance("in_progress", attempt_index=state.attempt_index)
        return await _after_cas_failure(store, task_id)

    async def _advance_preparing(
        self,
        store: TopazStateStore,
        task_id: str,
        state: TopazState,
        source_url_factory: SourceURLFactory,
        validate_source: SourceValidator | None,
    ) -> TopazAdvance:
        try:
            account = self.account_pool.by_ref(state.account_ref)
        except ValueError as error:
            return await _mark_indeterminate(store, task_id, state, str(error))
        client = self._client_for(account.account_ref)
        imported = await self._import_source(client, task_id, source_url_factory, validate_source, store, state)
        if isinstance(imported, TopazAdvance):
            return imported
        fence = replace(state, phase="creating", updated_at=self.now(), lease_until=self.now() + self.lease_seconds)
        if not await store.compare_set(task_id, state, fence):
            return await _after_cas_failure(store, task_id)
        created = await self._create_provider_task(client, task_id, fence, imported, store)
        if isinstance(created, TopazAdvance):
            return created
        return await self._submit_created(store, task_id, fence, created)

    async def _import_source(
        self,
        client: TopazClient,
        task_id: str,
        source_url_factory: SourceURLFactory,
        validate_source: SourceValidator | None,
        store: TopazStateStore,
        state: TopazState,
    ) -> str | TopazAdvance:
        try:
            source_url = await source_url_factory(task_id)
            if validate_source is not None:
                validate_source(source_url)
            return await client.aensure_libtv_url("url", source_url, None, "source.mp4", allow_cache=False)
        except _TOPAZ_SOURCE_ERRORS as error:
            return await self._retryable(store, task_id, state, str(error))

    async def _create_provider_task(
        self,
        client: TopazClient,
        task_id: str,
        state: TopazState,
        imported: str,
        store: TopazStateStore,
    ) -> dict[str, object] | TopazAdvance:
        try:
            return await client.acreate(
                TOPAZ_MODEL,
                TOPAZ_VENDOR,
                "video",
                _topaz_params(imported),
                "causyn-2k",
                allow_cached_project_retry=False,
                paid_submission=True,
            )
        except ProviderRejected as error:
            return await self._retryable(store, task_id, state, str(error))
        except ProviderTransportError as error:
            if error.crossed_create_boundary:
                return await _mark_indeterminate(store, task_id, state, str(error))
            return await self._retryable(store, task_id, state, str(error))
        except _TOPAZ_CLIENT_ERRORS as error:
            return await _mark_indeterminate(store, task_id, state, str(error))

    async def _submit_created(
        self, store: TopazStateStore, task_id: str, fence: TopazState, created: dict[str, object]
    ) -> TopazAdvance:
        provider_task_id = created.get("task_id")
        if not isinstance(provider_task_id, str) or not provider_task_id:
            return await _mark_indeterminate(store, task_id, fence, "Topaz create returned no task id")
        submitted = replace(
            fence,
            phase="submitted",
            provider_task_id=provider_task_id,
            confirmed_submission_count=fence.confirmed_submission_count + 1,
            updated_at=self.now(),
            lease_until=0,
        )
        if not await store.compare_set(task_id, fence, submitted):
            return await _after_cas_failure(store, task_id)
        return await self._poll_submitted(store, task_id, submitted)

    async def _poll_submitted(self, store: TopazStateStore, task_id: str, state: TopazState) -> TopazAdvance:
        if state.provider_task_id is None:
            return await self._retryable(store, task_id, state, "Topaz state has no provider task id")
        try:
            client = self._client_for(state.account_ref)
        except ValueError as error:
            return await _mark_indeterminate(store, task_id, state, str(error))
        try:
            progress = await client.apoll_once(state.provider_task_id, "video")
        except _TOPAZ_CLIENT_ERRORS:
            return TopazAdvance("in_progress", attempt_index=state.attempt_index)
        status = progress.get("status")
        if status not in (2, 3):
            return TopazAdvance(
                "in_progress", provider_task_id=state.provider_task_id, attempt_index=state.attempt_index
            )
        if status == 3:
            return await self._retryable(store, task_id, state, str(progress.get("failed_reason") or "Topaz failed"))
        urls = progress.get("urls")
        result_url = urls[0] if isinstance(urls, list) and urls and isinstance(urls[0], str) else None
        if result_url is None:
            return await self._retryable(store, task_id, state, "Topaz completed without a result URL")
        completed = replace(state, phase="completed", updated_at=self.now(), lease_until=0)
        if not await store.compare_set(task_id, state, completed):
            return await _after_cas_failure(store, task_id)
        return TopazAdvance(
            "completed",
            provider_task_id=state.provider_task_id,
            result_url=result_url,
            attempt_index=state.attempt_index,
        )

    async def advance(
        self,
        task_id: str,
        source_url_factory: SourceURLFactory,
        *,
        validate_source: SourceValidator | None = None,
    ) -> TopazAdvance:
        store = self._store()
        try:
            state, claimed = await self._claim_initial(store, task_id)
        except TopazStateCorruptError as error:
            logger.warning(
                "causyn Topaz state cannot be safely advanced",
                extra={"causyn_task_id": task_id, "state_error": type(error).__name__},
            )
            raise error
        if not claimed:
            resumed = await self._resume_existing(store, task_id, state)
            if isinstance(resumed, TopazAdvance):
                return resumed
            state = resumed
        return await self._advance_preparing(store, task_id, state, source_url_factory, validate_source)

    async def _resume_existing(
        self, store: TopazStateStore, task_id: str, state: TopazState
    ) -> TopazState | TopazAdvance:
        if state.phase == "indeterminate":
            raise _indeterminate(task_id, state, "Topaz submission state is indeterminate")
        if state.phase == "failed":
            return TopazAdvance("failed", attempt_index=state.attempt_index)
        if state.phase == "completed":
            return TopazAdvance("completed", provider_task_id=state.provider_task_id, attempt_index=state.attempt_index)
        if state.phase == "retryable_failure":
            next_state, claimed = await self._claim_next(store, task_id, state)
            return next_state if claimed else await _after_cas_failure(store, task_id)
        if state.phase == "submitted":
            return await self._poll_submitted(store, task_id, state)
        if state.phase == "creating" and state.lease_until < self.now():
            return await _mark_indeterminate(store, task_id, state, "stale Topaz create fence is indeterminate")
        return TopazAdvance("in_progress", attempt_index=state.attempt_index)

    async def content(self, task_id: str) -> bytes:
        store = self._store()
        state = await store.load(task_id)
        if state is None or state.phase != "completed" or state.provider_task_id is None:
            raise RuntimeError("Topaz video is not completed")
        try:
            client = self._client_for(state.account_ref)
        except ValueError as error:
            raise _indeterminate(task_id, state, str(error)) from error
        progress = await client.apoll_once(state.provider_task_id, "video")
        urls = progress.get("urls")
        url = urls[0] if isinstance(urls, list) and urls and isinstance(urls[0], str) else None
        if url is None:
            raise RuntimeError("Topaz video completed without a result URL")
        if self.content_getter is None:
            raise RuntimeError("Topaz client does not provide content download")
        return await self.content_getter(client, url)


__all__ = [
    "MAX_ATTEMPTS",
    "TOPAZ_CREATE_PARAMS",
    "TOPAZ_LEASE_SECONDS",
    "TopazAccount",
    "TopazAccountPool",
    "TopazAdvance",
    "TopazIndeterminateError",
    "TopazRedis",
    "TopazVideoAdapter",
]
