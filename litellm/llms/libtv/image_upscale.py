import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import secrets
from dataclasses import asdict, dataclass
from typing import Awaitable, Literal, Mapping, Protocol, Sequence

from redis.exceptions import RedisError

from litellm.llms.libtv.receipts import LibTVReceiptStore, ReceiptClaim, StoredReceipt, request_fingerprint

SubmissionState = Literal["not_submitted", "rejected", "unknown", "submitted"]
TaskState = Literal["active", "succeeded", "failed", "resolved"]


def make_resume_token(
    deployment_id: str,
    provider_task_id: str,
    secret: str,
    *,
    team_id: str | None = None,
    model: str | None = None,
    request_id: str | None = None,
    fingerprint: str | None = None,
) -> str:
    if not secret:
        raise ValueError("resume token secret is required")
    payload = json.dumps(
        {
            "deployment_id": deployment_id,
            "provider_task_id": provider_task_id,
            "team_id": team_id,
            "model": model,
            "request_id": request_id,
            "fingerprint": fingerprint,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    signature = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    return f"v2.{encoded}.{signature}"


def verify_resume_token(
    token: str,
    secret: str,
    *,
    deployment_id: str | None = None,
    provider_task_id: str | None = None,
    team_id: str | None = None,
    model: str | None = None,
    request_id: str | None = None,
    fingerprint: str | None = None,
) -> bool:
    if not secret:
        return False
    try:
        version, encoded, signature = token.split(".", 2)
        if version != "v2":
            return False
        expected = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return False
        payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    except (TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError, binascii.Error):
        return False
    return (
        isinstance(payload, dict)
        and (deployment_id is None or payload.get("deployment_id") == deployment_id)
        and (provider_task_id is None or payload.get("provider_task_id") == provider_task_id)
        and (team_id is None or payload.get("team_id") == team_id)
        and (model is None or payload.get("model") == model)
        and (request_id is None or payload.get("request_id") == request_id)
        and (fingerprint is None or payload.get("fingerprint") == fingerprint)
    )


@dataclass(frozen=True, slots=True)
class ImageUpscaleReceipt:
    request_id: str
    submission_state: SubmissionState
    deployment_id: str | None = None
    provider_task_id: str | None = None
    resume_token: str | None = None
    provider_code: str | None = None
    message: str | None = None
    response_cost: float | None = None
    billing_event_id: str | None = None
    task_state: TaskState = "active"

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


class IdempotencyFingerprintMismatch(Exception):
    status_code = 409
    code = "idempotency_fingerprint_mismatch"

    def __init__(self, message: str, *, receipt: ImageUpscaleReceipt | None = None):
        super().__init__(message)
        self.receipt = receipt


def normalize_image_upscale_receipt(
    payload: Mapping[str, object] | None,
    request_id: str,
    *,
    crossed_create_boundary: bool = False,
) -> ImageUpscaleReceipt:
    candidate = payload.get("receipt") if isinstance(payload, Mapping) else None
    if not isinstance(candidate, Mapping):
        metadata = payload.get("error") if isinstance(payload, Mapping) else None
        candidate = metadata.get("metadata", {}).get("submission_receipt") if isinstance(metadata, Mapping) else None
    if not isinstance(candidate, Mapping):
        return ImageUpscaleReceipt(
            request_id=request_id,
            submission_state="unknown" if crossed_create_boundary else "not_submitted",
            message="provider response did not include a submission receipt",
        )
    state = candidate.get("submission_state")
    if state not in {"not_submitted", "rejected", "unknown", "submitted"}:
        state = "unknown" if crossed_create_boundary else "not_submitted"
    task_id = candidate.get("provider_task_id")
    deployment_id = candidate.get("deployment_id")
    resume_token = candidate.get("resume_token")
    task_state = candidate.get("task_state") or "active"
    if candidate.get("request_id") not in (None, request_id):
        return ImageUpscaleReceipt(request_id=request_id, submission_state="unknown")
    if state == "submitted" and not all(
        isinstance(value, str) and value for value in (task_id, deployment_id, resume_token)
    ):
        state = "unknown"
    if state != "submitted" and any(value is not None for value in (task_id, resume_token)):
        return ImageUpscaleReceipt(request_id=request_id, submission_state="unknown")
    if task_state not in {"active", "succeeded", "failed", "resolved"}:
        return ImageUpscaleReceipt(request_id=request_id, submission_state="unknown")
    return ImageUpscaleReceipt(
        request_id=request_id,
        submission_state=state,
        deployment_id=deployment_id if isinstance(deployment_id, str) else None,
        provider_task_id=task_id if isinstance(task_id, str) else None,
        resume_token=resume_token if isinstance(resume_token, str) else None,
        provider_code=candidate.get("provider_code") if isinstance(candidate.get("provider_code"), str) else None,
        message=candidate.get("message") if isinstance(candidate.get("message"), str) else None,
        task_state=task_state,
    )


class ProviderTransportError(Exception):
    def __init__(self, message: str = "provider transport error", *, crossed_create_boundary: bool):
        super().__init__(message)
        self.crossed_create_boundary = crossed_create_boundary


class ProviderRejected(Exception):
    def __init__(self, message: str, provider_code: str | None = None):
        super().__init__(message)
        self.provider_code = provider_code


class ImageUpscaleProvider(Protocol):
    def create(self, payload: Mapping[str, object]) -> Awaitable[Mapping[str, object]]: ...


def _response_cost(payload: Mapping[str, object]) -> float | None:
    value = payload.get("response_cost")
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return float(value)
    return None


_STYLES = frozenset(
    {
        "Standard V2",
        "Low Resolution V2",
        "CGI",
        "High Fidelity V2",
        "Text Refine",
    }
)
_SCALES = frozenset({2, 4, 6})


class TopazImageUpscaleBuilder:
    def build(
        self,
        *,
        source_url: str | None = None,
        source_urls: Sequence[str] | None = None,
        style: str = "Standard V2",
        scale: int = 2,
    ) -> dict[str, object]:
        sources = tuple(source_urls or ())
        if source_url is not None:
            sources = (*sources, source_url)
        if len(sources) != 1:
            raise ValueError("Topaz image upscale requires exactly one source image")
        if style not in _STYLES:
            raise ValueError(f"unsupported Topaz image upscale style: {style}")
        if scale not in _SCALES:
            raise ValueError(f"unsupported Topaz image upscale scale: {scale}")
        return {"style": style, "scale": scale, "imageList": [sources[0]]}


class ImageUpscaleSubmitter:
    def __init__(
        self,
        *deployments: tuple[str, ImageUpscaleProvider] | tuple[str, ImageUpscaleProvider, str],
        resume_secret: str | None = None,
        receipt_store: LibTVReceiptStore | None = None,
        team_id: str | None = None,
        api_key: str | None = None,
        user_id: str | None = None,
        organization_id: str | None = None,
        model: str = "topaz-image-upscaler",
    ):
        if not deployments:
            raise ValueError("at least one image upscale deployment is required")
        self._deployments = tuple(deployments)
        self._resume_secret = (
            resume_secret or os.getenv("LIBTV_IMAGE_UPSCALE_RESUME_SECRET") or secrets.token_urlsafe(32)
        )
        self._receipt_store = receipt_store
        self._team_id = team_id
        self._api_key = api_key
        self._user_id = user_id
        self._organization_id = organization_id
        self._model = model

    @staticmethod
    def _from_stored(receipt: StoredReceipt | None, request_id: str, message: str | None = None) -> ImageUpscaleReceipt:
        if receipt is None:
            return ImageUpscaleReceipt(request_id=request_id, submission_state="unknown", message=message)
        state = receipt.submission_state
        public_state = "unknown" if state == "submitting" else state
        return ImageUpscaleReceipt(
            request_id=request_id,
            submission_state=public_state,
            deployment_id=receipt.deployment_id,
            provider_task_id=receipt.provider_task_id,
            resume_token=receipt.resume_token,
            provider_code=receipt.provider_code,
            message=message or receipt.message,
            response_cost=receipt.response_cost,
            billing_event_id=receipt.billing_event_id,
            task_state=receipt.task_state,
        )

    async def _claim(
        self,
        team_id: str,
        request_id: str,
        fingerprint: str,
        deployment_id: str,
        response_cost: float | None = None,
    ) -> tuple[ReceiptClaim, ImageUpscaleReceipt | None]:
        if self._receipt_store is None:
            raise RuntimeError("receipt store is not configured")
        if response_cost is None:
            claim_kwargs = self._identity_claim_kwargs()
            claim = await self._receipt_store.claim(
                team_id, self._model, request_id, fingerprint, deployment_id, **claim_kwargs
            )
        else:
            claim_kwargs = self._identity_claim_kwargs()
            claim = await self._receipt_store.claim(
                team_id,
                self._model,
                request_id,
                fingerprint,
                deployment_id,
                response_cost=response_cost,
                **claim_kwargs,
            )
        if claim.outcome == "mismatch":
            raise IdempotencyFingerprintMismatch(
                "request_id was already used with a different fingerprint",
                receipt=ImageUpscaleReceipt(
                    request_id=request_id,
                    submission_state="unknown",
                    deployment_id=deployment_id,
                    message="request_id was already used with a different fingerprint",
                ),
            )
        if claim.outcome == "missing":
            return claim, self._from_stored(None, request_id, "receipt missing after pending claim")
        if claim.outcome in {"existing", "rejected", "not_submitted"}:
            if claim.receipt is not None and claim.receipt.submission_state == "submitting":
                resolved = await self._receipt_store.wait(claim)
                return claim, self._from_stored(resolved or claim.receipt, request_id)
            return claim, self._from_stored(claim.receipt, request_id)
        return claim, None

    def _identity_claim_kwargs(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "api_key": self._api_key,
                "user_id": self._user_id,
                "organization_id": self._organization_id,
            }.items()
            if isinstance(value, str) and value
        }

    async def _submit_deployment(
        self,
        request_id: str,
        deployment_id: str,
        provider: ImageUpscaleProvider,
        resume_secret: str,
        claim: ReceiptClaim | None,
        payload: Mapping[str, object],
        team_id: str,
        fingerprint: str,
    ) -> tuple[ImageUpscaleReceipt | None, ProviderRejected | None]:
        try:
            response = await provider.create(payload)
        except ProviderRejected as error:
            if self._receipt_store is not None and claim is not None and claim.receipt is not None:
                await self._receipt_store.transition(
                    claim.receipt,
                    claim.receipt_key,
                    "rejected",
                    provider_code=error.provider_code,
                    message=str(error),
                )
            return None, error
        except ProviderTransportError as error:
            if self._receipt_store is not None and claim is not None and claim.receipt is not None:
                state = "unknown" if error.crossed_create_boundary else "not_submitted"
                stored = await self._receipt_store.transition(
                    claim.receipt, claim.receipt_key, state, message=str(error)
                )
                if error.crossed_create_boundary:
                    return self._from_stored(stored, request_id), None
                return self._from_stored(stored, request_id), None
            if error.crossed_create_boundary:
                return (
                    ImageUpscaleReceipt(
                        request_id=request_id,
                        submission_state="unknown",
                        deployment_id=deployment_id,
                        message=str(error),
                    ),
                    None,
                )
            return (
                ImageUpscaleReceipt(
                    request_id=request_id,
                    submission_state="not_submitted",
                    deployment_id=deployment_id,
                    message=str(error),
                ),
                None,
            )
        task_id = response.get("task_id") or response.get("taskId")
        if not isinstance(task_id, str) or not task_id:
            if self._receipt_store is not None and claim is not None and claim.receipt is not None:
                stored = await self._receipt_store.transition(
                    claim.receipt,
                    claim.receipt_key,
                    "unknown",
                    message="provider create returned no task id",
                )
                return self._from_stored(stored, request_id), None
            return (
                ImageUpscaleReceipt(
                    request_id=request_id,
                    submission_state="unknown",
                    deployment_id=deployment_id,
                    message="provider create returned no task id",
                ),
                None,
            )
        receipt = ImageUpscaleReceipt(
            request_id=request_id,
            submission_state="submitted",
            deployment_id=deployment_id,
            provider_task_id=task_id,
            resume_token=make_resume_token(
                deployment_id,
                task_id,
                resume_secret,
                team_id=team_id,
                model=self._model,
                request_id=request_id,
                fingerprint=fingerprint,
            ),
        )
        if self._receipt_store is None or claim is None or claim.receipt is None:
            return receipt, None
        stored = await self._receipt_store.transition(
            claim.receipt,
            claim.receipt_key,
            "submitted",
            provider_task_id=task_id,
            resume_token=receipt.resume_token,
        )
        return self._from_stored(stored, request_id), None

    async def submit(self, payload: Mapping[str, object]) -> ImageUpscaleReceipt:
        if self._receipt_store is None and os.getenv("LIBTV_RECEIPTS_REDIS_URL"):
            from litellm.llms.libtv.persistence import get_receipt_store

            self._receipt_store = get_receipt_store()
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("image upscale request_id is required")
        team_id = self._team_id or (payload.get("team_id") if isinstance(payload.get("team_id"), str) else "default")
        response_cost = _response_cost(payload)
        if response_cost is None:
            return ImageUpscaleReceipt(
                request_id=request_id,
                submission_state="not_submitted",
                message="paid image upscale requires a finite positive response cost",
            )
        api_key = self._api_key or (payload.get("api_key") if isinstance(payload.get("api_key"), str) else None)
        user_id = self._user_id or (payload.get("user_id") if isinstance(payload.get("user_id"), str) else None)
        organization_id = self._organization_id or (
            payload.get("organization_id") if isinstance(payload.get("organization_id"), str) else None
        )
        if not all((team_id, api_key, user_id, organization_id)):
            return ImageUpscaleReceipt(
                request_id=request_id,
                submission_state="not_submitted",
                message="paid image upscale requires complete billing identity",
            )
        TopazImageUpscaleBuilder().build(
            source_url=payload.get("source_url") if isinstance(payload.get("source_url"), str) else None,
            source_urls=payload.get("source_urls") if isinstance(payload.get("source_urls"), Sequence) else None,
            style=payload.get("style", "Standard V2") if isinstance(payload.get("style", "Standard V2"), str) else "",
            scale=payload.get("scale", 2) if isinstance(payload.get("scale", 2), int) else -1,
        )
        last_rejection: ProviderRejected | None = None
        last_not_submitted: ImageUpscaleReceipt | None = None
        fingerprint = request_fingerprint(payload, self._model)
        for deployment in self._deployments:
            deployment_id, provider = deployment[:2]
            resume_secret = deployment[2] if len(deployment) > 2 else self._resume_secret
            claim = None
            if self._receipt_store is not None:
                try:
                    claim, existing = await self._claim(
                        team_id,
                        request_id,
                        fingerprint,
                        deployment_id,
                        response_cost,
                    )
                except RedisError:
                    return ImageUpscaleReceipt(
                        request_id=request_id,
                        submission_state="not_submitted",
                        message="receipt store unavailable",
                    )
                if existing is not None:
                    if existing.submission_state in {"unknown", "submitted"}:
                        return existing
                    if existing.submission_state == "rejected":
                        last_rejection = ProviderRejected(
                            existing.message or "provider rejected", existing.provider_code
                        )
                        continue
                    if existing.submission_state == "not_submitted":
                        continue
            result, rejection = await self._submit_deployment(
                request_id, deployment_id, provider, resume_secret, claim, payload, team_id, fingerprint
            )
            if result is not None:
                if result.submission_state == "not_submitted":
                    last_not_submitted = result
                    continue
                return result
            if rejection is not None:
                last_rejection = rejection
                continue
        if last_not_submitted is not None:
            return last_not_submitted
        if last_rejection is not None:
            return ImageUpscaleReceipt(
                request_id=request_id,
                submission_state="rejected",
                provider_code=last_rejection.provider_code,
                message=str(last_rejection),
            )
        return ImageUpscaleReceipt(request_id=request_id, submission_state="not_submitted")
