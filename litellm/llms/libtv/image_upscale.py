from dataclasses import asdict, dataclass
from typing import Awaitable, Literal, Mapping, Protocol, Sequence

SubmissionState = Literal["not_submitted", "rejected", "unknown", "submitted"]


@dataclass(frozen=True, slots=True)
class ImageUpscaleReceipt:
    request_id: str
    submission_state: SubmissionState
    deployment_id: str | None = None
    provider_task_id: str | None = None
    resume_token: str | None = None
    provider_code: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


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
    if state == "submitted" and not all(
        isinstance(value, str) and value for value in (task_id, deployment_id, resume_token)
    ):
        state = "unknown"
    return ImageUpscaleReceipt(
        request_id=request_id,
        submission_state=state,
        deployment_id=deployment_id if isinstance(deployment_id, str) else None,
        provider_task_id=task_id if isinstance(task_id, str) else None,
        resume_token=resume_token if isinstance(resume_token, str) else None,
        provider_code=candidate.get("provider_code") if isinstance(candidate.get("provider_code"), str) else None,
        message=candidate.get("message") if isinstance(candidate.get("message"), str) else None,
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
    def __init__(self, *deployments: tuple[str, ImageUpscaleProvider]):
        if not deployments:
            raise ValueError("at least one image upscale deployment is required")
        self._deployments = tuple(deployments)

    async def submit(self, payload: Mapping[str, object]) -> ImageUpscaleReceipt:
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("image upscale request_id is required")
        TopazImageUpscaleBuilder().build(
            source_url=payload.get("source_url") if isinstance(payload.get("source_url"), str) else None,
            source_urls=payload.get("source_urls") if isinstance(payload.get("source_urls"), Sequence) else None,
            style=payload.get("style", "Standard V2") if isinstance(payload.get("style", "Standard V2"), str) else "",
            scale=payload.get("scale", 2) if isinstance(payload.get("scale", 2), int) else -1,
        )
        last_rejection: ProviderRejected | None = None
        for deployment_id, provider in self._deployments:
            try:
                response = await provider.create(payload)
            except ProviderRejected as error:
                last_rejection = error
                continue
            except ProviderTransportError as error:
                if error.crossed_create_boundary:
                    return ImageUpscaleReceipt(
                        request_id=request_id,
                        submission_state="unknown",
                        deployment_id=deployment_id,
                        message=str(error),
                    )
                continue
            task_id = response.get("task_id") or response.get("taskId")
            if not isinstance(task_id, str) or not task_id:
                return ImageUpscaleReceipt(
                    request_id=request_id,
                    submission_state="unknown",
                    deployment_id=deployment_id,
                    message="provider create returned no task id",
                )
            return ImageUpscaleReceipt(
                request_id=request_id,
                submission_state="submitted",
                deployment_id=deployment_id,
                provider_task_id=task_id,
                resume_token=f"v1:{deployment_id}:{task_id}",
            )
        if last_rejection is not None:
            return ImageUpscaleReceipt(
                request_id=request_id,
                submission_state="rejected",
                provider_code=last_rejection.provider_code,
                message=str(last_rejection),
            )
        return ImageUpscaleReceipt(request_id=request_id, submission_state="not_submitted")
