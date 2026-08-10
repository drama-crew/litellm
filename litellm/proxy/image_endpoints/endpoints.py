import asyncio
import traceback
from typing import Any, Dict, List, Literal

import orjson
from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_str_from_messages,
)
from litellm.llms.libtv.common import LibTVError
from litellm.llms.libtv.image_upscale import (
    ImageUpscaleReceipt,
    ProviderRejected,
    ProviderTransportError,
    normalize_image_upscale_receipt,
)
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.route_llm_request import route_request
from litellm.types.llms.openai import ChatCompletionUserMessage

router = APIRouter()


class ImageUpscaleSubmitRequest(BaseModel):
    model: str | None = None
    request_id: str | None = None
    source_url: str | None = None
    input_reference: str | None = None
    source_bytes: int = Field(gt=0)
    source_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")
    source_hard_cap: int | None = Field(default=None, gt=0)
    style: str = "Standard V2"
    scale: Literal[2, 4, 6] = 2
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def validate_source(self):
        if self.source_url is None and self.input_reference is None:
            raise ValueError("one of source_url or input_reference is required")
        if self.source_url is not None and self.input_reference is not None and self.source_url != self.input_reference:
            raise ValueError("source_url and input_reference must refer to the same source")
        source = self.input_reference or self.source_url
        if not isinstance(source, str) or not source.startswith(("http://", "https://")):
            raise ValueError("image upscale source must be an HTTP(S) URL")
        return self


class ImageUpscaleAcceptedResponse(BaseModel):
    receipt: Dict[str, Any]


class ImageUpscaleReceiptResponse(BaseModel):
    request_id: str
    submission_state: Literal["not_submitted", "rejected", "unknown", "submitted"]
    deployment_id: str | None = None
    provider_task_id: str | None = None
    resume_token: str | None = None
    provider_code: str | None = None
    message: str | None = None


class ImageUpscaleErrorMetadata(BaseModel):
    submission_receipt: ImageUpscaleReceiptResponse


class ImageUpscaleErrorBody(BaseModel):
    code: str
    metadata: ImageUpscaleErrorMetadata


class ImageUpscaleErrorResponse(BaseModel):
    error: ImageUpscaleErrorBody


def _image_upscale_response(receipt: dict) -> ORJSONResponse:
    state = receipt.get("submission_state")
    if state == "submitted":
        return ORJSONResponse(status_code=status.HTTP_202_ACCEPTED, content={"receipt": receipt})
    if state == "unknown":
        return ORJSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"error": {"code": "libtv_submission_unknown", "metadata": {"submission_receipt": receipt}}},
        )
    if state == "rejected":
        return ORJSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"error": {"code": "libtv_submission_rejected", "metadata": {"submission_receipt": receipt}}},
        )
    return ORJSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": {"code": "libtv_submission_not_submitted", "metadata": {"submission_receipt": receipt}}},
    )


@router.post(
    "/v1/libtv/image-upscale/submit",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["images"],
    response_model=ImageUpscaleAcceptedResponse,
    responses={
        202: {"model": ImageUpscaleAcceptedResponse, "description": "Provider task submitted"},
        409: {
            "model": ImageUpscaleErrorResponse,
            "description": "Submission outcome is unknown",
        },
        429: {
            "model": ImageUpscaleErrorResponse,
            "description": "Provider explicitly rejected submission",
        },
        503: {
            "model": ImageUpscaleErrorResponse,
            "description": "Submission was not sent",
        },
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": ImageUpscaleSubmitRequest.model_json_schema()}},
        }
    },
)
async def libtv_image_upscale_submit(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_model,
        version,
    )

    data: dict[str, Any] = {}
    try:
        raw = orjson.loads(await request.body())
        data = ImageUpscaleSubmitRequest.model_validate(raw).model_dump(exclude_none=True)
        if data.get("source_url") and not data.get("input_reference"):
            data["input_reference"] = data["source_url"]
        data.pop("source_url", None)
        data.setdefault("prompt", "")
        data["model"] = data.get("model") or user_model or "topaz-image-upscaler"
        data["libtv_image_upscale_submit"] = True
        data["user_api_key_dict"] = user_api_key_dict
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )
        prompt_value = data.get("prompt", "")
        data["messages"] = [{"role": "user", "content": prompt_value}]
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="image_generation"
        )
        messages = data.get("messages")
        if isinstance(messages, list) and messages:
            data["prompt"] = get_str_from_messages(messages)
        data.pop("messages", None)
        llm_call = await route_request(
            data=data,
            route_type="aimage_generation",
            llm_router=llm_router,
            user_model=user_model,
        )
        response = await llm_call
        receipt = (getattr(response, "_hidden_params", {}) or {}).get("submission_receipt")
        normalized = normalize_image_upscale_receipt(
            {"receipt": receipt} if isinstance(receipt, dict) else None,
            str(data.get("request_id") or "unknown"),
            crossed_create_boundary=True,
        )
        if hasattr(proxy_logging_obj, "post_call_success_hook"):
            response = await proxy_logging_obj.post_call_success_hook(
                data=data, user_api_key_dict=user_api_key_dict, response=response
            )
        if hasattr(proxy_logging_obj, "update_request_status"):
            asyncio.create_task(
                proxy_logging_obj.update_request_status(
                    litellm_call_id=data.get("litellm_call_id", ""), status="success"
                )
            )
        return _image_upscale_response(normalized.to_dict())
    except (orjson.JSONDecodeError, ValidationError, ValueError) as error:
        receipt = ImageUpscaleReceipt(
            request_id=str(data.get("request_id") or "unknown"),
            submission_state="not_submitted",
            message=str(error),
        ).to_dict()
        return _image_upscale_response(receipt)
    except Exception as error:  # noqa: BLE001  # receipt classification is handled below
        from litellm.proxy.proxy_server import proxy_logging_obj

        if hasattr(proxy_logging_obj, "post_call_failure_hook"):
            await proxy_logging_obj.post_call_failure_hook(
                user_api_key_dict=user_api_key_dict, original_exception=error, request_data=data
            )
        if isinstance(error, ProviderRejected):
            state = "rejected"
            provider_code = error.provider_code
            message = str(error)
        elif isinstance(error, ProviderTransportError):
            state = "unknown" if error.crossed_create_boundary else "not_submitted"
            provider_code = None
            message = str(error)
        elif isinstance(error, LibTVError) and error.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            state = "rejected"
            provider_code = str(error.status_code)
            message = str(error)
        elif isinstance(error, LibTVError):
            state = "not_submitted"
            provider_code = None
            message = str(error)
        else:
            state = "unknown"
            provider_code = None
            message = "provider submit result is unknown"
        receipt = ImageUpscaleReceipt(
            request_id=str(data.get("request_id") or "unknown"),
            submission_state=state,
            provider_code=provider_code,
            message=message,
        ).to_dict()
        return _image_upscale_response(receipt)


import io

from fastapi import UploadFile


async def uploadfile_to_bytesio(upload: UploadFile) -> io.BytesIO:
    """
    Read a FastAPI UploadFile into a BytesIO and set .name so OpenAI SDK
    infers filename/content-type correctly.
    """
    data = await upload.read()
    buffer = io.BytesIO(data)
    buffer.name = upload.filename
    return buffer


async def batch_to_bytesio(
    uploads: Optional[List[UploadFile]],
) -> Optional[List[io.BytesIO]]:
    """
    Convert a list of UploadFiles to a list of BytesIO buffers, or None.
    """
    if not uploads:
        return None
    return [await uploadfile_to_bytesio(u) for u in uploads]


@router.post(
    "/v1/images/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)
@router.post(
    "/images/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)
@router.post(
    "/openai/deployments/{model:path}/images/generations",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)  # azure compatible endpoint
async def image_generation(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    model: Optional[str] = None,
):
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        user_model,
        version,
    )

    data = {}
    try:
        # Use orjson to parse JSON data, orjson speeds up requests significantly
        body = await request.body()
        data = orjson.loads(body)

        # Include original request and headers in the data
        data = await add_litellm_data_to_request(
            data=data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        data["model"] = (
            model
            or general_settings.get("image_generation_model", None)  # server default
            or user_model  # model name passed via cli args
            or data.get("model", None)  # default passed in http request
        )
        if user_model:
            data["model"] = user_model

        ### MODEL ALIAS MAPPING ###
        # check if model name in model alias map
        # get the actual model name
        if data["model"] in litellm.model_alias_map:
            data["model"] = litellm.model_alias_map[data["model"]]

        ### CALL HOOKS ### - modify incoming data / reject request before calling the model
        prompt_value = data.get("prompt")
        if prompt_value is not None:
            # Reformat the image prompt as a chat message so guardrails can process it.
            user_message: ChatCompletionUserMessage = {
                "role": "user",
                "content": prompt_value,
            }
            data["messages"] = [user_message]
        data = await proxy_logging_obj.pre_call_hook(
            user_api_key_dict=user_api_key_dict, data=data, call_type="image_generation"
        )

        messages = data.get("messages")
        if isinstance(messages, list) and messages:
            data["prompt"] = get_str_from_messages(messages)
        data.pop("messages", None)

        ## ROUTE TO CORRECT ENDPOINT ##
        llm_call = await route_request(
            data=data,
            route_type="aimage_generation",
            llm_router=llm_router,
            user_model=user_model,
        )
        response = await llm_call

        ### ALERTING ###
        asyncio.create_task(
            proxy_logging_obj.update_request_status(litellm_call_id=data.get("litellm_call_id", ""), status="success")
        )

        ### CALL HOOKS ### - modify outgoing data (guardrails, otel, etc.)
        response = await proxy_logging_obj.post_call_success_hook(
            data=data, user_api_key_dict=user_api_key_dict, response=response
        )

        ### RESPONSE HEADERS ###
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        model_id = hidden_params.get("model_id", None) or ""
        cache_key = hidden_params.get("cache_key", None) or ""
        api_base = hidden_params.get("api_base", None) or ""
        response_cost = hidden_params.get("response_cost", None) or ""
        litellm_call_id = hidden_params.get("litellm_call_id", None) or ""

        fastapi_response.headers.update(
            ProxyBaseLLMRequestProcessing.get_custom_headers(
                user_api_key_dict=user_api_key_dict,
                model_id=model_id,
                cache_key=cache_key,
                api_base=api_base,
                version=version,
                response_cost=response_cost,
                model_region=getattr(user_api_key_dict, "allowed_model_region", ""),
                call_id=litellm_call_id,
                request_data=data,
                hidden_params=hidden_params,
            )
        )

        # Call response headers hook (matches base_process_llm_request behavior)
        callback_headers = await proxy_logging_obj.post_call_response_headers_hook(
            data=data,
            user_api_key_dict=user_api_key_dict,
            response=response,
            request_headers=dict(request.headers),
        )
        if callback_headers:
            fastapi_response.headers.update(callback_headers)

        return response
    except Exception as e:
        await proxy_logging_obj.post_call_failure_hook(
            user_api_key_dict=user_api_key_dict, original_exception=e, request_data=data
        )
        verbose_proxy_logger.error(
            "litellm.proxy.proxy_server.image_generation(): Exception occured - {}".format(str(e))
        )
        verbose_proxy_logger.debug(traceback.format_exc())
        if isinstance(e, HTTPException):
            raise ProxyException(
                message=getattr(e, "message", str(e)),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                code=getattr(e, "status_code", status.HTTP_400_BAD_REQUEST),
            )
        else:
            error_msg = f"{str(e)}"
            raise ProxyException(
                message=getattr(e, "message", error_msg),
                type=getattr(e, "type", "None"),
                param=getattr(e, "param", "None"),
                openai_code=getattr(e, "code", None),
                code=getattr(e, "status_code", 500),
            )


@router.post(
    "/v1/images/edits",
    dependencies=[Depends(user_api_key_auth)],
    tags=["images"],
)
@router.post(
    "/images/edits",
    dependencies=[Depends(user_api_key_auth)],
    tags=["images"],
)
@router.post(
    "/openai/deployments/{model:path}/images/edits",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["images"],
)  # azure compatible endpoint
async def image_edit_api(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    image: Optional[List[UploadFile]] = File(None),
    image_array: Optional[List[UploadFile]] = File(None, alias="image[]"),
    mask: Optional[List[UploadFile]] = File(None),
    mask_array: Optional[List[UploadFile]] = File(None, alias="mask[]"),
    model: Optional[str] = None,
):
    """
    Follows the OpenAI Images API spec: https://platform.openai.com/docs/api-reference/images/create

    ```bash
    curl -s -D >(grep -i x-request-id >&2) \
    -o >(jq -r '.data[0].b64_json' | base64 --decode > gift-basket.png) \
    -X POST "http://localhost:4000/v1/images/edits" \
    -H "Authorization: Bearer sk-1234" \
        -F "model=gpt-image-1" \
        -F "image[]=@soap.png" \
        -F 'prompt=Create a studio ghibli image of this'
    ```
    """
    if image is not None and image_array is not None:
        raise HTTPException(status_code=422, detail="Cannot specify both 'image' and 'image[]'")
    if mask is not None and mask_array is not None:
        raise HTTPException(status_code=422, detail="Cannot specify both 'mask' and 'mask[]'")
    if image is None and image_array is not None:
        image = image_array
    if mask is None and mask_array is not None:
        mask = mask_array

    # if image is None:
    #     raise HTTPException(status_code=422, detail="Field required: image")
    # Note: Image is optional for some models (e.g., Bedrock Stability style-transfer)
    # The validation will be done at the model level if image is truly required

    from litellm.proxy.proxy_server import (
        _read_request_body,
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    #########################################################
    # Read request body and convert UploadFiles to BytesIO
    #########################################################
    data = await _read_request_body(request=request)
    image_files = await batch_to_bytesio(image)
    mask_files = await batch_to_bytesio(mask)
    if image_files:
        data["image"] = image_files
    if mask_files:
        data["mask"] = mask_files

    for _field in ("image", "mask"):
        if _field in data and isinstance(data[_field], str):
            raise HTTPException(
                status_code=422,
                detail=f"'{_field}' must be provided as a multipart file upload, not a string.",
            )

    # Ensure prompt exists in data (default to None for models that don't require it)
    if "prompt" not in data:
        data["prompt"] = None

    data["model"] = (
        model
        or general_settings.get("image_generation_model", None)  # server default
        or user_model  # model name passed via cli args
        or data.get("model", None)  # default passed in http request
    )
    #########################################################
    # Process request
    #########################################################

    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aimage_edit",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
