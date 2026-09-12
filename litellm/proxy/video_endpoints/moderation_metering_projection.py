from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from decimal import Decimal
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from litellm.proxy.video_endpoints.openapi_logs import Database

BillingRoute = Literal[
    "avideo_generation",
    "avideo_remix",
    "avideo_edit",
    "avideo_extension",
    "avideo_status",
    "avideo_create_character",
    "video_generation",
    "h3_context_ir",
    "image_upscale",
]


class BillingFacts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    started_at: datetime
    ended_at: datetime
    route: BillingRoute
    pricing: tuple[tuple[str, Decimal], ...] = ()
    duration_seconds: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)
    resolution: str | None = Field(default=None, max_length=24, pattern=r"^[A-Za-z0-9x]+$")
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def immutable_facts(self) -> Self:
        if self.started_at.tzinfo is None or self.ended_at.tzinfo is None or self.ended_at < self.started_at:
            raise ValueError("financial timestamps must be ordered and timezone-aware")
        if len({key for key, _ in self.pricing}) != len(self.pricing) or any(
            not re.fullmatch(r"(?:input|output)_cost_per_[a-z0-9_]{1,64}", key) or not amount.is_finite() or amount < 0
            for key, amount in self.pricing
        ):
            raise ValueError("financial price snapshot is not allowlisted")
        return self


class LegacyMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    libtv_billing_key: str | None = None
    causyn_billing_key: str | None = None
    provider: str | None = None
    scale: int | None = None
    project_id: str | None = None
    artifact_id: str | None = None
    user_id: str | None = None


class FinancialProjection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    request_id: str
    fingerprint: str | None
    user_id: str | None
    team_id: str | None
    organization_id: str | None = None
    end_user_id: str | None = None
    tag_ids: tuple[str, ...] = ()
    global_user_id: str | None = None
    provider: str
    deployment_id: str
    provider_task_id: str
    model: str
    amount: Decimal = Field(ge=0, allow_inf_nan=False)
    facts: BillingFacts
    intent_id: str | None = None
    legacy_metadata: LegacyMetadata | None = None


class FinancialAdmission(BaseModel):
    payload: FinancialProjection
    payload_hash: str


async def freeze_legacy(tx: Database, value: FinancialProjection) -> FinancialProjection:
    payload = value.model_dump_json()
    await tx.execute_raw(
        'INSERT INTO "LiteLLM_LegacyFinancialAdmission" (request_id,payload,payload_hash) VALUES ($1,$2::jsonb,$3) '
        "ON CONFLICT (request_id) DO NOTHING",
        value.request_id,
        payload,
        hashlib.sha256(payload.encode()).hexdigest(),
    )
    rows = TypeAdapter(tuple[FinancialAdmission, ...]).validate_python(
        await tx.query_raw(
            'SELECT payload,payload_hash FROM "LiteLLM_LegacyFinancialAdmission" WHERE request_id=$1', value.request_id
        )
    )
    if len(rows) != 1:
        raise ValueError("legacy financial admission missing")
    stored = rows[0]
    if hashlib.sha256(stored.payload.model_dump_json().encode()).hexdigest() != stored.payload_hash:
        raise ValueError("legacy financial admission corrupt")
    if value.model_copy(update={"global_user_id": stored.payload.global_user_id}) != stored.payload:
        raise ValueError("legacy financial admission replay conflict")
    return stored.payload


async def project(
    tx: Database, value: FinancialProjection, *, protected: frozenset[str], debit_unprotected: bool
) -> None:
    if value.fingerprint:
        rows = TypeAdapter(tuple[dict[str, str | None], ...]).validate_python(
            await tx.query_raw(
                'SELECT user_id,team_id,organization_id FROM "LiteLLM_VerificationToken" WHERE token=$1 FOR UPDATE',
                value.fingerprint,
            )
        )
        if len(rows) != 1 or any(
            expected is not None and rows[0][column] != expected
            for column, expected in (
                ("user_id", value.user_id),
                ("team_id", value.team_id),
                ("organization_id", value.organization_id),
            )
        ):
            raise ValueError("financial projection key identity mismatch")
    if debit_unprotected:
        targets = (
            ("key", "LiteLLM_VerificationToken", "token", value.fingerprint),
            ("user", "LiteLLM_UserTable", "user_id", value.user_id),
            ("team", "LiteLLM_TeamTable", "team_id", value.team_id),
            *(
                (("org", "LiteLLM_OrganizationTable", "organization_id", value.organization_id),)
                if value.organization_id
                else ()
            ),
            *((("user", "LiteLLM_UserTable", "user_id", value.global_user_id),) if value.global_user_id else ()),
            *((("end_user", "LiteLLM_EndUserTable", "user_id", value.end_user_id),) if value.end_user_id else ()),
            *(("tag", "LiteLLM_TagTable", "tag_name", tag) for tag in value.tag_ids),
        )
        for kind, table, column, identity in targets:
            if identity and "spend:" + kind + ":" + identity not in protected:
                count = await tx.execute_raw(
                    f'UPDATE "{table}" SET spend=COALESCE(spend,0)+$1 WHERE {column}=$2', float(value.amount), identity
                )
                if count != 1:
                    raise ValueError("financial projection identity missing")
        if value.user_id and value.team_id and f"spend:team_member:{value.user_id}:{value.team_id}" not in protected:
            count = await tx.execute_raw(
                'UPDATE "LiteLLM_TeamMembership" SET spend=spend+$1,total_spend=total_spend+$1 WHERE user_id=$2 AND team_id=$3',
                float(value.amount),
                value.user_id,
                value.team_id,
            )
            if count not in (0, 1):
                raise ValueError("financial membership identity ambiguous")
    inserted = await tx.execute_raw(
        'INSERT INTO "LiteLLM_SpendLogs" (request_id,call_type,api_key,spend,"startTime","endTime",model,model_id,model_group,custom_llm_provider,"user",team_id,organization_id,end_user,metadata,request_tags,prompt_tokens,completion_tokens,total_tokens) '
        "VALUES ($1,$2,$3,$4,$5::timestamptz,$6::timestamptz,$7,$8,$7,$9,$10,$11,$12,$13,$14::jsonb,$15::jsonb,$16,$17,$18) ON CONFLICT(request_id) DO NOTHING",
        value.request_id,
        value.facts.route,
        value.fingerprint or "",
        float(value.amount),
        value.facts.started_at,
        value.facts.ended_at,
        value.model,
        value.deployment_id,
        value.provider,
        value.user_id or "",
        value.team_id,
        value.organization_id,
        value.end_user_id,
        json.dumps(
            {
                "moderation_intent_id": value.intent_id,
                "provider_task_id": value.provider_task_id,
                "billing_facts": value.facts.model_dump(mode="json"),
                **(value.legacy_metadata.model_dump(exclude_none=True) if value.legacy_metadata else {}),
            }
        ),
        json.dumps(value.tag_ids),
        value.facts.prompt_tokens or 0,
        value.facts.completion_tokens or 0,
        (value.facts.prompt_tokens or 0) + (value.facts.completion_tokens or 0),
    )
    if inserted != 1:
        raise ValueError("financial projection request identity already exists outside its receipt")
    daily = (
        ("LiteLLM_DailyUserSpend", "user_id", value.user_id),
        ("LiteLLM_DailyTeamSpend", "team_id", value.team_id),
        *(
            (("LiteLLM_DailyOrganizationSpend", "organization_id", value.organization_id),)
            if value.organization_id
            else ()
        ),
        *((("LiteLLM_DailyEndUserSpend", "end_user_id", value.end_user_id),) if value.end_user_id else ()),
        *(("LiteLLM_DailyTagSpend", "tag", tag) for tag in value.tag_ids),
    )
    for table, dimension, identity in daily:
        if identity is None:
            continue
        await tx.execute_raw(
            f'INSERT INTO "{table}" (id,{dimension},date,api_key,model,model_group,custom_llm_provider,mcp_namespaced_tool_name,endpoint,spend,api_requests,successful_requests,prompt_tokens,completion_tokens,updated_at) '
            "VALUES ($1,$2,$3,$4,$5,$5,$6,'',$7,$8,1,1,$9,$10,NOW()) "
            f"ON CONFLICT({dimension},date,api_key,model,custom_llm_provider,mcp_namespaced_tool_name,endpoint) DO UPDATE "
            f'SET spend="{table}".spend+EXCLUDED.spend,api_requests="{table}".api_requests+1,successful_requests="{table}".successful_requests+1,prompt_tokens="{table}".prompt_tokens+EXCLUDED.prompt_tokens,completion_tokens="{table}".completion_tokens+EXCLUDED.completion_tokens,updated_at=NOW()',
            uuid4().hex,
            identity,
            value.facts.started_at.date().isoformat(),
            value.fingerprint or "",
            value.model,
            value.provider,
            value.facts.route,
            float(value.amount),
            value.facts.prompt_tokens or 0,
            value.facts.completion_tokens or 0,
        )
