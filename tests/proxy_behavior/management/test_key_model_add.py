import asyncio

import pytest

from litellm.proxy.utils import hash_token

from .actors import Actor
from .conftest import create_scratch_key

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_key_model_add_preserves_existing_concurrent_grants_and_unrestricted(
    proxy_client,
    prisma,
    scratch,
    world,
):
    caller = world.keys[Actor.PROXY_ADMIN].cleartext
    cleartext_key = await create_scratch_key(
        proxy_client,
        caller,
        scratch.prefix,
        user_id=world.keys[Actor.OWNER].user_id,
        key_alias=scratch.tag("model-add"),
    )
    hashed_token = hash_token(cleartext_key)
    await prisma.db.litellm_verificationtoken.update(
        where={"token": hashed_token},
        data={"models": ["existing-model"]},
    )

    responses = await asyncio.gather(
        proxy_client.post(
            "/key/model/add",
            headers={"Authorization": f"Bearer {caller}"},
            json={"key": cleartext_key, "models": ["model-a"]},
        ),
        proxy_client.post(
            "/key/model/add",
            headers={"Authorization": f"Bearer {caller}"},
            json={"key": cleartext_key, "models": ["model-b"]},
        ),
    )
    assert all(response.status_code == 200 for response in responses), [
        response.text for response in responses
    ]
    finite_row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": hashed_token}
    )
    assert finite_row is not None
    assert set(finite_row.models) == {"existing-model", "model-a", "model-b"}

    await prisma.db.litellm_verificationtoken.update(
        where={"token": hashed_token},
        data={"models": []},
    )
    unrestricted_response = await proxy_client.post(
        "/key/model/add",
        headers={"Authorization": f"Bearer {caller}"},
        json={"key": cleartext_key, "models": ["model-c"]},
    )
    assert unrestricted_response.status_code == 200, unrestricted_response.text
    assert unrestricted_response.json()["models"] == []
    assert unrestricted_response.json()["unrestricted"] is True
    unrestricted_row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": hashed_token}
    )
    assert unrestricted_row is not None
    assert unrestricted_row.models == []


async def test_key_model_add_waits_for_concurrent_project_revoke(
    proxy_client,
    prisma,
    scratch,
    world,
):
    caller = world.keys[Actor.PROXY_ADMIN].cleartext
    project_id = scratch.tag("model-add-project")
    cleartext_key = await create_scratch_key(
        proxy_client,
        caller,
        scratch.prefix,
        user_id=world.keys[Actor.OWNER].user_id,
        key_alias=scratch.tag("project-model-add"),
    )
    hashed_token = hash_token(cleartext_key)
    await prisma.db.litellm_projecttable.create(
        data={
            "project_id": project_id,
            "models": ["existing-model", "new-model"],
            "created_by": world.keys[Actor.PROXY_ADMIN].user_id,
            "updated_by": world.keys[Actor.PROXY_ADMIN].user_id,
        }
    )
    await prisma.db.litellm_verificationtoken.update(
        where={"token": hashed_token},
        data={
            "models": ["existing-model"],
            "project_id": project_id,
        },
    )

    response_task = None
    try:
        async with prisma.writer_db.tx() as revoke_transaction:
            await revoke_transaction.query_raw(
                'UPDATE "LiteLLM_ProjectTable" '
                "SET models = $1::text[], updated_at = NOW() "
                "WHERE project_id = $2",
                ["existing-model"],
                project_id,
            )
            response_task = asyncio.create_task(
                proxy_client.post(
                    "/key/model/add",
                    headers={"Authorization": f"Bearer {caller}"},
                    json={"key": cleartext_key, "models": ["new-model"]},
                )
            )
            waiting_rows = []
            for _ in range(200):
                waiting_rows = await prisma.writer_db.query_raw(
                    "SELECT pid FROM pg_stat_activity "
                    "WHERE datname = current_database() "
                    "AND wait_event_type = 'Lock' "
                    "AND query LIKE '%LiteLLM_ProjectTable%'"
                )
                if waiting_rows:
                    break
                await asyncio.sleep(0.01)
            assert waiting_rows
            assert not response_task.done()

        response = await response_task
        assert response.status_code == 400, response.text
        row = await prisma.db.litellm_verificationtoken.find_unique(
            where={"token": hashed_token}
        )
        assert row is not None
        assert row.models == ["existing-model"]
    finally:
        if response_task is not None and not response_task.done():
            response_task.cancel()
            await asyncio.gather(response_task, return_exceptions=True)
        await prisma.db.litellm_verificationtoken.update(
            where={"token": hashed_token},
            data={"project_id": None},
        )
        await prisma.db.litellm_projecttable.delete(
            where={"project_id": project_id}
        )
