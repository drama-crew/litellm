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
