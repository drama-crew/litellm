import asyncio
import os
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from prisma import Prisma


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv('MODERATION_METERING_POSTGRES_URL'), reason='isolated PostgreSQL required')
async def test_financial_migrations_deploy_before_consumer_and_replay_concurrently(tmp_path: Path) -> None:
    schema = 'deploy_' + uuid4().hex
    url = os.environ['MODERATION_METERING_POSTGRES_URL']
    control = Prisma(datasource={'url': url})
    db = Prisma(datasource={'url': url + '?schema=' + schema})
    migrations = ('20260913000000_moderation_metering', '20260913010000_legacy_financial_admission')
    source = Path('litellm-proxy-extras/litellm_proxy_extras/migrations')
    shutil.copyfile('schema.prisma', tmp_path / 'schema.prisma')
    for migration in migrations:
        shutil.copytree(source / migration, tmp_path / 'migrations' / migration)
    (tmp_path / 'migrations' / 'migration_lock.toml').write_text('provider = "postgresql"\n')

    async def deploy() -> None:
        process = await asyncio.create_subprocess_exec(
            str(Path('.venv/bin/prisma').resolve()), 'migrate', 'deploy',
            '--schema=' + str(tmp_path / 'schema.prisma'),
            env={**os.environ, 'DATABASE_URL': url + '?schema=' + schema},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await asyncio.wait_for(process.communicate(), 60)
        except BaseException:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        assert process.returncode == 0, output.decode()

    try:
        await control.connect()
        await control.execute_raw(f'CREATE SCHEMA "{schema}"')
        await db.connect()
        await deploy()
        await asyncio.gather(deploy(), deploy())
        rows = await db.query_raw('SELECT migration_name FROM "_prisma_migrations" WHERE finished_at IS NOT NULL ORDER BY migration_name')
        assert rows == [{'migration_name': name} for name in migrations]
        for table in ('LiteLLM_LegacyFinancialAdmission', 'LiteLLM_BudgetCutover', 'LiteLLM_BudgetOperation', 'LiteLLM_ModerationMeteringCounter'):
            assert await db.query_raw(f'SELECT COUNT(*)::int AS count FROM "{table}"') == [{'count': 0}]
    finally:
        if db.is_connected():
            await db.disconnect()
        if control.is_connected():
            try:
                await control.execute_raw(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            finally:
                await control.disconnect()
