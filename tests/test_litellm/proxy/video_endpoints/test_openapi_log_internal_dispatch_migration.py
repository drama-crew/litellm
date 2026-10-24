"""internal_dispatch = "not a customer open-API call" (internal moderation continuation or Studio ticket dispatch).

The migration marks the already-written (legacy) rows once; the customer query then needs no runtime heuristic.
"""

import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

MIGRATIONS = Path(__file__).resolve().parents[4] / "litellm-proxy-extras/litellm_proxy_extras/migrations"
MIGRATION = MIGRATIONS / "20261025000000_openapi_log_internal_dispatch/migration.sql"
T0 = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def test_migration_adds_the_column_then_backfills_in_one_file():
    sql = MIGRATION.read_text()
    add = sql.index("ADD COLUMN IF NOT EXISTS")
    assert add < sql.index("UPDATE")
    assert "internal_dispatch" in sql and "interval '10 minutes'" in sql and "'mod_video_'" in sql
    assert "call_source" in sql and "'studio'" in sql


def test_customer_query_has_no_runtime_legacy_heuristic():
    from litellm.proxy.video_endpoints.openapi_log_query import LogFilters, query_conditions

    sql, args = query_conditions("alice", LogFilters(), T0)
    assert "internal_dispatch = false" in sql
    assert "EXISTS" not in sql.upper() and "mod_video_" not in sql and "pub." not in sql
    assert len(args) == 3


def test_all_schema_copies_agree_and_carry_the_marker():
    root = Path(__file__).resolve().parents[4]
    copies = [
        root / "schema.prisma",
        root / "litellm/proxy/schema.prisma",
        root / "litellm-proxy-extras/litellm_proxy_extras/schema.prisma",
    ]
    texts = [c.read_text() for c in copies]
    assert texts[0] == texts[1] == texts[2]
    assert re.search(r"internal_dispatch\s+Boolean\s+@default\(false\)", texts[0])
    assert "submission_failure" in texts[0] and "failure_reason" in texts[0]


@pytest.mark.skipif(not os.getenv("OPENAPI_LOG_TEST_PG_DSN"), reason="throwaway PostgreSQL required")
def test_backfill_marks_exactly_the_structural_set():
    import psycopg

    base = (MIGRATIONS / "20260909094200_openapi_task_history/migration.sql").read_text()
    attribution = (MIGRATIONS / "20260930050000_video_request_attribution/migration.sql").read_text()
    with psycopg.connect(os.environ["OPENAPI_LOG_TEST_PG_DSN"], autocommit=True) as conn:
        schema = "oalog_" + uuid.uuid4().hex
        conn.execute(f'CREATE SCHEMA "{schema}"')
        try:
            conn.execute(f'SET search_path TO "{schema}"')
            conn.execute(base)
            conn.execute(attribution)

            def row(id, owner, endpoint, task, offset_s, *, user="u1", source="open_api"):
                at = T0 + timedelta(seconds=offset_s)
                conn.execute(
                    'INSERT INTO "LiteLLM_OpenApiLog"(id,owner,user_id,endpoint,model,task_id,public_task_id,status,'
                    "started_at,observed_at,call_source) VALUES (%s,%s,%s,%s,'m',%s,%s,'queued',%s,%s,%s)",
                    (id, owner, user, endpoint, task, task, at, at, source),
                )

            # public rows: never touched
            row("pub1", "o1", "minimax_h3", "mod_video_a", 0)
            row("pub-test", "o4", "minimax_h3_test", "mod_video_t", 0)
            row("pub-studio-attr", "o6", "videos", "mod_video_s", 0, source="studio")
            # phantoms of the internal continuation: 138 s (measured prod max), exactly 10 min, failed w/o task id
            row("ph-138s", "o1", "videos", "video_bGl0A", 138)
            row("ph-10min", "o1", "videos", "video_bGl0B", 600)
            row("ph-null", "o1", "videos", None, 5)
            row("ph-test", "o4", "minimax_h3_test", "video_bGl0T", 3)
            # Studio ticket dispatch: no public row at all, marked by call_source
            row("studio1", "o5", "videos", "video_bGl0S", 1, source="studio")
            # genuine customer calls: no prior public row within 10 minutes
            row("real-alone", "o2", "videos", "video_bGl0C", 10)
            row("real-late", "o1", "videos", "video_bGl0D", 601 + 138)  # public row 0 is >10 min before
            row("real-other-user", "o1", "videos", "video_bGl0E", 20, user="u2")
            row("real-before-public", "o3", "videos", "video_bGl0F", -5)
            row("pub3", "o3", "videos", "mod_video_c", 60)
            row("real-h3-endpoint", "o1", "minimax_h3", "video_bGl0G", 30)  # not a continuation endpoint

            migration = MIGRATION.read_text()
            conn.execute(migration)
            conn.execute(migration)  # re-running is harmless

            marked = {r[0] for r in conn.execute('SELECT id FROM "LiteLLM_OpenApiLog" WHERE internal_dispatch')}
            assert marked == {"ph-138s", "ph-10min", "ph-null", "ph-test", "studio1"}
            total = conn.execute('SELECT count(*) FROM "LiteLLM_OpenApiLog"').fetchone()[0]
            assert total == 14
        finally:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
