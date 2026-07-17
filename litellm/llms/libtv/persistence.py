import hashlib
import logging
import time
from typing import Any, Optional, TypedDict
from urllib.parse import urlsplit

import httpx

WARN_INTERVAL_SECONDS = 300.0

logger = logging.getLogger(__name__)

_tables_ready = False
_last_warn_ts = 0.0

CREATE_UPLOAD_CACHE_TABLE = """
CREATE TABLE IF NOT EXISTS "LiteLLM_LibTVUploadCache" (
  account_key TEXT NOT NULL,
  source_key TEXT NOT NULL,
  cdn_url TEXT NOT NULL,
  size_bytes BIGINT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_used_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (account_key, source_key)
)
"""

CREATE_PROJECTS_TABLE = """
CREATE TABLE IF NOT EXISTS "LiteLLM_LibTVProjects" (
  account_key TEXT NOT NULL,
  day TEXT NOT NULL,
  project_uuid TEXT NOT NULL,
  team_id TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (account_key, day)
)
"""

CREATE_VIDEO_TASK_USAGE_TABLE = """
CREATE TABLE IF NOT EXISTS "LiteLLM_LibTVVideoTaskUsage" (
  billing_key TEXT NOT NULL PRIMARY KEY,
  duration_seconds DOUBLE PRECISION NOT NULL,
  video_resolution TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

CREATE_BILLED_VIDEO_TASKS_TABLE = """
CREATE TABLE IF NOT EXISTS "LiteLLM_LibTVBilledVideoTasks" (
  billing_key TEXT NOT NULL PRIMARY KEY,
  duration_seconds DOUBLE PRECISION NOT NULL,
  response_cost DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def _warn(message: str, exc_info: bool = False) -> None:
    global _last_warn_ts
    now = time.time()
    if now - _last_warn_ts >= WARN_INTERVAL_SECONDS:
        _last_warn_ts = now
        logger.warning(message, exc_info=exc_info)


def account_key(token: str) -> str:
    return hashlib.sha1(token.encode()).hexdigest()[:16]


def normalize_source_key(kind: str, url: Optional[str], data: Optional[bytes]) -> Optional[str]:
    if kind == "bytes":
        if not data:
            return None
        return f"sha1:{hashlib.sha1(data).hexdigest()}"
    if kind == "url":
        if not url:
            return None
        parsed = urlsplit(url)
        if not parsed.netloc:
            return None
        return f"{parsed.netloc}{parsed.path}"
    return None


class ProjectCacheEntry(TypedDict):
    project_uuid: str
    team_id: Optional[str]


async def url_alive(url: str, timeout: float = 5.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers={"Range": "bytes=0-0"}, follow_redirects=True)
        if resp.status_code in (200, 206):
            return True
        if 400 <= resp.status_code < 500:
            return False
        return True
    except Exception:
        return True


class LibTVPersistence:
    def __init__(self, db: Any):
        self.db = db

    async def ensure_tables(self) -> None:
        global _tables_ready
        if _tables_ready:
            return
        try:
            await self.db.execute_raw(CREATE_UPLOAD_CACHE_TABLE)
            await self.db.execute_raw(CREATE_PROJECTS_TABLE)
            await self.db.execute_raw(CREATE_VIDEO_TASK_USAGE_TABLE)
            await self.db.execute_raw(CREATE_BILLED_VIDEO_TASKS_TABLE)
            _tables_ready = True
        except Exception:
            _warn("libtv persistence: failed to create tables", exc_info=True)

    async def cached_upload(self, account_key: str, source_key: str) -> Optional[str]:
        try:
            await self.ensure_tables()
            rows = await self.db.query_raw(
                'SELECT cdn_url FROM "LiteLLM_LibTVUploadCache" WHERE account_key=$1 AND source_key=$2',
                account_key,
                source_key,
            )
            if not rows:
                return None
            cdn_url = rows[0]["cdn_url"]
            try:
                await self.db.execute_raw(
                    'UPDATE "LiteLLM_LibTVUploadCache" SET last_used_at=now() WHERE account_key=$1 AND source_key=$2',
                    account_key,
                    source_key,
                )
            except Exception:
                _warn("libtv persistence: failed to touch last_used_at", exc_info=True)
            return cdn_url
        except Exception:
            _warn("libtv persistence: cached_upload failed", exc_info=True)
            return None

    async def store_upload(self, account_key: str, source_key: str, cdn_url: str, size_bytes: int) -> None:
        try:
            await self.ensure_tables()
            await self.db.execute_raw(
                'INSERT INTO "LiteLLM_LibTVUploadCache" '
                "(account_key, source_key, cdn_url, size_bytes) VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (account_key, source_key) DO UPDATE SET "
                "cdn_url=EXCLUDED.cdn_url, size_bytes=EXCLUDED.size_bytes, last_used_at=now()",
                account_key,
                source_key,
                cdn_url,
                size_bytes,
            )
        except Exception:
            _warn("libtv persistence: store_upload failed", exc_info=True)

    async def delete_upload(self, account_key: str, source_key: str) -> None:
        try:
            await self.ensure_tables()
            await self.db.execute_raw(
                'DELETE FROM "LiteLLM_LibTVUploadCache" WHERE account_key=$1 AND source_key=$2',
                account_key,
                source_key,
            )
        except Exception:
            _warn("libtv persistence: delete_upload failed", exc_info=True)

    async def cached_project(self, account_key: str, day: str) -> Optional[ProjectCacheEntry]:
        try:
            await self.ensure_tables()
            rows = await self.db.query_raw(
                'SELECT project_uuid, team_id FROM "LiteLLM_LibTVProjects" WHERE account_key=$1 AND day=$2',
                account_key,
                day,
            )
            if not rows:
                return None
            return {"project_uuid": rows[0]["project_uuid"], "team_id": rows[0]["team_id"]}
        except Exception:
            _warn("libtv persistence: cached_project failed", exc_info=True)
            return None

    async def store_project(self, account_key: str, day: str, project_uuid: str, team_id: Optional[str]) -> None:
        try:
            await self.ensure_tables()
            await self.db.execute_raw(
                'INSERT INTO "LiteLLM_LibTVProjects" '
                "(account_key, day, project_uuid, team_id) VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (account_key, day) DO NOTHING",
                account_key,
                day,
                project_uuid,
                team_id,
            )
        except Exception:
            _warn("libtv persistence: store_project failed", exc_info=True)

    async def store_video_task_usage(
        self, billing_key: str, duration_seconds: float, video_resolution: Optional[str]
    ) -> None:
        """Persist the requested duration/resolution of a just-created video task.

        The status poll (the only point that knows generation completed, and thus
        the only correct billing point) receives none of the create request's
        parameters, so it reads this record back to price the task.
        """
        try:
            await self.ensure_tables()
            await self.db.execute_raw(
                'INSERT INTO "LiteLLM_LibTVVideoTaskUsage" '
                "(billing_key, duration_seconds, video_resolution) VALUES ($1, $2, $3) "
                "ON CONFLICT (billing_key) DO NOTHING",
                billing_key,
                duration_seconds,
                video_resolution,
            )
        except Exception:
            _warn("libtv persistence: store_video_task_usage failed", exc_info=True)

    async def get_video_task_usage(self, billing_key: str) -> Optional[dict]:
        """Read back a task's recorded usage: {'duration_seconds', 'video_resolution'} or None."""
        try:
            await self.ensure_tables()
            rows = await self.db.query_raw(
                'SELECT duration_seconds, video_resolution FROM "LiteLLM_LibTVVideoTaskUsage" WHERE billing_key=$1',
                billing_key,
            )
            if not rows:
                return None
            return {
                "duration_seconds": float(rows[0]["duration_seconds"]),
                "video_resolution": rows[0]["video_resolution"],
            }
        except Exception:
            _warn("libtv persistence: get_video_task_usage failed", exc_info=True)
            return None

    async def mark_video_billed(self, billing_key: str, duration_seconds: float, response_cost: float) -> bool:
        """Record a completed video task as billed, once.

        Returns True only the first time this billing_key is recorded (the caller
        should charge response_cost); returns False on every subsequent call for the
        same key (already billed, must not double-charge) and on any persistence
        failure (fail-safe: skip the charge rather than risk billing twice).
        """
        try:
            await self.ensure_tables()
            affected = await self.db.execute_raw(
                'INSERT INTO "LiteLLM_LibTVBilledVideoTasks" '
                "(billing_key, duration_seconds, response_cost) VALUES ($1, $2, $3) "
                "ON CONFLICT (billing_key) DO NOTHING",
                billing_key,
                duration_seconds,
                response_cost,
            )
            return bool(affected)
        except Exception:
            _warn("libtv persistence: mark_video_billed failed", exc_info=True)
            return False

    async def invalidate_project(self, account_key: str, day: str) -> None:
        try:
            await self.ensure_tables()
            await self.db.execute_raw(
                'DELETE FROM "LiteLLM_LibTVProjects" WHERE account_key=$1 AND day=$2',
                account_key,
                day,
            )
        except Exception:
            _warn("libtv persistence: invalidate_project failed", exc_info=True)


def get_persistence() -> Optional[LibTVPersistence]:
    try:
        from litellm.proxy import proxy_server

        pc = getattr(proxy_server, "prisma_client", None)
    except Exception:
        return None
    if pc is None or getattr(pc, "db", None) is None:
        return None
    return LibTVPersistence(pc.db)
