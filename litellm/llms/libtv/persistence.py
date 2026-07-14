import hashlib
import logging
import time
from typing import Any, Optional
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
            _tables_ready = True
        except Exception:
            _warn("libtv persistence: failed to create tables", exc_info=True)

    async def cached_upload(self, account_key: str, source_key: str) -> Optional[str]:
        await self.ensure_tables()
        try:
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
        await self.ensure_tables()
        try:
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
        await self.ensure_tables()
        try:
            await self.db.execute_raw(
                'DELETE FROM "LiteLLM_LibTVUploadCache" WHERE account_key=$1 AND source_key=$2',
                account_key,
                source_key,
            )
        except Exception:
            _warn("libtv persistence: delete_upload failed", exc_info=True)


def get_persistence() -> Optional[LibTVPersistence]:
    try:
        from litellm.proxy import proxy_server

        pc = getattr(proxy_server, "prisma_client", None)
    except Exception:
        return None
    if pc is None or getattr(pc, "db", None) is None:
        return None
    return LibTVPersistence(pc.db)
