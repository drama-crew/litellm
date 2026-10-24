-- internal_dispatch = "not a customer open-API call": the platform's own continuation of an already-admitted public
-- task, or a Studio ticket dispatch. Such rows are kept (the failure monitor probes the provider task through them)
-- but are hidden from the customer task log, which filters on internal_dispatch = false.
ALTER TABLE "LiteLLM_OpenApiLog"
    ADD COLUMN IF NOT EXISTS "internal_dispatch" BOOLEAN NOT NULL DEFAULT false;

-- One-time backfill of rows written before the marker existed. New rows are marked at write time, and this migration
-- runs before any new code serves, so the legacy set is closed. Public tasks (task_id mod_video_*) are never marked.
--   * Studio generations arrive with an admission ticket and have no public row; they carry call_source = 'studio'.
--   * An internal continuation is a videos / minimax_h3_test row with a non-public id that follows, within 10 minutes
--     (measured maximum gap in production: 138 s), a public task of the same owner and user.
UPDATE "LiteLLM_OpenApiLog" AS l
SET "internal_dispatch" = true
WHERE l."internal_dispatch" = false
  AND left(coalesce(l."task_id", ''), 10) <> 'mod_video_'
  AND (
    l."call_source" = 'studio'
    OR (
      l."endpoint" IN ('videos', 'minimax_h3_test')
      AND EXISTS (
        SELECT 1 FROM "LiteLLM_OpenApiLog" AS pub
        WHERE pub."owner" = l."owner"
          AND pub."user_id" = l."user_id"
          AND left(coalesce(pub."task_id", ''), 10) = 'mod_video_'
          AND pub."started_at" <= l."started_at"
          AND pub."started_at" >= l."started_at" - interval '10 minutes'
      )
    )
  );
