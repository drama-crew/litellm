ALTER TABLE "LiteLLM_OpenApiLog"
    ADD COLUMN IF NOT EXISTS "call_source" TEXT NOT NULL DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS "project_id" TEXT,
    ADD COLUMN IF NOT EXISTS "generation_id" TEXT,
    ADD COLUMN IF NOT EXISTS "artifact_id" TEXT;
