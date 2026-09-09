-- Additive task history; safe for db-push installations and migration retries.
BEGIN;

-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_OpenApiLog" (
    "id" TEXT NOT NULL,
    "owner" TEXT NOT NULL,
    "user_id" TEXT NOT NULL,
    "endpoint" TEXT NOT NULL,
    "model" TEXT NOT NULL,
    "task_id" TEXT,
    "public_task_id" TEXT,
    "billing_request_id" TEXT,
    "status" TEXT NOT NULL,
    "started_at" TIMESTAMPTZ(3) NOT NULL,
    "observed_at" TIMESTAMPTZ(3) NOT NULL,
    "finished_at" TIMESTAMPTZ(3),
    "elapsed_estimated" BOOLEAN NOT NULL DEFAULT false,
    "historical" BOOLEAN NOT NULL DEFAULT false,
    "input" JSONB,
    "result" JSONB,
    "error" TEXT,

    CONSTRAINT "LiteLLM_OpenApiLog_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE IF NOT EXISTS "LiteLLM_OpenApiLogSpend" (
    "id" TEXT NOT NULL,
    "owner" TEXT NOT NULL,
    "task_id" TEXT NOT NULL,
    "amount" DOUBLE PRECISION NOT NULL,

    CONSTRAINT "LiteLLM_OpenApiLogSpend_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_OpenApiLog_user_id_started_at_id_idx" ON "LiteLLM_OpenApiLog"("user_id", "started_at", "id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_OpenApiLog_user_id_status_started_at_id_idx" ON "LiteLLM_OpenApiLog"("user_id", "status", "started_at", "id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_OpenApiLog_user_id_endpoint_started_at_id_idx" ON "LiteLLM_OpenApiLog"("user_id", "endpoint", "started_at", "id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_OpenApiLog_user_id_model_started_at_id_idx" ON "LiteLLM_OpenApiLog"("user_id", "model", "started_at", "id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_OpenApiLog_user_id_public_task_id_idx" ON "LiteLLM_OpenApiLog"("user_id", "public_task_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_OpenApiLog_owner_task_id_idx" ON "LiteLLM_OpenApiLog"("owner", "task_id");

-- CreateIndex
CREATE INDEX IF NOT EXISTS "LiteLLM_OpenApiLogSpend_owner_task_id_idx" ON "LiteLLM_OpenApiLogSpend"("owner", "task_id");

COMMIT;
