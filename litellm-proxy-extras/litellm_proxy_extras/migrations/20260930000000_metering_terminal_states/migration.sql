ALTER TABLE "LiteLLM_ModerationMeteringPhase" ADD COLUMN IF NOT EXISTS "failure_reason" TEXT;
ALTER TABLE "LiteLLM_ModerationMeteringTask" ADD COLUMN IF NOT EXISTS "submission_failure" JSONB;
