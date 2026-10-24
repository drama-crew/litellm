-- Marks the platform's own continuation of an already-admitted public task. Such rows are kept for the failure
-- monitor but are hidden from the customer task log. Existing rows default to visible (legacy dispatch rows are
-- excluded by a structural predicate in openapi_log_query.py).
ALTER TABLE "LiteLLM_OpenApiLog"
    ADD COLUMN IF NOT EXISTS "internal_dispatch" BOOLEAN NOT NULL DEFAULT false;
