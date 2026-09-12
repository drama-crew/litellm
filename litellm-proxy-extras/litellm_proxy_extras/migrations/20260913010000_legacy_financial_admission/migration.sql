CREATE TABLE IF NOT EXISTS "LiteLLM_LegacyFinancialAdmission" (
    "request_id" TEXT NOT NULL PRIMARY KEY,
    "payload" JSONB NOT NULL,
    "payload_hash" TEXT NOT NULL,
    "created_at" TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
