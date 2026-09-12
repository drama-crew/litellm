CREATE TABLE "LiteLLM_ModerationMeteringTask" (
  intent_id TEXT PRIMARY KEY,
  binding JSONB NOT NULL,
  reservations JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE "LiteLLM_ModerationMeteringCounter" (
  counter_key TEXT PRIMARY KEY,
  generation TEXT NOT NULL,
  committed_seq INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'ready',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE "LiteLLM_ModerationMeteringPhase" (
  request_id TEXT PRIMARY KEY,
  intent_id TEXT NOT NULL REFERENCES "LiteLLM_ModerationMeteringTask"(intent_id),
  phase TEXT NOT NULL,
  payload JSONB NOT NULL,
  payload_hash TEXT NOT NULL,
  status TEXT NOT NULL,
  lease_token TEXT,
  lease_until TIMESTAMPTZ,
  attempts INTEGER NOT NULL DEFAULT 0,
  available_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  receipt JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(intent_id, phase)
);
CREATE INDEX "moderation_metering_phase_ready" ON "LiteLLM_ModerationMeteringPhase"(status, available_at, lease_until);
