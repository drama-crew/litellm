CREATE TABLE "LiteLLM_ModerationMeteringTask" (
  intent_id TEXT PRIMARY KEY,
  binding JSONB NOT NULL,
  reservations JSONB NOT NULL,
  reservation_id TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE "LiteLLM_ModerationMeteringCounter" (
  counter_key TEXT PRIMARY KEY,
  generation TEXT NOT NULL,
  committed_seq BIGINT NOT NULL DEFAULT 0,
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

ALTER TABLE "LiteLLM_ModerationMeteringCounter" ADD COLUMN target JSONB;
ALTER TABLE "LiteLLM_ModerationMeteringCounter" ADD COLUMN period_start TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP;
ALTER TABLE "LiteLLM_ModerationMeteringCounter" ADD COLUMN pending_operation TEXT;
ALTER TABLE "LiteLLM_ModerationMeteringCounter" ADD COLUMN cutover_id TEXT;
ALTER TABLE "LiteLLM_ModerationMeteringCounter" ADD COLUMN failure_reason TEXT;
CREATE TABLE "LiteLLM_BudgetCutover" (
  receipt_id TEXT PRIMARY KEY,
  payload JSONB NOT NULL,
  payload_hash TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE "LiteLLM_BudgetOperation" (
  operation_id TEXT PRIMARY KEY,
  payload JSONB NOT NULL,
  payload_hash TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  fence INTEGER NOT NULL DEFAULT 0,
  lease_until TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX "budget_operation_pending" ON "LiteLLM_BudgetOperation"(status,lease_until);
CREATE TABLE "LiteLLM_BudgetReservation" (
  reservation_id TEXT NOT NULL,
  counter_key TEXT NOT NULL REFERENCES "LiteLLM_ModerationMeteringCounter"(counter_key),
  generation TEXT NOT NULL,
  remaining NUMERIC(65,30) NOT NULL CHECK (remaining >= 0),
  valid_until TIMESTAMPTZ NOT NULL,
  status TEXT NOT NULL,
  PRIMARY KEY(reservation_id,counter_key)
);
CREATE INDEX "budget_reservation_active" ON "LiteLLM_BudgetReservation"(counter_key,status,valid_until);
CREATE TABLE "LiteLLM_BudgetBirth" (
  counter_key TEXT PRIMARY KEY,
  birth_id TEXT NOT NULL UNIQUE,
  target JSONB NOT NULL,
  initial_spend NUMERIC(65,30) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
