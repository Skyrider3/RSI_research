-- DriftLab experiment store (SQLite, WAL mode). schema_version = 1
--
-- Separation of concerns:
--   generations : one row per PHYSICAL model output (content-addressed by gen_key)
--   cells       : one row per LOGICAL generation event (seed, split, decoding, slot, draw, item) -> gen_key
--                 Many cells may point to the same gen_key (cache hit). physical=1 marks cells whose
--                 generation was forced to be fresh (nonce) or is an independent sample.
--   scores      : extractor output for a generation; scores are computed for EVERY registered extractor.
-- Derived tables (decisions, reference_snapshots, analysis_results) are rebuilt by `driftlab analyze`.

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS runs (
  run_id          TEXT PRIMARY KEY,
  config_json     TEXT NOT NULL,
  config_hash     TEXT NOT NULL,
  plan_hash       TEXT,
  provenance_json TEXT,
  engine_fp       TEXT,
  synthetic       INTEGER NOT NULL DEFAULT 0,
  created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
  split       TEXT NOT NULL,
  idx         INTEGER NOT NULL,
  question    TEXT NOT NULL,
  answer_text TEXT NOT NULL,
  gold        TEXT NOT NULL,          -- canonical numeric string, e.g. "18"
  PRIMARY KEY (split, idx)
);

CREATE TABLE IF NOT EXISTS prompts (
  prompt_hash TEXT PRIMARY KEY,       -- sha256 of text
  text        TEXT NOT NULL
);

-- Slot k = prompt p_k of the trajectory: slot 0 = initial prompt, slot t = candidate proposed at round t.
CREATE TABLE IF NOT EXISTS slots (
  run_id        TEXT NOT NULL,
  seed          INTEGER NOT NULL,
  slot          INTEGER NOT NULL,
  prompt_hash   TEXT NOT NULL REFERENCES prompts(prompt_hash),
  created_round INTEGER NOT NULL,
  parent_slot   INTEGER,              -- trajectory incumbent the candidate was proposed from
  origin        TEXT NOT NULL CHECK (origin IN ('initial', 'proposer', 'fallback')),
  PRIMARY KEY (run_id, seed, slot)
);

CREATE TABLE IF NOT EXISTS proposals (
  run_id             TEXT NOT NULL,
  seed               INTEGER NOT NULL,
  round              INTEGER NOT NULL,
  attempt            INTEGER NOT NULL,
  meta_prompt_hash   TEXT NOT NULL,
  error_item_idxs    TEXT NOT NULL,   -- JSON list of DEV (train-split) item indices shown to the proposer
  gen_key            TEXT,
  parsed_prompt_hash TEXT,
  valid              INTEGER NOT NULL,
  violations         TEXT,            -- JSON list of validation failures
  PRIMARY KEY (run_id, seed, round, attempt)
);

CREATE TABLE IF NOT EXISTS trajectory_rounds (
  run_id         TEXT NOT NULL,
  seed           INTEGER NOT NULL,
  round          INTEGER NOT NULL,
  incumbent_slot INTEGER NOT NULL,    -- trajectory incumbent when the candidate was proposed
  candidate_slot INTEGER NOT NULL,
  inc_dev_acc    REAL,
  cand_dev_acc   REAL,
  advanced       INTEGER NOT NULL,
  n_attempts     INTEGER NOT NULL,
  is_fallback    INTEGER NOT NULL,
  completed_at   TEXT NOT NULL,
  PRIMARY KEY (run_id, seed, round)
);

CREATE TABLE IF NOT EXISTS generations (
  gen_key             TEXT PRIMARY KEY,
  engine_fp           TEXT NOT NULL,
  model_id            TEXT NOT NULL,
  model_revision      TEXT NOT NULL,
  rendered_sha        TEXT NOT NULL,
  system_hash         TEXT NOT NULL,
  user_hash           TEXT NOT NULL,
  decoding_json       TEXT NOT NULL,
  seed                INTEGER,
  nonce               TEXT,
  response            TEXT NOT NULL,
  finish_reason       TEXT,
  n_prompt_tokens     INTEGER,
  n_completion_tokens INTEGER,
  latency_ms          REAL,
  batch_id            TEXT,
  created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS gen_sys_user ON generations (system_hash, user_hash);

CREATE TABLE IF NOT EXISTS cells (
  run_id      TEXT NOT NULL,
  seed        INTEGER NOT NULL,
  split       TEXT NOT NULL,          -- 'train' (dev) | 'test' (eval)
  decoding_id TEXT NOT NULL,          -- 'greedy' | 't02' | 'proposer'
  slot        INTEGER NOT NULL,
  draw_kind   TEXT NOT NULL CHECK (draw_kind IN ('round', 'gt', 'audit')),
  draw        INTEGER NOT NULL,       -- round index for 'round'; draw index for 'gt'/'audit'
  item_idx    INTEGER NOT NULL,
  gen_key     TEXT NOT NULL,
  physical    INTEGER NOT NULL,       -- 1 = fresh generation forced (nonce) or independent sample
  PRIMARY KEY (run_id, seed, split, decoding_id, slot, draw_kind, draw, item_idx)
);
CREATE INDEX IF NOT EXISTS cells_gen ON cells (gen_key);

CREATE TABLE IF NOT EXISTS scores (
  gen_key   TEXT NOT NULL,
  extractor TEXT NOT NULL,            -- registry name, e.g. 'v1'
  ext_hash  TEXT NOT NULL,            -- source hash of the frozen extractor implementation
  extracted TEXT,                     -- canonical numeric string or NULL if nothing extractable
  method    TEXT NOT NULL,            -- which rule fired (boxed, hash, answer_phrase, bold, last_number, none, ...)
  span_start INTEGER,
  span_end   INTEGER,
  gold      TEXT NOT NULL,
  correct   INTEGER NOT NULL,
  PRIMARY KEY (gen_key, extractor, gold)
);

CREATE TABLE IF NOT EXISTS ledger (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id            TEXT NOT NULL,
  seed              INTEGER,
  stage             TEXT NOT NULL,
  purpose           TEXT NOT NULL,    -- trajectory_dev | candidate_dev | proposer | eval_matrix | gt_draw | audit | interactive
  round             INTEGER,
  n_requested       INTEGER NOT NULL,
  n_executed        INTEGER NOT NULL,
  n_cache_hits      INTEGER NOT NULL,
  prompt_tokens     INTEGER,
  completion_tokens INTEGER,
  wall_s            REAL,
  created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stage_status (
  run_id     TEXT NOT NULL,
  stage      TEXT NOT NULL,
  seed       INTEGER NOT NULL DEFAULT -1,
  n_planned  INTEGER,
  n_done     INTEGER,
  status     TEXT,
  updated_at TEXT,
  PRIMARY KEY (run_id, stage, seed)
);

CREATE TABLE IF NOT EXISTS shard_log (
  seq    INTEGER PRIMARY KEY,
  path   TEXT NOT NULL,
  n_rows INTEGER NOT NULL,
  sha256 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_results (
  run_id             TEXT NOT NULL,
  decoding_id        TEXT NOT NULL,
  slot               INTEGER NOT NULL,
  repeat             INTEGER NOT NULL,
  pct_text_identical REAL,
  pct_correct_flip   REAL,
  n                  INTEGER,
  PRIMARY KEY (run_id, decoding_id, slot, repeat)
);

-- ------------------------------------------------------------------ derived (rebuilt by analyze)
CREATE TABLE IF NOT EXISTS analysis_results (
  analysis_id  TEXT NOT NULL,         -- plan_hash
  name         TEXT NOT NULL,         -- e.g. 'allpairs', 'policy_rounds', 'tables/T3'
  payload_json TEXT NOT NULL,
  created_at   TEXT NOT NULL,
  PRIMARY KEY (analysis_id, name)
);

CREATE TABLE IF NOT EXISTS interactive_events (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  backend    TEXT NOT NULL,
  seed       INTEGER,
  slot       INTEGER,
  env_id     TEXT,
  item_idx   INTEGER,
  gen_key    TEXT,
  note       TEXT
);
