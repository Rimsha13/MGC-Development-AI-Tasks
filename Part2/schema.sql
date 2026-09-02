-- MGC Leads — minimal schema
-- Dialect: written for PostgreSQL; two lines flagged below need a swap for
-- MySQL/SQLite (SQLite in particular has no real ENUM/CHECK-list issue, and
-- no native BOOLEAN — it stores 0/1 in an INTEGER, which is fine as-is).

-- ---------------------------------------------------------------------------
-- Why one table and not several:
-- Every row is a single, atomic event — "this contact enquired about this
-- property at this time." The only other "entity" hiding in the data is the
-- underlying customer, identified by crm_record_hash (a hash of name/phone/
-- CNIC — we never see the real PII). But we have no other attributes of that
-- customer to store (no name, phone, email columns), so a separate
-- `customers` table today would just be `(hash)` with nothing else in it —
-- a join for no benefit. If MGC ever hands over the raw identity fields, that
-- is the moment to split into `customers (1) -> leads (many)`; until then one
-- table is the honest model of what we actually have.
-- ---------------------------------------------------------------------------

CREATE TABLE leads (
    lead_id                       TEXT PRIMARY KEY,
        -- e.g. 'MGC-104067'. Kept as the CRM's own id rather than inventing a
        -- surrogate key — it's already unique and human-referenceable.

    crm_record_hash               BIGINT NOT NULL,
        -- Hash identifying the underlying contact (phone/CNIC/email, hashed
        -- upstream — we never store raw PII here). NOT unique by itself: the
        -- same person legitimately generates more than one lead over time
        -- (e.g. enquires again six months later about a different unit).

    created_at                    TIMESTAMP NOT NULL,

    source                        TEXT NOT NULL
        CHECK (source IN (
            'Facebook Ads','Property Portal','Google Search','Instagram',
            'Referral','Walk-in','WhatsApp Campaign','Expo Stall','Billboard'
        )),
        -- CHECK list taken from the values actually present in the dump.
        -- A lookup table (`lead_sources`) would be the "more normalized"
        -- answer, but for 9 fixed values a CHECK is simpler and just as safe;
        -- I'd switch to a lookup table if the source list is expected to grow
        -- or need per-source metadata (e.g. cost per lead).

    city                          TEXT NOT NULL,
    area                          TEXT,
        -- Nullable — ~5% missing in the source dump (e.g. leads logged before
        -- the area was captured, or non-local enquiries). Not worth rejecting
        -- the whole row over.

    property_type                 TEXT NOT NULL,

    budget_pkr_lac                NUMERIC(10,2),
        -- Nullable — ~3% missing (unstated/unqualified budget at enquiry time).

    bedrooms                      SMALLINT,
        -- Nullable — ~40% missing. Not a data-quality problem: commercial and
        -- studio leads legitimately have no bedroom count.

    first_response_minutes        NUMERIC(10,2),
    calls_made                    SMALLINT NOT NULL DEFAULT 0,
    total_call_seconds            NUMERIC(10,2) NOT NULL DEFAULT 0,
    whatsapp_replies              SMALLINT NOT NULL DEFAULT 0,
    site_visits                   SMALLINT NOT NULL DEFAULT 0,

    agent_experience_years        NUMERIC(5,2),
        -- Nullable — lead not yet assigned to an agent.

    is_overseas                   BOOLEAN NOT NULL DEFAULT FALSE,          -- SQLite: INTEGER NOT NULL DEFAULT 0
    referred_by_existing_client   BOOLEAN NOT NULL DEFAULT FALSE,          -- SQLite: INTEGER NOT NULL DEFAULT 0
    has_financing_approved        BOOLEAN NOT NULL DEFAULT FALSE,          -- SQLite: INTEGER NOT NULL DEFAULT 0
    token_amount_received_pkr     NUMERIC(14,2) NOT NULL DEFAULT 0,
    converted                     BOOLEAN NOT NULL DEFAULT FALSE,          -- SQLite: INTEGER NOT NULL DEFAULT 0

    -- --- Duplicate prevention lives here ---
    -- The duplicates in this dump are the SAME customer, SAME timestamp,
    -- entered under two different lead_ids (two agents logging the same
    -- call/walk-in independently — see queries.sql for how this was
    -- confirmed). Keying on (hash, created_at) blocks that exact case at
    -- INSERT time, while still allowing the same person to come back and
    -- generate a genuinely new lead on a different day.
    CONSTRAINT uq_lead_identity UNIQUE (crm_record_hash, created_at)
);

CREATE INDEX idx_leads_source    ON leads(source);
CREATE INDEX idx_leads_converted ON leads(converted);
CREATE INDEX idx_leads_hash      ON leads(crm_record_hash);
