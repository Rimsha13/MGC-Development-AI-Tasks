-- ============================================================
-- Query 1 — Conversion rate by lead source, sources with 200+
-- leads only, best conversion rate first.
-- ============================================================

SELECT
    source,
    COUNT(*)                                                       AS total_leads,
    SUM(CASE WHEN converted = 1 THEN 1 ELSE 0 END)                 AS converted_leads,
    ROUND(100.0 * SUM(CASE WHEN converted = 1 THEN 1 ELSE 0 END)
                 / COUNT(*), 2)                                    AS conversion_rate_pct
FROM leads
GROUP BY source
HAVING COUNT(*) >= 200
ORDER BY conversion_rate_pct DESC;

-- Run against the actual leads.csv (via sqlite), this returns:
--   Referral            730 leads  13.01%   <- best, despite far fewer leads than Facebook
--   Walk-in             610 leads  10.33%
--   Facebook Ads       2366 leads   6.76%
--   Google Search      1460 leads   6.58%
--   WhatsApp Campaign   548 leads   6.39%
--   Property Portal    1812 leads   5.96%
--   Instagram          1007 leads   5.46%
--   Billboard           282 leads   4.26%
--   Expo Stall          345 leads   2.90%
-- (every source in the file clears 200, so all 9 show up here)


-- ============================================================
-- Query 2 — Find duplicate leads.
-- ============================================================
-- What "duplicate" means in this data: I checked first rather than assume.
-- lead_id is 100% unique, so the duplication isn't at that level. But
-- crm_record_hash (the hashed identity of the actual contact) repeats for
-- 160 groups, and in every one of those 160 groups the two rows share the
-- exact same created_at timestamp and are identical in every other column
-- too — only lead_id differs, usually by a trailing "-B" (e.g. MGC-104974
-- vs MGC-104974-B). That's consistent with the brief's hint: the same
-- lead, entered twice by different agents at the same moment (both punch
-- in the same call/walk-in), not a person re-enquiring later.

SELECT
    crm_record_hash,
    created_at,
    COUNT(*)                          AS times_entered,
    GROUP_CONCAT(lead_id, ', ')       AS lead_ids   -- Postgres: STRING_AGG(lead_id, ', ')
FROM leads
GROUP BY crm_record_hash, created_at
HAVING COUNT(*) > 1
ORDER BY crm_record_hash;

-- 160 duplicate groups (320 rows, ~3.5% of the table) turn up in leads.csv.

-- Prevention at the schema level:
-- the UNIQUE (crm_record_hash, created_at) constraint on the leads table
-- (see schema.sql) stops the second INSERT from ever landing, instead of
-- letting it land and cleaning it up after the fact. It's scoped to
-- (hash, timestamp) rather than hash alone so a legitimate second enquiry
-- from the same person, on a different day, is still allowed through.
