# Part 2 — Database

## What this is

MGC's `leads.csv` (~9,000 rows) is a messy CRM export. This folder has:

- **`schema.sql`** — the table I'd actually store this in, with types, nullable
  columns explained, and a constraint that stops duplicate leads at insert time.
- **`queries.sql`** — the two required queries: conversion rate by lead source
  (200+ leads only, best first), and a query that finds the duplicate leads.

## What I found and decided

- **One table, not several.** The only "other entity" hiding in the data is the
  customer behind `crm_record_hash` — but we have no other attributes for that
  customer (no name/phone/email), so splitting one out today would just be an
  empty table joined for no reason. Noted in `schema.sql` where I'd split it
  later if MGC hands over real identity fields.
- **Duplicates, checked not assumed.** `lead_id` is 100% unique on its own, so
  I grouped by `(crm_record_hash, created_at)` instead and found 160 pairs of
  rows that are identical in every column except `lead_id` (e.g. `MGC-104974`
  vs `MGC-104974-B`). That matches the brief's hint — same lead, entered twice
  by two agents at the same moment — not a person genuinely re-enquiring
  later. That's why the schema's `UNIQUE` constraint is keyed on
  `(crm_record_hash, created_at)`, not on the hash alone: it blocks the exact
  double-entry case without blocking a real second enquiry from the same
  person on a different day.
- **Query 1 result** (for reference, computed from the actual CSV):
  Referral leads convert best (13.0%) despite being far fewer than Facebook
  Ads (6.8%) or Property Portal (6.0%) — worth a mention to the sales team.

## How to test this yourself

You need three files together: `leads.csv`, `schema.sql`, `queries.sql`.

### Option A — no install, in your browser

1. Go to **https://extendsclass.com/sqlite-browser.html**
2. Click **"Import CSV file"** and select `leads.csv`. Name the table `leads`.
3. Open the **"SQL"** or query tab, paste the contents of `queries.sql`, and
   click **Run**. You'll see the conversion-rate table, then the 160
   duplicate-lead groups.
4. Don't import `schema.sql` into this same database — see the note below.

   *(Alternative site if that one is down or unfamiliar: https://sqliteonline.com/
   — same idea: create a SQLite database, import the CSV, paste and run the
   queries.)*

### Option B — on your own machine (SQLite CLI)

Covered step-by-step earlier in this chat: put the three files in one folder,
run `sqlite3 test.db`, then `.mode csv` and `.import leads.csv leads`, then
`.read queries.sql`.

### Why you should NOT load `schema.sql` into the same table as the CSV

`schema.sql`'s `UNIQUE (crm_record_hash, created_at)` constraint is designed to
**reject** the 160 duplicate pairs that are already sitting in `leads.csv`.
If you create the table from `schema.sql` first and then try to import the raw
CSV into it, the import will fail partway through with a "UNIQUE constraint
failed" error — that's the constraint working correctly, not a bug.
`schema.sql` is the target design for new leads going forward, not a loader
for the existing messy dump.

If you want to see this happen on purpose: create a fresh database, run
`schema.sql` against it first, *then* try importing `leads.csv` — it'll stop
on the first duplicate.
