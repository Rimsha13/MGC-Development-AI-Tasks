# Part 3 — ML: Lead Scoring Baseline

Trains a baseline model on `leads.csv` that scores how likely a lead is to
convert, so the sales team knows who to call first.

## How to run

**Locally**
```bash
pip install pandas scikit-learn
python3 part3_ml_baseline.py
```
Needs `leads.csv` in the same folder. Prints the cleaning steps, feature
list, class balance, and the evaluation metrics for two models.

**Google Colab**
```python
!pip install pandas scikit-learn -q

from google.colab import files
uploaded = files.upload()   # upload leads.csv and part3_ml_baseline.py

!python part3_ml_baseline.py
```

## What it does

1. **Loads and de-duplicates** `leads.csv` on `crm_record_hash` (see data
   decisions below) — 9,160 rows → 9,000 unique leads.
2. **Cleans** city spelling, fixes structural NaNs in `bedrooms`, fills
   missing `area`.
3. **Drops leakage and non-feature columns** (`token_amount_received_pkr`,
   `lead_id`, `crm_record_hash`, `created_at`).
4. **Splits** 80/20, stratified on `converted` (positives are rare — a
   random split could easily starve the test set of them).
5. **Trains** two baselines through an impute → encode/scale → classify
   pipeline: logistic regression and a random forest, both with
   `class_weight="balanced"`.
6. **Reports** PR-AUC (average precision) as the headline metric, ROC-AUC
   as a secondary number.

## Data decisions

- **Deduplication:** 160 rows are the same lead entered twice under
  different `lead_id`s (one has a `-B` suffix), same `crm_record_hash`,
  same `created_at`, matching every other field. This matches the pattern
  the brief describes — "same lead entered twice by different agents."
  Kept the first occurrence.
- **`token_amount_received_pkr` dropped — this is target leakage, not a
  feature.** 100% of converted leads have a token amount > 0; only ~1% of
  non-converted leads do (data-entry noise, not signal). A token exists
  *because* the lead converted, so it isn't known at prediction time.
  Training on it produces a model that looks excellent and is useless in
  production — it would learn "did they pay the token" as a proxy for
  "did they convert," which is circular.
- **`lead_id`, `crm_record_hash` dropped** — identifiers, no predictive
  content.
- **`created_at` dropped** — a baseline isn't the place to model
  seasonality; would revisit for a v2 (day-of-week, month, lead age).
- **`bedrooms` NaN is structural, not missing at random.** Checked: every
  NaN belongs to a `Plot` or `Commercial Shop` row (100% consistent) —
  those property types don't have a bedroom count. Filled with `0`
  rather than imputing a fake bedroom number, which would have invented
  signal that isn't there.
- **`city` had inconsistent casing/abbreviations** (`Islamabad` /
  `ISLAMABAD` / `ISB`, `Rawalpindi` / `RWP`, etc.) — normalized to one
  canonical name per city before encoding. Left as separate categories,
  the model would have treated `Isb` and `Islamabad` as unrelated cities.
- **`area`** — low cardinality (10 values), some genuine missingness,
  filled with `"Unknown"` and kept as a feature.
- **`budget_pkr_lac`, `first_response_minutes`,
  `agent_experience_years`** — real missing values, left for the
  pipeline's median imputer (fit on train only, so no leakage across the
  split).

## Metric: PR-AUC (average precision) = 0.395

Chosen because only **6.9% of leads convert**. With that imbalance:

- **Accuracy is useless** — predicting "never converts" for every lead
  scores ~93% accuracy while being worthless.
- **ROC-AUC (0.816) can flatter a model** — it's evaluated against a
  50/50 baseline and is dominated by how well the model ranks the huge
  majority (non-converting) class, which isn't the business problem.
- **PR-AUC is judged against the actual positive rate (0.069) as its
  baseline**, not 50%. It directly measures how well the model separates
  real converters from the noise — which is the entire point: ranking
  the small pool of likely converters above the rest so the sales team
  calls them first.

Logistic regression outperformed the random forest on PR-AUC (0.395 vs
0.282) despite similar ROC-AUC (0.816 vs 0.803) — worth investigating
further with proper tuning, but not in scope for a no-tuning baseline.

## What I'd do next (out of scope for the hour)

- Feature: lead age / recency instead of dropping `created_at` outright.
- Calibrate probabilities (Platt scaling / isotonic) if the raw scores
  are ever shown to salespeople as "% chance," not just used for ranking.
- Cross-validation instead of a single train/test split, given how few
  positives there are (626) — a single split's PR-AUC has real variance.
- Investigate the random forest underperforming logistic regression —
  possibly needs shallower trees or more estimators tuned properly.
