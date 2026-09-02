"""
Part 3 — ML: baseline lead-conversion scorer for MGC Aurora Heights leads.

Run: python3 part3_ml_baseline.py
Requires: pandas, scikit-learn  (pip install pandas scikit-learn --break-system-packages)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    classification_report
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load + de-duplicate
# ---------------------------------------------------------------------------
df = pd.read_csv("leads.csv")
print(f"Raw rows: {len(df)}")

# crm_record_hash identifies the same lead re-entered by a different agent
# (lead_id gets a '-B' suffix on the re-entry). Keep the first occurrence.
before = len(df)
df = df.sort_values("created_at").drop_duplicates(subset="crm_record_hash", keep="first")
print(f"Dropped {before - len(df)} duplicate re-entries -> {len(df)} unique leads")

# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------

# --- city: same city typed in different cases / abbreviations ---
CITY_MAP = {
    "islamabad": "Islamabad", "isb": "Islamabad",
    "rawalpindi": "Rawalpindi", "rwp": "Rawalpindi",
    "lahore": "Lahore",
    "karachi": "Karachi", "khi": "Karachi",
    "peshawar": "Peshawar",
    "faisalabad": "Faisalabad",
    "multan": "Multan",
    "gujranwala": "Gujranwala",
    "abbottabad": "Abbottabad",
}
df["city"] = df["city"].str.strip().str.lower().map(CITY_MAP).fillna(df["city"])

# --- bedrooms: NaN is not "missing" here, it's structural. Plots and
# Commercial Shops don't have a bedroom count, they show NaN 100% of the
# time (checked). Fill with 0 rather than imputing a fake bedroom count.
df["bedrooms"] = df["bedrooms"].fillna(0)

# --- area: genuinely missing sometimes, low cardinality (10 values) ---
df["area"] = df["area"].fillna("Unknown")

# budget_pkr_lac, first_response_minutes, agent_experience_years: real
# missingness (data not captured), handled by the pipeline's median
# imputer below rather than by hand here.

# ---------------------------------------------------------------------------
# 3. Columns dropped and why (see README for the full writeup)
# ---------------------------------------------------------------------------
DROP_COLS = [
    "lead_id",            # unique identifier, no predictive signal
    "crm_record_hash",    # dedup key only, not a feature
    "created_at",         # raw timestamp; no seasonality modelling in a baseline
    "token_amount_received_pkr",  # LEAKAGE: 100% of converted leads have a
                                   # token amount > 0, ~1% of non-converted do
                                   # (data-entry noise). A token is *received
                                   # because* the lead converted — this column
                                   # doesn't exist yet at prediction time.
]
df_model = df.drop(columns=DROP_COLS)

TARGET = "converted"
y = df_model[TARGET]
X = df_model.drop(columns=[TARGET])

CATEGORICAL = ["source", "city", "area", "property_type"]
NUMERIC = [c for c in X.columns if c not in CATEGORICAL]
print("\nNumeric features:", NUMERIC)
print("Categorical features:", CATEGORICAL)
print(f"\nClass balance: {y.mean():.1%} converted ({y.sum()} of {len(y)})")

# ---------------------------------------------------------------------------
# 4. Train / test split (stratified, because positives are rare)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 5. Pipeline: impute -> encode/scale -> classifier
# ---------------------------------------------------------------------------
numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", numeric_pipe, NUMERIC),
    ("cat", categorical_pipe, CATEGORICAL),
])

models = {
    "logistic_regression": Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                    random_state=RANDOM_STATE)),
    ]),
    "random_forest": Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=8,
                                        class_weight="balanced_subsample",
                                        random_state=RANDOM_STATE, n_jobs=-1)),
    ]),
}

# ---------------------------------------------------------------------------
# 6. Fit + evaluate
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)

    print(f"\n--- {name} ---")
    print(f"PR-AUC (average precision): {pr_auc:.3f}   "
          f"[baseline = positive rate = {y_test.mean():.3f}]")
    print(f"ROC-AUC (secondary):        {roc_auc:.3f}")

print("\n" + "=" * 60)
print("""
Chosen metric: PR-AUC (average precision).

Why: only 6.9% of leads convert. With that imbalance, accuracy is useless
(predicting "never converts" scores ~93% accuracy) and ROC-AUC can look
deceptively good because it's dominated by the huge non-converting class.
PR-AUC focuses on the positive (converting) class and is judged against the
positive rate as its baseline, not 50%, which is what actually matters here:
this model's whole job is to rank the small pool of real converters above
the noise so the sales team calls them first.
""")
