"""
Part 4 — Web: minimal salesperson-facing interface.

Lets a salesperson enter a lead's details and see its likelihood-to-convert
score, using the same cleaning + pipeline as Part 3.

Run:
    pip install streamlit pandas scikit-learn --break-system-packages
    streamlit run app.py

Needs leads.csv in the same folder (used to train the baseline model at
startup — no separate training step, no saved model file to ship).
"""

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

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

CATEGORICAL = ["source", "city", "area", "property_type"]
NUMERIC = [
    "budget_pkr_lac", "bedrooms", "first_response_minutes", "calls_made",
    "total_call_seconds", "whatsapp_replies", "site_visits",
    "agent_experience_years", "is_overseas", "referred_by_existing_client",
    "has_financing_approved",
]


@st.cache_resource
def load_model():
    df = pd.read_csv("leads.csv")
    df = df.sort_values("created_at").drop_duplicates(
        subset="crm_record_hash", keep="first"
    )
    df["city"] = df["city"].str.strip().str.lower().map(CITY_MAP).fillna(df["city"])
    df["bedrooms"] = df["bedrooms"].fillna(0)
    df["area"] = df["area"].fillna("Unknown")

    y = df["converted"]
    X = df[NUMERIC + CATEGORICAL]

    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    pipe = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                    random_state=RANDOM_STATE)),
    ])
    pipe.fit(X, y)

    sources = sorted(df["source"].dropna().unique())
    cities = sorted(df["city"].dropna().unique())
    areas = sorted(df["area"].dropna().unique())
    ptypes = sorted(df["property_type"].dropna().unique())
    return pipe, sources, cities, areas, ptypes


st.set_page_config(page_title="MGC Lead Scorer")
st.title("MGC Lead Scorer")
st.caption("Baseline model — enter a lead's details, get a likelihood-to-convert score.")

model, sources, cities, areas, ptypes = load_model()

with st.form("lead_form"):
    col1, col2 = st.columns(2)
    with col1:
        source = st.selectbox("Source", sources)
        city = st.selectbox("City", cities)
        area = st.selectbox("Area", areas)
        property_type = st.selectbox("Property type", ptypes)
        bedrooms = st.number_input("Bedrooms (0 for Plot/Commercial)", 0, 6, 0)
        budget_pkr_lac = st.number_input("Budget (PKR lac)", 0.0, 2000.0, 150.0)
    with col2:
        first_response_minutes = st.number_input("First response time (min)", 0.0, 5000.0, 30.0)
        calls_made = st.number_input("Calls made", 0, 20, 2)
        total_call_seconds = st.number_input("Total call seconds", 0.0, 10000.0, 60.0)
        whatsapp_replies = st.number_input("WhatsApp replies", 0, 20, 1)
        site_visits = st.number_input("Site visits", 0, 10, 0)
        agent_experience_years = st.number_input("Agent experience (years)", 0.0, 30.0, 3.0)

    col3, col4, col5 = st.columns(3)
    with col3:
        is_overseas = st.checkbox("Overseas buyer")
    with col4:
        referred = st.checkbox("Referred by existing client")
    with col5:
        has_financing = st.checkbox("Financing pre-approved")

    submitted = st.form_submit_button("Score this lead")

if submitted:
    row = pd.DataFrame([{
        "source": source, "city": city, "area": area,
        "property_type": property_type, "bedrooms": bedrooms,
        "budget_pkr_lac": budget_pkr_lac,
        "first_response_minutes": first_response_minutes,
        "calls_made": calls_made, "total_call_seconds": total_call_seconds,
        "whatsapp_replies": whatsapp_replies, "site_visits": site_visits,
        "agent_experience_years": agent_experience_years,
        "is_overseas": int(is_overseas),
        "referred_by_existing_client": int(referred),
        "has_financing_approved": int(has_financing),
    }])
    score = model.predict_proba(row)[0, 1]
    st.metric("Likelihood to convert", f"{score:.1%}")
    if score >= 0.5:
        st.success("High priority — call this lead first.")
    elif score >= 0.2:
        st.info("Medium priority.")
    else:
        st.warning("Low priority.")
