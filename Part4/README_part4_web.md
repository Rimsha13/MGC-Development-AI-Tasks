# Part 4 — Web: Lead Scorer Interface

A single-page Streamlit app so a salesperson can enter a lead's details and
get a likelihood-to-convert score, without touching a notebook.

## How to run

**Locally**
```bash
pip install streamlit pandas scikit-learn
streamlit run app.py
```
Needs `leads.csv` in the same folder. Opens in your browser automatically
(default `http://localhost:8501`).

**Google Colab** (Colab can't serve a local port to your browser directly,
so it needs a tunnel)
```python
!pip install streamlit pandas scikit-learn -q

from google.colab import files
uploaded = files.upload()   # upload leads.csv and app.py

!streamlit run app.py &>/content/logs.txt &
!npx localtunnel --port 8501
```
This prints a URL like `https://xxxx.loca.lt`. Before opening it, run:
```python
!curl -s https://loca.lt/mytunnelpassword
```
and paste the returned IP into the "Tunnel Password" field the page asks
for on first load.

## What it does

1. On startup, trains the same baseline pipeline from Part 3
   (deduplicate → clean → impute/encode/scale → logistic regression) on
   `leads.csv`, cached with `@st.cache_resource` so it only trains once
   per session, not on every form submit.
2. Renders a form covering every feature the model uses: source, city,
   area, property type, bedrooms, budget, engagement signals (calls made,
   WhatsApp replies, site visits, response time), agent experience, and
   the three yes/no flags (overseas, referred, financing approved).
3. Dropdown options (source, city, area, property type) are populated
   from the actual cleaned values in `leads.csv`, so a salesperson can't
   enter a city spelling the model has never seen.
4. On submit, builds a one-row DataFrame from the form inputs, runs
   `model.predict_proba`, and shows:
   - the raw probability as a percentage,
   - a High / Medium / Low priority label (thresholds at 50% / 20%) so
     the number is immediately actionable, not just a score to interpret.

## Design choices

- **Streamlit, not Flask/FastAPI** — the brief explicitly said "no
  styling points" and stack is free choice; Streamlit gets a working
  form + result page with the least code, which is what a 15-minute
  budget calls for.
- **Trains at runtime instead of loading a saved model file** — keeps
  the repo to two files (`app.py`, `leads.csv`) with nothing to
  regenerate or go stale. Fine at this data size (9k rows trains in
  under a second); would switch to a pre-trained, saved model
  (`joblib`) if this needed to survive real traffic or a larger dataset.
- **Reuses the exact cleaning logic from Part 3** (city normalization,
  bedroom NaN → 0, area NaN → "Unknown") rather than a separate copy, so
  the score a salesperson sees matches what the Part 3 evaluation
  actually measured.

## Known limitations / what's unfinished

- Only the lead-scoring half of Part 4 is built. The document-assistant
  half (Part 1 tied into this same page) is not wired in here — scoring
  a lead was the one I picked, per the brief's "either one is enough."
- I could not launch Streamlit inside the sandbox I built this in (no
  network access to `pip install streamlit`), so the UI itself is
  **not smoke-tested end-to-end**. I did verify the model-training and
  scoring logic the app calls standalone (outside Streamlit) and it
  behaves sensibly — e.g. a referred lead with financing approved and a
  fast response scored ~99%, a cold billboard lead with zero engagement
  scored ~2%. Please do a quick local run before relying on it.
- No input validation beyond Streamlit's own numeric bounds — a
  salesperson entering nonsense (e.g. 500 calls made) won't be stopped,
  just scored on it.
