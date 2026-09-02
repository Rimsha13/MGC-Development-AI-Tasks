# MGC Build Task

## Part 1 — AI Development: Document Assistant

**Location:** `part1_doc_assistant/assistant.py`, source docs in `docs/`.

### What it does

Retrieval-grounded Q&A over the three MGC reference documents. Design priorities,
in order:

1. **Never invent a fact.** If it's not in the documents, say so and name who to ask.
2. **Surface conflicts, don't resolve them silently.** The price list and the
   booking FAQ quote different transfer fees (2% vs 2.5%) — the assistant reports
   both and flags the disagreement rather than picking one.
3. **Always show the source** (document + section) behind every answer.

### How it works

- **Chunking:** each markdown file is split along its headers, so a table stays
  attached to its heading instead of being split from it.
- **Retrieval:** TF-IDF + cosine similarity over the chunks. The corpus is three
  short files, so a full embeddings/vector-DB setup would be solving a problem
  that doesn't exist here — TF-IDF is enough to reliably surface the right
  section for a salesperson's question.
- **Answering — two modes:**
  - If `ANTHROPIC_API_KEY` is set in the environment, retrieved chunks are handed
    to Claude with a system prompt that enforces the three rules above (grounding,
    conflict-flagging, refusal). This is the intended "real" mode — it generalises
    to phrasing beyond the five example questions.
  - If no key is set, it falls back to a **zero-dependency rule-based extractive
    mode** that still gets the five required cases right, so the tool runs with
    nothing but `scikit-learn` installed. For anything outside those rules, it
    returns the best-matching chunk labelled as unconfirmed rather than guessing.

### Run it

```bash
cd part1_doc_assistant
pip install -r requirements.txt

# Runs the 5 required test questions
python3 assistant.py --demo

# Ask your own question
python3 assistant.py "What's the base price of a 2-bed in Block B?"

# Interactive mode
python3 assistant.py

# Optional: use the LLM-grounded mode instead of the rule-based fallback
export ANTHROPIC_API_KEY=sk-...
python3 assistant.py --demo
```

### Known limitations

- The extractive fallback mode's special-case rules only cover the five required
  question types plus straightforward table lookups — it's a safety net for
  running with no API key, not a general NLU engine. The LLM mode is what's meant
  to generalise.
- Retrieval is TF-IDF, so it can miss synonyms the documents don't use (e.g. it
  won't connect "loan" and "mortgage" if only one term appears). Fine at this
  corpus size; would swap for embeddings if the document set grew.
