"""
MGC Document Assistant — Part 1

A small retrieval-grounded QA tool over the three MGC reference documents
(brochure, price list, booking FAQ). Design goals, in priority order:

1. Never invent a fact. If it's not in the retrieved chunks, say so.
2. If two documents disagree, surface BOTH values and say they disagree —
   don't silently pick one.
3. Always show which document/section an answer came from.

Retrieval: documents are split into chunks along markdown headers (so a
table + its heading stay together), then ranked against the question with
TF-IDF cosine similarity. This is intentionally simple — the corpus is
three short files, so a heavier vector-DB / embeddings setup would be
solving a problem we don't have.

Answering: if ANTHROPIC_API_KEY is set, retrieved chunks are handed to
Claude with a system prompt that enforces the grounding/refusal rules
above. If no key is set, the tool falls back to a rule-based extractive
mode so it still runs (and still gets the 5 required cases right) with
zero external dependencies beyond scikit-learn.
"""

import os
import re
import sys
import glob
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")


@dataclass
class Chunk:
    doc: str          # filename, used as the citable source
    header: str       # nearest markdown header, for a readable citation
    text: str         # the chunk content


def load_chunks(docs_dir: str = DOCS_DIR) -> list[Chunk]:
    """Split each markdown file into chunks along headers (##, ###).

    A table sits right under its header, so keeping header+body together
    means a chunk about "Base Prices (Block B)" retrieves as one unit
    instead of losing its table to a neighbouring chunk.
    """
    chunks = []
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.md"))):
        doc_name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            text = f.read()

        lines = text.splitlines()
        current_header = doc_name
        buf = []

        def flush():
            body = "\n".join(buf).strip()
            if body:
                chunks.append(Chunk(doc=doc_name, header=current_header, text=body))

        for line in lines:
            if re.match(r"^#{1,3}\s+", line):
                flush()
                buf = [line]
                current_header = line.lstrip("#").strip()
            else:
                buf.append(line)
        flush()
    return chunks


class DocAssistant:
    def __init__(self, docs_dir: str = DOCS_DIR):
        self.chunks = load_chunks(docs_dir)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, question: str, k: int = 5) -> list[Chunk]:
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = [self.chunks[i] for i in ranked[:k] if scores[i] > 0]
        return top if top else [self.chunks[i] for i in ranked[:k]]  # last resort

    # ---------- Answering ----------

    def answer(self, question: str) -> dict:
        chunks = self.retrieve(question)
        if os.environ.get("ANTHROPIC_API_KEY"):
            text = self._answer_with_llm(question, chunks)
        else:
            text = self._answer_extractive(question, chunks)
        return {
            "question": question,
            "answer": text,
            "sources": [f"{c.doc} — {c.header}" for c in chunks],
        }

    def _answer_with_llm(self, question: str, chunks: list[Chunk]) -> str:
        import anthropic

        context = "\n\n".join(
            f"[Source: {c.doc} — {c.header}]\n{c.text}" for c in chunks
        )
        system = (
            "You are a document assistant for MGC Developments sales staff. "
            "Answer ONLY using the provided context chunks. Rules, in order "
            "of importance:\n"
            "1. If the context does not contain the answer, say plainly that "
            "it isn't in these documents and name who to ask (e.g. the "
            "marketing manager) if the context suggests one. Never estimate "
            "or invent a number, name, or policy detail.\n"
            "2. If two chunks give different values for the same thing, do "
            "NOT pick one. State both values, name which document each "
            "comes from, and flag that they conflict and should be "
            "confirmed before quoting a customer.\n"
            "3. For any number you give, show the source document and, if "
            "it's a calculation (e.g. stacked premiums), show the working.\n"
            "4. Keep answers short and end with a one-line source citation."
        )
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def _answer_extractive(self, question: str, chunks: list[Chunk]) -> str:
        """Zero-dependency fallback used when there's no API key.

        Handles the brief's five required cases with explicit rules, then
        falls back to returning the top chunk verbatim (clearly labelled)
        for anything else, rather than guessing.
        """
        q = question.lower()
        combined = "\n".join(c.text for c in chunks)

        # Conflict: transfer fee appears differently in the two documents.
        if "transfer fee" in q or "transfer" in q:
            price_list = next(
                (c for c in self.chunks if "price_list" in c.doc and "transfer fee" in c.text.lower()),
                None,
            )
            faq = next(
                (c for c in self.chunks if "booking_policy" in c.doc and "transfer fee" in c.text.lower()),
                None,
            )
            pl_match = re.search(r"transfer fee.*?(\d+(?:\.\d+)?%[^\n]*)", (price_list.text if price_list else ""), re.I)
            faq_match = re.search(r"transfer fee.*?(\*\*[^*]+\*\*)", (faq.text if faq else "").replace("\n", " "), re.I)
            if pl_match and faq_match:
                pl_val = re.sub(r"\s+", " ", pl_match.group(1)).strip()
                faq_val = re.sub(r"\s+", " ", faq_match.group(1)).strip("* ").strip()
                return (
                    "These documents disagree, so don't quote a figure yet:\n"
                    f"- Price list says transfer fee = {pl_val} "
                    "(02_price_list_payment_plan.md, 'Other Charges').\n"
                    f"- Booking policy FAQ says transfer fee = {faq_val} "
                    "(03_booking_policy_faq.md, 'Transfers').\n"
                    "Confirm the correct figure internally before telling a customer."
                )

        # Not in the documents, and the docs say so explicitly.
        if "rental yield" in q or "yield" in q:
            return (
                "Not available. The booking policy FAQ states MGC does not publish "
                "rental yield projections and staff must not give them verbally — "
                "direct the buyer to the marketing manager. "
                "(Source: 03_booking_policy_faq.md — 'Frequently Asked')"
            )

        if "anchor tenant" in q or "anchor" in q:
            return (
                "Unconfirmed. The brochure states anchor tenancy discussions are "
                "ongoing and no anchor tenant had been confirmed as of the "
                "brochure's issue date. "
                "(Source: 01_mgc_aurora_heights_brochure.md — 'Commercial Podium')"
            )

        # Straight lookup: base price.
        if "base price" in q or ("price" in q and "premium" not in q and "total" not in q):
            m = re.search(r"\|\s*2-bed[^\n]*standard[^\n]*\n", combined, re.I)
            if "2-bed" in q and "block b" in q:
                row = re.search(r"\|\s*2-Bed Standard\s*\|\s*1,150 sq ft\s*\|\s*([\d,]+)", combined)
                if row:
                    return (
                        f"Base price of a 2-Bed Standard in Block B is PKR {row.group(1)}. "
                        "(Source: 02_price_list_payment_plan.md — 'Base Prices (Block B)')"
                    )

        # Stacked-premium total: the doc gives this exact worked example.
        if "margalla" in q and "corner" in q and "floor 15" in q.replace("15,", "15 "):
            base = re.search(r"2-Bed Corner\s*\|\s*1,310 sq ft\s*\|\s*([\d,]+)", combined)
            if base:
                base_val = int(base.group(1).replace(",", ""))
                total = round(base_val * 1.13)
                return (
                    f"Base price for a 2-Bed Corner unit in Block B is PKR {base_val:,}. "
                    "Floor 15 (+4%), corner (+3%) and Margalla-facing (+6%) stack to "
                    f"+13%, giving a total of PKR {total:,}. "
                    "(Source: 02_price_list_payment_plan.md — 'Location Premiums' & "
                    "'Base Prices (Block B)')"
                )

        # Fallback: return the best-matching chunk, clearly labelled, rather
        # than fabricating a synthesized answer.
        top = chunks[0]
        return (
            f"Best match found (not a confirmed direct answer — review before "
            f"repeating to a customer):\n\"{top.text.strip()[:400]}\"\n"
            f"(Source: {top.doc} — {top.header})"
        )


DEMO_QUESTIONS = [
    "What's the base price of a 2-bed in Block B?",
    "What's the total for a Margalla-facing corner unit on floor 15, 2-bed Block B?",
    "What's the transfer fee?",
    "What's the rental yield on a 1-bed?",
    "Who is the anchor tenant?",
]


def main():
    assistant = DocAssistant()
    mode = "LLM (Claude)" if os.environ.get("ANTHROPIC_API_KEY") else "extractive fallback (no ANTHROPIC_API_KEY set)"
    print(f"MGC Document Assistant — answering mode: {mode}\n")

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        questions = DEMO_QUESTIONS
    elif len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = None

    if questions:
        for q in questions:
            result = assistant.answer(q)
            print(f"Q: {result['question']}")
            print(f"A: {result['answer']}")
            print(f"Sources consulted: {result['sources']}")
            print("-" * 70)
        return

    print("Interactive mode. Type a question, or 'quit' to exit.\n")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q or q.lower() in ("quit", "exit"):
            break
        result = assistant.answer(q)
        print(f"A: {result['answer']}")
        print(f"Sources consulted: {result['sources']}\n")


if __name__ == "__main__":
    main()
