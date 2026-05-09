# SHL Conversational Assessment Recommender

A RAG-based conversational API that helps hiring managers find the right SHL assessments through natural dialogue. Built with FastAPI, FAISS vector search, and Google Gemini. Deployed on Render.com.

---

## What it does

Most hiring managers don't know which SHL assessment they need — they know the role, sometimes the seniority, and not much else. This API takes a conversation and guides them from a vague intent to a grounded shortlist of real assessments from the SHL catalog.

It handles four things:
- **Clarify** — asks questions before recommending when the query is vague
- **Recommend** — returns 1–10 assessments once it has enough context
- **Refine** — updates the shortlist when the user adds or removes requirements
- **Compare** — explains the difference between two assessments using catalog data only

It only discusses SHL assessments. Legal questions, salary questions, and general hiring advice are refused.

---

## Project Structure

```
shl_assignment/
├── main.py               # FastAPI app — endpoints, validation, error handling
├── agent.py              # Agent logic — retrieval, Gemini calls, JSON parsing, URL validation
├── retrieval.py          # CatalogRetriever — FAISS index loading and semantic search
├── build_index.py        # One-time script — builds FAISS index from catalog.json
├── catalog.json          # SHL product catalog (377 items)
├── faiss_index/          # Pre-built vector index (committed to repo)
│   ├── index.faiss       # FAISS IndexFlatIP — 377 vectors × 3072 dims
│   └── metadata.json     # Catalog metadata aligned to index
├── requirements.txt
├── render.yaml
└── .env                  # GEMINI_API_KEY
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | FastAPI + Uvicorn |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | gemini-embedding-001 (3072 dimensions) |
| Vector Store | FAISS IndexFlatIP |
| Validation | Pydantic v2 |
| Deployment | Render.com free tier |

---

## Setup

### Prerequisites
- Python 3.10+
- Google Gemini API key — [get one free here](https://aistudio.google.com/apikey)

### Run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API key
echo GEMINI_API_KEY=your_key_here > .env

# 3. Start the server (faiss_index/ is pre-built and committed)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4. Verify
curl http://localhost:8000/health
# → {"status": "ok"}
```

> To rebuild the FAISS index from scratch: `python build_index.py` (takes ~5 minutes on free tier)

---

## API Reference

### GET /health

Always returns HTTP 200. No external dependencies.

```json
{"status": "ok"}
```

---

### POST /chat

Stateless — full conversation history must be sent on every request.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need to hire a senior Java developer"},
    {"role": "assistant", "content": "{\"reply\": \"What seniority level?\", \"recommendations\": [], \"end_of_conversation\": false}"},
    {"role": "user", "content": "Senior, 5+ years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are assessments for a senior Java developer:",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

**Schema rules:**
- `recommendations` is always an array — empty `[]` when clarifying or refusing, 1–10 items when recommending
- `end_of_conversation` is a boolean, never null
- Every URL is validated against the real catalog before being returned

**Error handling** — the API never returns 5xx. All errors come back as valid `ChatResponse`:

| Scenario | end_of_conversation |
|---|---|
| Turn limit hit (≥ 8 messages) | true |
| Timeout (> 25s) | false |
| Internal error | false |

---

## Test Type Abbreviations

| Category | Code |
|---|---|
| Ability & Aptitude | A |
| Personality & Behavior | P |
| Knowledge & Skills | K |
| Simulations | S |
| Biodata & Situational Judgment | B |
| Competencies | C |
| Development & 360 | D |
| Assessment Exercises | E |

Example: `keys: ["Knowledge & Skills", "Simulations"]` → `test_type: "K,S"`

---

## How it works

**Index building (offline):**

Each catalog item is embedded as a combined text string: name + description + keys + job levels + top 5 languages. This multi-field embedding means queries mentioning seniority or language constraints retrieve the right items even if those words don't appear in the assessment name.

Embeddings use `gemini-embedding-001` (3072 dims), L2-normalized, stored in a FAISS `IndexFlatIP`. Inner product on normalized vectors = cosine similarity.

**At query time:**

The last user message is combined with the names of any previously recommended assessments before searching. This matters on refine and compare turns — without the prior context, FAISS doesn't have enough signal to pull up items already in the shortlist.

Top 20 results are retrieved and injected into the system prompt. The LLM picks the final 1–10. After the response, every URL is checked against a set of valid catalog links — anything not in the catalog is dropped before returning.

---

## Deployment

The `faiss_index/` directory is committed to the repo (~4.6 MB) so Render doesn't need to rebuild it on cold start.

```yaml
# render.yaml
services:
  - type: web
    name: shl-recommender
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: GEMINI_API_KEY
        sync: false
```

Steps:
1. Push repo to GitHub including `faiss_index/`
2. Render → New Web Service → connect repo
3. Add `GEMINI_API_KEY` in the Environment tab
4. Deploy — first `/health` call may take up to 2 minutes on cold start

---

## Known limitation

The free-tier Gemini API has daily rate limits. Under heavy load the service may slow down or return fallback responses. A paid API key removes this constraint.
