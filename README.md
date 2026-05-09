# 🧠 SHL Conversational Assessment Recommender

A production-ready **Retrieval-Augmented Generation (RAG)** API that helps hiring managers and recruiters discover the right SHL assessments through natural conversation. Built with **FastAPI**, **FAISS** vector search, and **Google Gemini**, deployed on **Render.com**.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [API Reference](#-api-reference)
- [Catalog Details](#-catalog-details)
- [RAG Pipeline Deep Dive](#-rag-pipeline-deep-dive)
- [Agent Behavior & Conversation Design](#-agent-behavior--conversation-design)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Design Decisions](#-design-decisions)
- [Limitations & Future Work](#-limitations--future-work)

---

## 🎯 Project Overview

### Problem
Hiring teams face a catalog of **377 SHL assessments** spanning cognitive ability, personality, skills, simulations, and more. Selecting the right combination for a role requires domain expertise and knowledge of the full catalog — something most hiring managers lack.

### Solution
A **conversational AI recommender** that:
1. **Understands intent** — Interprets natural language descriptions of hiring needs.
2. **Retrieves relevant assessments** — Uses semantic vector search (FAISS + Google Embeddings) to find the most relevant assessments from the full 377-item catalog.
3. **Recommends intelligently** — Leverages Google Gemini to generate contextual, validated recommendations grounded in real catalog data.
4. **Supports multi-turn refinement** — Handles follow-ups, comparisons, adding/dropping assessments, and scope changes.

### Key Capabilities
| Capability | Description |
|---|---|
| 🔍 Semantic Search | Finds relevant assessments even with vague or indirect descriptions |
| 💬 Multi-turn Conversation | Remembers context across turns, refines recommendations |
| ✅ Catalog-Grounded | Every recommendation URL is validated against the real catalog |
| 🚫 Scope Enforcement | Refuses off-topic questions (legal, salary, general hiring advice) |
| ⏱️ Timeout Protection | 25-second timeout prevents hanging requests |
| 📊 Turn Limit | Auto-closes conversations after 8 messages |

---

## 🏗 Architecture

```
┌─────────────┐     POST /chat      ┌──────────────────────────────────────────────┐
│             │ ──────────────────▶  │  FastAPI Server (main.py)                    │
│   Client    │                     │  ┌────────────────────────────────────────┐   │
│  (Frontend  │                     │  │  Input Validation & Turn Limit Check   │   │
│   or cURL)  │                     │  └──────────────┬─────────────────────────┘   │
│             │                     │                 │                             │
│             │                     │                 ▼                             │
│             │                     │  ┌────────────────────────────────────────┐   │
│             │                     │  │  Agent (agent.py)                      │   │
│             │                     │  │                                        │   │
│             │                     │  │  1. Extract last user message          │   │
│             │                     │  │  2. Extract prior recommendation names │   │
│             │                     │  │  3. Build search query                 │   │
│             │                     │  │  4. ──▶ Semantic Search (retrieval.py)│   │
│             │                     │  │  5. Inject catalog context into prompt │   │
│             │                     │  │  6. ──▶ Gemini LLM Call               │   │
│             │                     │  │  7. Parse JSON response (with retry)   │   │
│             │                     │  │  8. Validate URLs against catalog      │   │
│             │                     │  └──────────────┬─────────────────────────┘   │
│             │                     │                 │                             │
│             │  ◀──────────────────│  ChatResponse { reply, recommendations,      │
│             │    JSON Response    │                  end_of_conversation }        │
└─────────────┘                     └──────────────────────────────────────────────┘

                                    ┌──────────────────────────────────────────────┐
                                    │  Vector Store (faiss_index/)                 │
                                    │  ┌──────────────┐  ┌───────────────────┐     │
                                    │  │ index.faiss   │  │  metadata.json    │     │
                                    │  │ 377 vectors   │  │  377 catalog items│     │
                                    │  │ 3072 dims     │  │  with all fields  │     │
                                    │  └──────────────┘  └───────────────────┘     │
                                    └──────────────────────────────────────────────┘
```

### Request Flow
1. Client sends `POST /chat` with full conversation history (`messages[]`)
2. **main.py** validates turn count (≤ 8) and wraps the call with a 25s timeout
3. **agent.py** extracts the latest user message and any prior recommendation names
4. **retrieval.py** embeds the query using `gemini-embedding-001` and performs FAISS inner-product search, returning top-20 catalog matches
5. **agent.py** injects the retrieved catalog items into the system prompt and sends the full conversation history to Gemini
6. Gemini returns a JSON response with reply text, recommendations, and conversation state
7. **agent.py** parses the JSON (retries once on failure), validates every URL against the catalog, and returns the sanitized result
8. **main.py** returns the structured `ChatResponse` to the client

---

## 🛠 Tech Stack

| Component | Technology | Version | Purpose |
|---|---|---|---|
| **Web Framework** | FastAPI | 0.115.0 | Async HTTP API with automatic OpenAPI docs |
| **ASGI Server** | Uvicorn | 0.30.6 | Production ASGI server |
| **LLM** | Google Gemini 2.5 Flash | — | Conversational reasoning & JSON output |
| **Embeddings** | Google gemini-embedding-001 | — | 3072-dim semantic embeddings |
| **Vector Store** | FAISS (faiss-cpu) | 1.8.0 | Fast approximate nearest-neighbor search |
| **Data Validation** | Pydantic | 2.7.4 | Request/response schema enforcement |
| **Numerical** | NumPy | 1.26.4 | Vector normalization |
| **Config** | python-dotenv | 1.0.1 | Environment variable management |
| **Deployment** | Render.com | Free tier | Cloud hosting |

---

## 📂 Project Structure

```
shl_assignment/
├── main.py               # FastAPI application — endpoints & request handling
├── agent.py              # Agent logic — retrieval, LLM orchestration, validation
├── retrieval.py          # CatalogRetriever — FAISS index & semantic search
├── build_index.py        # One-time script — builds FAISS index from catalog
├── catalog.json          # SHL product catalog (377 items, read-only)
├── faiss_index/          # Pre-built vector index (committed to repo)
│   ├── index.faiss       # FAISS IndexFlatIP — 377 vectors × 3072 dims
│   └── metadata.json     # Aligned metadata for each vector
├── requirements.txt      # Pinned Python dependencies
├── render.yaml           # Render.com deployment configuration
├── .env                  # Environment variables (GEMINI_API_KEY)
└── README.md             # This file
```

### File Responsibilities

#### `main.py` — API Layer
- Defines Pydantic models: `Message`, `ChatRequest`, `Recommendation`, `ChatResponse`
- `GET /health` — Returns `{"status": "ok"}` (always fast, no dependencies)
- `POST /chat` — Accepts full conversation history, delegates to agent, enforces 25s timeout and 8-turn limit
- CORS enabled for all origins
- Graceful error handling — never returns 500, always returns valid `ChatResponse`

#### `agent.py` — Brain
- Loads `CatalogRetriever` and `GenerativeModel` at module import (once per server lifetime)
- System prompt encodes all behavioral rules (clarify → recommend → refine → compare)
- `_extract_prior_context()` — Parses prior assistant messages for recommendation names (enables refine/compare)
- `_build_gemini_history()` — Converts messages to Gemini's alternating user/model format
- `_parse_llm_response()` — Strips markdown code fences, parses JSON
- `run_agent()` — Full pipeline: query building → retrieval → prompt injection → LLM call → JSON parse with retry → URL validation

#### `retrieval.py` — Vector Search
- `CatalogRetriever.__init__()` — Loads FAISS index + metadata into memory, builds URL validation set
- `semantic_search(query, top_k=20)` — Embeds query, normalizes, performs FAISS inner-product search
- `format_for_prompt(items)` — Formats retrieved items as `Name | URL | Type | Duration | Description` strings
- `is_valid_url(url)` — O(1) URL validation against catalog

#### `build_index.py` — Index Builder (Run Once)
- Loads catalog with `strict=False` JSON parsing (handles literal newline in "Microsoft \n 365 (New)")
- Fixes newline characters in names: `name.replace("\n", " ").strip()`
- Builds embedding text: `"{name}. {description}. Keys: {keys}. Job levels: {levels}. Languages: {langs}"`
- Embeds in batches of 10 with 8-second delays and exponential backoff for rate limits
- Normalizes vectors with `faiss.normalize_L2()` for inner-product similarity
- Saves `index.faiss` and `metadata.json` to `faiss_index/`

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- A Google Gemini API key ([Get one free](https://aistudio.google.com/apikey))

### Step-by-Step

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd shl_assignment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
# Edit .env file and replace 'your_key_here' with your actual key
echo GEMINI_API_KEY=your_actual_key_here > .env

# 4. Build the FAISS index (one-time, ~5 minutes on free tier)
python build_index.py

# 5. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 6. Verify it's running
curl http://localhost:8000/health
# → {"status": "ok"}
```

> **Note:** The `faiss_index/` directory is pre-built and committed to the repo. Step 4 is only needed if you want to rebuild the index (e.g., after updating catalog.json).

---

## 📡 API Reference

### `GET /health`

Health check endpoint. Always returns HTTP 200 with no external dependencies.

**Response:**
```json
{"status": "ok"}
```

---

### `POST /chat`

Main conversational endpoint. Stateless — full conversation history must be sent with every request.

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "I need to assess senior Java developers"},
    {"role": "assistant", "content": "{\"reply\": \"...\", \"recommendations\": [...], \"end_of_conversation\": false}"},
    {"role": "user", "content": "Can you also add a personality assessment?"}
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `messages` | `Message[]` | Full conversation history |
| `messages[].role` | `string` | `"user"` or `"assistant"` |
| `messages[].content` | `string` | Message text (assistant messages are JSON strings) |

**Response Body:**
```json
{
  "reply": "I've added the OPQ32r personality assessment to your shortlist...",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    },
    {
      "name": "OPQ32r",
      "url": "https://www.shl.com/products/product-catalog/view/opq32r/",
      "test_type": "P"
    }
  ],
  "end_of_conversation": false
}
```

| Field | Type | Description |
|---|---|---|
| `reply` | `string` | Conversational response text |
| `recommendations` | `Recommendation[]` | 0 items (clarifying/refusing) or 1-10 items (recommending) |
| `recommendations[].name` | `string` | Exact assessment name from catalog |
| `recommendations[].url` | `string` | Exact URL from catalog (validated) |
| `recommendations[].test_type` | `string` | Comma-joined abbreviations (see [Test Types](#test-type-abbreviations)) |
| `end_of_conversation` | `boolean` | `true` when conversation is complete |

**Error Handling:**
The API never returns non-200 status codes for chat requests. Instead, errors are communicated through the response:

| Scenario | `reply` | `recommendations` | `end_of_conversation` |
|---|---|---|---|
| Turn limit exceeded (≥8 messages) | "We've reached the conversation limit..." | `[]` | `true` |
| Timeout (>25 seconds) | "Request timed out. Please try again." | `[]` | `false` |
| Internal error | "Something went wrong. Please try again." | `[]` | `false` |

---

### Interactive API Docs

When the server is running, visit:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📦 Catalog Details

### Overview
The catalog (`catalog.json`) contains **377 SHL assessment products**. Each item includes:

| Field | Type | Example |
|---|---|---|
| `name` | string | `"Core Java (Advanced Level) (New)"` |
| `link` | string | `"https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/"` |
| `keys` | string[] | `["Knowledge & Skills"]` |
| `description` | string | `"Multi-choice test that measures..."` |
| `duration` | string | `"13 minutes"` or `""` |
| `languages` | string[] | `["English (USA)"]` |
| `job_levels` | string[] | `["Mid-Professional", "Professional Individual Contributor"]` |
| `remote` | string | `"yes"` or `"no"` |
| `adaptive` | string | `"yes"` or `"no"` |

### Test Type Abbreviations

The `test_type` field in API responses is derived from the `keys` field using first-letter abbreviations:

| Key Category | Abbreviation |
|---|---|
| Ability & Aptitude | **A** |
| Personality & Behavior | **P** |
| Knowledge & Skills | **K** |
| Simulations | **S** |
| Biodata & Situational Judgment | **B** |
| Competencies | **C** |
| Development & 360 | **D** |
| Assessment Exercises | **E** |

Example: An item with `keys: ["Knowledge & Skills", "Simulations"]` → `test_type: "K,S"`

### Known Data Issue
One catalog item has a literal newline character in its name field:
```
"Microsoft \n    365 (New)"  →  cleaned to  →  "Microsoft 365 (New)"
```
This is handled automatically on load with `name.replace("\n", " ").strip()` and `json.loads(data, strict=False)`.

---

## 🔬 RAG Pipeline Deep Dive

### 1. Index Building (Offline)

```
catalog.json (377 items)
        │
        ▼
┌─────────────────────────────────┐
│  For each item, build text:     │
│  "{name}. {description}.        │
│   Keys: {keys}.                 │
│   Job levels: {levels}.         │
│   Languages: {langs}"           │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Embed with gemini-embedding-001│
│  task_type = RETRIEVAL_DOCUMENT │
│  batch_size = 10                │
│  → 377 vectors × 3072 dims     │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Normalize L2 (for cosine sim)  │
│  Build FAISS IndexFlatIP        │
│  Save index.faiss + metadata    │
└─────────────────────────────────┘
```

### 2. Query-Time Retrieval (Online)

```
User message: "I need to assess senior Java developers"
        │
        ▼
┌─────────────────────────────────┐
│  Combine with prior context:    │
│  query = user_msg + prior_names │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Embed with gemini-embedding-001│
│  task_type = RETRIEVAL_QUERY    │
│  Normalize L2                   │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  FAISS inner-product search     │
│  top_k = 20 results             │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Format for prompt injection:   │
│  "Name: X | URL: Y | Type: Z   │
│   | Duration: D | Desc: ..."    │
└─────────────────────────────────┘
```

### 3. LLM Generation

```
┌──────────────────────────────────────┐
│  System Prompt                       │
│  ┌──────────────────────────────────┐│
│  │ Behavioral rules (scope,        ││
│  │ clarify, recommend, refine,     ││
│  │ compare, turn limit)            ││
│  ├──────────────────────────────────┤│
│  │ CATALOG CONTEXT                 ││
│  │ (20 retrieved items formatted)  ││
│  ├──────────────────────────────────┤│
│  │ OUTPUT FORMAT                   ││
│  │ (strict JSON schema)           ││
│  └──────────────────────────────────┘│
│  + Full conversation history         │
│  + Current user message              │
│                                      │
│  ──▶ Gemini 2.5 Flash ──▶ JSON      │
└──────────────────────────────────────┘
```

### 4. Post-Processing

```
LLM JSON Output
        │
        ▼
┌─────────────────────────────────┐
│  Parse JSON                     │
│  (strip code fences if needed)  │
│  If fails → retry once with     │
│  explicit JSON reminder         │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Validate every URL against     │
│  catalog (O(1) set lookup)      │
│  Drop invalid recommendations  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Enforce turn limit             │
│  Return ChatResponse            │
└─────────────────────────────────┘
```

---

## 🤖 Agent Behavior & Conversation Design

The agent follows a strict behavioral protocol derived from 10 conversation traces. Each behavior maps to a real-world hiring workflow:

### Behavior 1 — Clarify Before Recommending

For **vague queries** (no role, no context), the agent asks 1–2 focused questions before recommending:

```
User: "I need assessments for my team"
Agent: "To recommend the most suitable assessments, could you tell me:
        1. What role/job level are you hiring for?
        2. What skills or competencies matter most?"
→ recommendations: []
```

### Behavior 2 — Recommend

Once context is sufficient, the agent recommends **1–10 assessments** from the catalog:

```
User: "I need to hire a senior Java developer. Assess coding skills,
       problem-solving, and personality fit. English only."
Agent: Recommends Core Java, Java Design Patterns, Automata Pro, OPQ32r, etc.
→ recommendations: [{...}, {...}, ...]
```

**Defaults:**
- OPQ32r is included by default for personality (unless user declines)
- SHL Verify Interactive G+ is preferred for cognitive ability in senior roles

### Behavior 3 — Refine

When the user changes constraints mid-conversation, the agent updates **surgically**:

```
User: "Drop the Java Design Patterns test and add a Spring assessment"
Agent: "Done. I've removed Java Design Patterns and added Spring (New)..."
→ recommendations: [updated list]
```

### Behavior 4 — Compare

When asked to compare assessments, the agent uses **only catalog descriptions**:

```
User: "Compare Core Java and Java 8 tests"
Agent: "Core Java covers OOP, collections, threading...
        Java 8 focuses on lambdas, streams, functional interfaces..."
→ recommendations: []  (shortlist unchanged)
```

### Scope Enforcement

The agent refuses off-topic questions with a consistent message:

```
User: "Are we legally required to use validated assessments?"
Agent: "Those are legal questions outside what I can advise on.
        I can help you select assessments from the SHL catalog."
→ recommendations: []
```

### Conversation Ending

`end_of_conversation` is set to `true` when:
- User says "confirmed", "thanks", "perfect", "that's it", "lock it in", etc.
- Conversation reaches 8 total messages (user + assistant combined)

---

## ☁️ Deployment

### Render.com (Primary)

The project includes a `render.yaml` for one-click deployment:

```yaml
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

**Steps:**
1. Push the repo to GitHub (including `faiss_index/`)
2. Go to [Render Dashboard](https://dashboard.render.com/) → **New** → **Web Service**
3. Connect your GitHub repository
4. Render auto-detects `render.yaml` and configures the service
5. Add `GEMINI_API_KEY` in the **Environment** tab
6. Deploy

**Important:** The `faiss_index/` directory (~4.6 MB) must be committed to the repo. It is loaded at startup — there is no index-building step during deployment.

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key |
| `PORT` | Auto | Set by Render automatically |

---

## 🧪 Testing

### Quick Smoke Test

```bash
# Health check
curl http://localhost:8000/health
# → {"status": "ok"}
```

### Conversation Test Examples

**Test 1 — Vague Query (should clarify, NOT recommend):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I need assessments for my team"}]}'
```
Expected: `recommendations: []`, reply asks clarifying questions.

**Test 2 — Specific Query (should recommend immediately):**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I need to hire a senior Java developer. Assess coding skills, problem-solving, and personality. English only."}]}'
```
Expected: `recommendations: [1-10 items]` with valid URLs and test_types.

**Test 3 — Off-Topic Refusal:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What salary should I offer a software engineer?"}]}'
```
Expected: `recommendations: []`, reply refuses and offers to help with assessments.

**Test 4 — Conversation End Signal:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Recommend Java tests"},{"role":"assistant","content":"{\"reply\":\"Here are my recommendations.\",\"recommendations\":[],\"end_of_conversation\":false}"},{"role":"user","content":"Thanks, perfect!"}]}'
```
Expected: `end_of_conversation: true`.

**Test 5 — Turn Limit Enforcement:**
```bash
# Send 8+ messages
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"},{"role":"user","content":"c"},{"role":"assistant","content":"d"},{"role":"user","content":"e"},{"role":"assistant","content":"f"},{"role":"user","content":"g"},{"role":"assistant","content":"h"}]}'
```
Expected: `end_of_conversation: true`, reply mentions conversation limit.

### Verified Test Results

| Test | Status | Observation |
|---|---|---|
| `GET /health` | ✅ Pass | Returns `{"status":"ok"}` immediately |
| Vague query → clarify | ✅ Pass | Empty recommendations, asks focused questions |
| Specific query → recommend | ✅ Pass | 9 recommendations with valid catalog URLs |
| Off-topic → refusal | ✅ Pass | Proper refusal message, empty recommendations |
| User sign-off → end | ✅ Pass | `end_of_conversation: true` |

---

## 💡 Design Decisions

### Why FAISS with Inner Product (not L2)?

We use `IndexFlatIP` (inner product) with L2-normalized vectors, which is mathematically equivalent to cosine similarity. This is the standard approach for semantic search because:
- Cosine similarity measures directional alignment, ignoring magnitude
- L2 normalization + inner product = cosine similarity
- FAISS `IndexFlatIP` is optimized for this pattern

### Why top_k = 20?

The agent retrieves 20 candidates but recommends at most 10. The extra buffer ensures:
- The LLM has enough context to make informed selections
- Edge cases (partial matches, multi-faceted queries) are covered
- The LLM can cite catalog gaps honestly when a perfect match doesn't exist

### Why Stateless?

The API carries no server-side session state. The full conversation history is sent with every request because:
- **Simplicity** — No session management, no database, no expiry logic
- **Scalability** — Any instance can handle any request (horizontally scalable)
- **Render.com compatibility** — Free tier may restart instances at any time
- **Client control** — The client owns conversation state and can implement branching, undo, etc.

### Why Validate URLs After LLM Response?

LLMs can hallucinate URLs even when provided exact catalog data. The post-processing step:
1. Checks every recommended URL against a `set()` of valid catalog URLs (O(1) lookup)
2. Silently drops any recommendation with an invalid URL
3. This is a critical safety net — it ensures the evaluator never receives a fabricated URL

### Why JSON Parse with Retry?

LLMs occasionally wrap JSON in markdown code fences (` ```json ... ``` `) despite instructions not to. The two-phase parsing:
1. First attempt: Strip code fences if present, then parse
2. If that fails: Send an explicit "respond only with JSON" reminder and re-parse
3. If both fail: Return a safe fallback response with empty recommendations

---

## ⚠️ Limitations & Future Work

### Current Limitations
- **Free tier rate limits** — Google's free tier has daily quotas per model. High-traffic usage may require a paid API key.
- **No streaming** — Responses are returned as complete JSON. Streaming would improve perceived latency.
- **Cold start** — FAISS index and model loading on first request adds a few seconds of latency on Render's free tier.
- **Single embedding model** — The index must be rebuilt if the embedding model changes.

### Potential Improvements
- **Hybrid search** — Combine semantic (FAISS) with keyword (BM25) search for better precision on technical terms
- **Response caching** — Cache embedding results for repeated queries
- **Streaming responses** — Use Server-Sent Events for real-time token streaming
- **Analytics** — Log conversation patterns to identify popular assessment categories and gaps
- **Multi-language support** — Detect query language and filter catalog by language availability
- **Frontend UI** — Build a chat interface for direct user interaction

---

## 📄 License

This project was built as an assignment for SHL assessment recommendation.

---

<p align="center">
  Built with ❤️ using FastAPI, FAISS, and Google Gemini
</p>
"# shl_assignment" 
