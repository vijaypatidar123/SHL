# SHL Assessment Recommender

A conversational AI assistant that recommends the right SHL assessments for any hiring role. Built on a catalog of 377 SHL Individual Test Solutions with hybrid BM25 + semantic retrieval and a two-pass LLM pipeline.

---

## What It Does

You describe a role you're hiring for — job title, seniority, skills needed — and the assistant recommends the most relevant SHL assessments from the official catalog. Each recommendation links directly to the SHL product page.

---

## Quick Start (Local)

### 1. Prerequisites
- Python 3.10 or higher
- An [OpenRouter](https://openrouter.ai) API key (free tier works)

### 2. Install

```bash
git clone https://github.com/YOUR_USERNAME/shl-recommender.git
cd shl-recommender

python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure

Copy the example env file and add your key:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENROUTER_API_KEY=sk-or-your-actual-key-here
OPENROUTER_MODEL=openai/gpt-oss-120b:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### 4. Run

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000** in your browser — the chat UI will load.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check — returns `{"status":"ok"}` |
| POST | `/chat` | Main chat endpoint |
| GET | `/docs` | Interactive Swagger UI |

### POST /chat — Request format

```json
{
  "messages": [
    { "role": "user", "content": "I need assessments for a senior Java engineer" }
  ]
}
```

Every request must include the **full conversation history** — the server is stateless.

### POST /chat — Response format

```json
{
  "reply": "For a senior Java backend role, here are the assessments I recommend...",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/solutions/products/...",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

---

## How to Use the Chat — User Guide

### The Basics

The assistant works best when you give it **role context**. You don't need to be precise — just describe who you're hiring and what matters for the job.

**You can ask about:**
- Any job role or function
- Specific skills or tools (Java, Excel, SQL, customer service, etc.)
- Seniority level (graduate, mid-level, senior, executive)
- Purpose (hiring/selection vs. development of existing employees)
- Industry-specific knowledge (HIPAA, financial accounting, safety, etc.)

**The assistant will NOT:**
- Invent assessments that don't exist in the SHL catalog
- Give legal or compliance advice
- Answer questions unrelated to SHL assessments

---

### Conversation Tips

#### Be specific about the role
> "I'm hiring a senior data analyst with Python and SQL skills"

is better than:

> "I need some tests"

#### Mention seniority
> "Graduate entry-level", "mid-level manager", "C-suite executive"

#### State your purpose
> "This is for selection" or "We want to develop our existing team"

#### Add skills for technical roles
> "They need to know Java, Spring, Docker, and AWS"

#### Refine as you go
You can add, remove, or swap assessments across turns:
> "Can you also add a personality test?"
> "Remove the cognitive test, we don't need that"
> "What if I also need something for SQL?"

#### Compare two assessments
> "What's the difference between OPQ32r and Verify G+?"

#### Confirm when done
> "Perfect, that's everything" or "Lock it in"

---

### Example Conversations

#### Example 1 — Simple (2 turns)
```
You:  I need to hire a warehouse operative. Safety awareness is most important.
Bot:  [recommends safety and reliability assessments]

You:  Include a personality test as well. That's all we need.
Bot:  [adds personality assessment, confirms shortlist]
```

#### Example 2 — Technical role (3 turns)
```
You:  Senior backend engineer. Skills: Java advanced, Spring, SQL, AWS, Docker.
Bot:  [recommends Core Java, Spring, SQL, AWS assessments]

You:  Add a cognitive ability test like Verify G+.
Bot:  [adds SHL Verify Interactive G+]

You:  Also add OPQ for culture fit. That's everything.
Bot:  [adds OPQ32r, closes conversation]
```

#### Example 3 — Development (not hiring)
```
You:  We want to develop our existing director-level leaders, not hire new ones.
Bot:  [recommends development-focused tools like OPQ UCR, 360 reports]

You:  Include a 360-degree feedback tool as well.
Bot:  [adds Global Skills Development Report, confirms]
```

---

### Assessment Type Reference

When you see a type code on a recommendation card, here's what it means:

| Code | Type |
|------|------|
| A | Ability & Aptitude |
| B | Biodata & Situational Judgement |
| C | Competencies |
| D | Development & 360 Feedback |
| E | Assessment Exercises |
| K | Knowledge & Skills |
| P | Personality & Behaviour |
| S | Simulations |
| T | Job-Specific Assessments |

---

## Project Structure

```
shl-recommender/
├── app/
│   ├── main.py          # FastAPI app, /health and /chat endpoints
│   ├── agent.py         # Two-pass LLM pipeline (planner + writer)
│   ├── retrieval.py     # Hybrid BM25 + dense retrieval
│   ├── catalog.py       # Catalog loader and helpers
│   ├── llm.py           # OpenRouter API client
│   ├── prompts.py       # Planner / writer / compare prompt templates
│   ├── schemas.py       # Pydantic request/response models
│   └── static/
│       └── index.html   # Chat UI
├── data/
│   ├── catalog.json         # 377 SHL Individual Test Solutions
│   └── embeddings/
│       └── corpus.npy       # Pre-computed sentence-transformer embeddings
├── scripts/
│   ├── eval_traces.py       # Recall@10 evaluation harness
│   ├── smoke_agent.py       # Quick smoke test (no LLM needed)
│   └── build_embeddings.py  # Regenerate corpus.npy
├── scraper/
│   └── scrape.py            # SHL catalog scraper
├── traces/                  # 10 evaluation conversation traces
├── .env.example
├── render.yaml              # Render deployment config
└── requirements.txt
```

---

## Running the Eval Harness

To measure Recall@10 against the 10 sample traces:

```bash
python -m scripts.eval_traces
```

To run a quick smoke test without an API key:

```bash
python -m scripts.smoke_agent
```

---

## Deployment (Render)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo — Render auto-detects `render.yaml`
4. Add `OPENROUTER_API_KEY` as an environment variable in the Render dashboard
5. Deploy — your public URL will be `https://shl-recommender.onrender.com`

> **Note:** The free Render tier has a cold-start delay of ~30s after inactivity while the sentence-transformer model loads. Subsequent requests are fast.
