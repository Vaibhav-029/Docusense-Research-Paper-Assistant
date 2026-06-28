# DocuSense — Research Paper Assistant

An AI-powered conversational assistant that lets you upload multiple PDFs and have a natural conversation about them. Features agentic query routing, persistent conversation memory, auto-generated summaries, flashcards, and a unique paper-vs-paper debate mode.

## Features

- 📄 **Multi-PDF upload** — process multiple documents in one session
- 🤖 **Agentic query routing** — classifies your question (summary / factual / comparison / general) and dynamically adapts retrieval strategy for each
- 🧠 **Persistent conversation memory** — SQLite-backed chat history with automatic question rewriting for natural follow-ups
- 📋 **Auto-summary on upload** — structured summary generated automatically for every document
- 🃏 **Flashcard generator** — auto-generates study flashcards from any uploaded document
- ⚔️ **Paper vs Paper debate** — pick two documents and watch them "debate" a topic, with a structured verdict
- 📌 **Source citations** — every answer shows exactly which document and chunk it came from
- 💬 **Clean chat UI** — purpose-built frontend, no extra frameworks

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python |
| RAG Pipeline | LangChain |
| Vector Database | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| LLM | Gemini 2.5 Flash |
| Persistence | SQLite |
| Frontend | HTML, CSS, Vanilla JS |

## Architecture

```
PDF Upload → Chunking (500 tokens) → Embeddings → ChromaDB
                                                     │
User Question → Agent classifies intent ────────────┤
                                                     │
        ┌────────────────┬────────────────┬─────────┴────────┐
   summary mode      factual mode      compare mode      general mode
   (8 chunks,         (3 chunks,        (per-document      (2 chunks,
   full doc)          targeted)         retrieval)         minimal)
        └────────────────┴────────────────┴──────────────────┘
                                  │
                          Gemini LLM + history
                                  │
                       Cited answer + SQLite save
```

### How agentic query routing works

Most RAG systems treat every question identically. DocuSense first classifies the question's intent using the LLM itself, then changes its retrieval behavior based on that classification:

- **Summary** queries pull more chunks (8) for broad coverage
- **Factual** queries use precise, targeted retrieval (3 chunks)
- **Comparison** queries retrieve separately from each document for balanced representation
- **General** conversation uses minimal retrieval

This is a genuine agentic pattern — the system decides *how* to act before acting, rather than applying one fixed pipeline to every input.

### How conversation memory works

Chat history is persisted in SQLite (not in-memory), so conversations survive server restarts. On every follow-up question, the system:
1. Pulls the last 3 exchanges from the database
2. Rewrites the follow-up as a standalone question with full context
3. Retrieves using the rewritten query
4. Answers using both retrieved context and conversation history

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Vaibhav-029/Docusense-Research-Paper-Assistant.git
cd Docusense-Research-Paper-Assistant
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Gemini API key**

Create a `.env` file in the root folder:
```
GEMINI_API_KEY=your_key_here
```
Get a free key at [aistudio.google.com](https://aistudio.google.com)

**5. Run the server**
```bash
uvicorn main:app --reload
```

**6. Open the app**
- Chat UI: `http://localhost:8000/app`
- API docs: `http://localhost:8000/docs`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/app` | Serves the chat UI |
| POST | `/upload` | Upload one or more PDFs, returns auto-summaries |
| POST | `/ask` | Ask a question — routed by intent, with session memory |
| POST | `/flashcards` | Generate flashcards from a specific document |
| POST | `/debate` | Run a structured debate between two documents |
| DELETE | `/clear/{session_id}` | Clear chat history for a session |

## What I'd Improve Next

- Hybrid search (semantic + keyword/BM25) for better recall on exact terms
- Cross-encoder reranking for higher precision on factual queries
- Streaming responses for better perceived latency
- Deployment with a managed vector store for multi-user scale

## Built By

**Vaibhav Srivastava** — Final Year Student, building production-style AI systems with FastAPI, LangChain, and GenAI.

[LinkedIn](linkedin.com/in/vaibhav-srivastava-851927285) · [GitHub](https://github.com/Vaibhav-029)
