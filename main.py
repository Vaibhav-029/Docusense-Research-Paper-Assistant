from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import shutil
import uuid
import json
import re
import sqlite3
from datetime import datetime

load_dotenv()

app = FastAPI(title="DocuSense — Research Paper Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = None
llm = None

def get_embeddings():
    global embeddings
    if embeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

def get_llm():
    global llm
    if llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    return llm

def get_vectorstore():
    from langchain_community.vectorstores import Chroma
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=get_embeddings()
    )

# ── Database (SQLite chat history) ─────────────────────────────────────────

def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            human_message TEXT,
            ai_message TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_message(session_id, human_msg, ai_msg):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute(
        'INSERT INTO chat_history (session_id, human_message, ai_message, timestamp) VALUES (?, ?, ?, ?)',
        (session_id, human_msg, ai_msg, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def get_history(session_id, limit=3):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute(
        'SELECT human_message, ai_message FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?',
        (session_id, limit)
    )
    rows = c.fetchall()
    conn.close()
    return [{"human": h, "ai": a} for h, a in reversed(rows)]

def clear_session(session_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('DELETE FROM chat_history WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()

init_db()

# ── Helper functions ─────────────────────────────────────────────────────────

async def generate_summary(chunks, filename):
    sample_content = "\n\n".join([c.page_content for c in chunks[:10]])
    prompt = f"""You are analyzing a document called "{filename}".
Based on the following content, generate a structured summary.

Content:
{sample_content}

Generate a summary in exactly this format:
📌 **Title/Topic**: [what this document is about]
🎯 **Main Purpose**: [what problem it solves or what it covers]
🔑 **Key Concepts**: [3-5 bullet points of main topics covered]
💡 **Key Findings/Points**: [3-5 most important takeaways]
⚠️ **Limitations/Gaps**: [what's missing or limitations mentioned]
🏷️ **Best For**: [who should read this and why]"""
    response = get_llm().invoke(prompt)
    return response.content

def classify_query(question, available_files):
    """Agent decision step - classifies the question before deciding how to retrieve"""
    files_list = ", ".join(available_files) if available_files else "none"

    classify_prompt = f"""You are a routing agent for a document Q&A system.
Available documents: {files_list}

Classify this user question into EXACTLY ONE category:
- "summary" - user wants a summary/overview of a document
- "factual" - user wants a specific fact or detail from a document
- "compare" - user wants to compare two or more documents
- "general" - general conversation, greetings, or unclear intent

Question: {question}

Respond with ONLY the category word, nothing else."""

    response = get_llm().invoke(classify_prompt)
    category = response.content.strip().lower()

    if category not in ["summary", "factual", "compare", "general"]:
        category = "factual"

    return category

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "DocuSense API is running"}

@app.get("/app")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_paper(files: list[UploadFile] = File(...), generate_summaries: bool = True):
    total_pages = 0
    total_chunks = 0
    uploaded_files = []
    all_chunks_by_file = {}

    for file in files:
        temp_path = f"temp_{uuid.uuid4().hex}.pdf"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma

        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        for page in pages:
            page.metadata["source_file"] = file.filename

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(pages)

        Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            persist_directory="./chroma_db"
        )

        os.remove(temp_path)
        total_pages += len(pages)
        total_chunks += len(chunks)
        uploaded_files.append(file.filename)
        all_chunks_by_file[file.filename] = chunks

    summaries = {}
    if generate_summaries:
        for filename, chunks in all_chunks_by_file.items():
            summaries[filename] = await generate_summary(chunks, filename)

    return {
        "message": "Papers uploaded successfully",
        "files": uploaded_files,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "summaries": summaries
    }

@app.post("/ask")
async def ask_question(question: str, session_id: str = "default"):
    history = get_history(session_id)

    vectorstore = get_vectorstore()

    all_docs = vectorstore.get()
    available_files = list(set([
        meta.get("source_file", "unknown")
        for meta in all_docs.get("metadatas", [])
    ])) if all_docs.get("metadatas") else []

    query_type = classify_query(question, available_files)

    if history:
        history_text = "\n".join([
            f"Human: {msg['human']}\nAssistant: {msg['ai']}"
            for msg in history[-3:]
        ])
        rewrite_prompt = f"""Given this conversation history:
{history_text}

And this follow-up question: {question}

Rewrite the follow-up question as a standalone question with full context.
Return only the rewritten question, nothing else."""
        rewritten = get_llm().invoke(rewrite_prompt)
        search_query = rewritten.content
    else:
        search_query = question

    if query_type == "summary":
        docs = vectorstore.similarity_search(search_query, k=8)
        retrieval_note = "Full-document retrieval (summary mode)"

    elif query_type == "compare" and len(available_files) >= 2:
        docs = []
        for filename in available_files:
            file_docs = vectorstore.similarity_search(
                search_query, k=3, filter={"source_file": filename}
            )
            docs.extend(file_docs)
        retrieval_note = "Per-document retrieval (comparison mode)"

    elif query_type == "general":
        docs = vectorstore.similarity_search(search_query, k=2)
        retrieval_note = "Minimal retrieval (general conversation)"

    else:
        docs = vectorstore.similarity_search(search_query, k=3)
        retrieval_note = "Targeted retrieval (factual mode)"

    context = "\n\n".join([doc.page_content for doc in docs])

    history_context = ""
    if history:
        history_context = "\n\nConversation so far:\n" + "\n".join([
            f"Human: {msg['human']}\nAssistant: {msg['ai']}"
            for msg in history[-3:]
        ])

    prompt = f"""You are a document Q&A assistant. You must ONLY answer based on the provided document context below.
Do NOT use any outside knowledge. If the answer is not found in the context, say "I couldn't find this information in the uploaded document(s)."

Document context:
{context}
{history_context}

Question: {question}

Answer (based strictly on the above context):"""

    response = get_llm().invoke(prompt)
    save_message(session_id, question, response.content)

    return {
        "question": question,
        "answer": response.content,
        "query_type": query_type,
        "retrieval_strategy": retrieval_note,
        "sources": [
            {
                "file": doc.metadata.get("source_file", "unknown"),
                "content": doc.page_content[:200]
            }
            for doc in docs
        ],
        "history_length": len(history) + 1
    }

@app.post("/flashcards")
async def generate_flashcards(filename: str, count: int = 10):
    vectorstore = get_vectorstore()

    docs = vectorstore.similarity_search(
        f"key concepts from {filename}", k=10,
        filter={"source_file": filename}
    )
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""Based on this content from "{filename}", generate {count} flashcards for studying.

Content:
{context}

Return ONLY a JSON array in this exact format, no other text:
[
  {{"question": "What is X?", "answer": "X is..."}},
  {{"question": "Define Y", "answer": "Y refers to..."}}
]"""

    response = get_llm().invoke(prompt)
    clean = re.sub(r'```json|```', '', response.content).strip()
    flashcards = json.loads(clean)

    return {
        "filename": filename,
        "flashcards": flashcards,
        "count": len(flashcards)
    }

@app.post("/debate")
async def paper_debate(question: str, paper_a: str, paper_b: str):
    vectorstore = get_vectorstore()

    docs_a = vectorstore.similarity_search(question, k=4, filter={"source_file": paper_a})
    docs_b = vectorstore.similarity_search(question, k=4, filter={"source_file": paper_b})

    context_a = "\n\n".join([doc.page_content for doc in docs_a])
    context_b = "\n\n".join([doc.page_content for doc in docs_b])

    prompt = f"""You are moderating a debate between two documents on this topic: "{question}"

Document A ({paper_a}):
{context_a}

Document B ({paper_b}):
{context_b}

Structure your response exactly like this:

⚔️ **DEBATE: {question}**

📘 **{paper_a} argues:**
[What document A says about this topic, 3-4 points]

📗 **{paper_b} argues:**
[What document B says about this topic, 3-4 points]

🤝 **Common ground:**
[What both documents agree on]

⚡ **Key differences:**
[Where they fundamentally disagree]

🏆 **Verdict:**
[Which document makes a stronger case and why]"""

    response = get_llm().invoke(prompt)

    return {
        "question": question,
        "paper_a": paper_a,
        "paper_b": paper_b,
        "debate": response.content
    }

@app.delete("/clear/{session_id}")
def clear_history_route(session_id: str = "default"):
    clear_session(session_id)
    return {"message": "Chat history cleared"}

@app.delete("/reset")
def reset_vectorstore():
    """Clear all uploaded document data from ChromaDB"""
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    return {"message": "All uploaded documents cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))