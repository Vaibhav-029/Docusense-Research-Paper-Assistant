from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import shutil
import uuid

load_dotenv()

app = FastAPI(title="DocuSense — Research Paper Assistant")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Store chat histories per session
chat_histories = {}

@app.get("/")
def root():
    return {"message": "DocuSense API is running"}

@app.get("/app")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_paper(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4().hex}.pdf"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(temp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    os.remove(temp_path)

    return {
        "message": "Paper uploaded successfully",
        "pages": len(pages),
        "chunks": len(chunks)
    }

@app.post("/ask")
async def ask_question(question: str, session_id: str = "default"):
    # Get or create chat history for this session
    if session_id not in chat_histories:
        chat_histories[session_id] = []

    history = chat_histories[session_id]

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # If there's history, rewrite the question with context
    if history:
        history_text = "\n".join([
            f"Human: {msg['human']}\nAssistant: {msg['ai']}"
            for msg in history[-3:]  # last 3 exchanges
        ])

        rewrite_prompt = f"""Given this conversation history:
{history_text}

And this follow-up question: {question}

Rewrite the follow-up question as a standalone question that includes 
all necessary context from the conversation history.
Return only the rewritten question, nothing else."""

        rewritten = llm.invoke(rewrite_prompt)
        search_query = rewritten.content
    else:
        search_query = question

    # Search ChromaDB with the rewritten question
    docs = vectorstore.similarity_search(search_query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build history context for the prompt
    history_context = ""
    if history:
        history_context = "\n\nConversation so far:\n" + "\n".join([
            f"Human: {msg['human']}\nAssistant: {msg['ai']}"
            for msg in history[-3:]
        ])

    prompt = f"""You are a research paper assistant.
Use the following context from the document to answer the question.
Always cite which part of the document your answer comes from.
If the answer builds on previous conversation, acknowledge that naturally.

Document context:
{context}
{history_context}

Current question: {question}

Answer:"""

    response = llm.invoke(prompt)

    # Save to history
    chat_histories[session_id].append({
        "human": question,
        "ai": response.content
    })

    return {
        "question": question,
        "answer": response.content,
        "sources": [doc.page_content[:200] for doc in docs],
        "history_length": len(chat_histories[session_id])
    }

@app.delete("/clear/{session_id}")
def clear_history(session_id: str = "default"):
    if session_id in chat_histories:
        chat_histories[session_id] = []
    return {"message": "Chat history cleared"}