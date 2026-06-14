from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import shutil
import uuid

load_dotenv()

app = FastAPI(title="Research Paper Assistant API")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

@app.get("/")
def root():
    return {"message": "Research Paper Assistant is running"}
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
async def ask_question(question: str):
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""You are a research paper assistant.
Use the following context to answer the question.
Always cite which part of the document your answer comes from.

Context:
{context}

Question: {question}

Answer:"""
    
    response = llm.invoke(prompt)
    
    return {
        "question": question,
        "answer": response.content,
        "sources": [doc.page_content[:200] for doc in docs]
    }