# DocuSense — Research Paper Assistant

An AI-powered research paper assistant that lets you upload any PDF 
and ask questions about it in natural language. Get cited answers 
instantly using RAG (Retrieval Augmented Generation).

## Demo
> Upload any research paper → Ask questions → Get answers with citations

## Tech Stack
- **FastAPI** — REST API backend
- **LangChain** — RAG pipeline orchestration
- **ChromaDB** — Vector database for semantic search
- **Gemini 2.5 Flash** — LLM for answer generation
- **HuggingFace Embeddings** — all-MiniLM-L6-v2
- **Python** — Core language

## Architecture
PDF Upload → Chunking → Embeddings → ChromaDB → Semantic Search → Gemini LLM → Cited Answer

## Features
- Upload any PDF research paper
- Ask questions in natural language
- Get answers with source citations
- Clean web UI for easy interaction
- REST API with auto-generated docs

## How to Run Locally

**1. Clone the repo**
git clone https://github.com/Vaibhav-029/Docusense-Research-Paper-Assistant.git

cd Docusense-Research-Paper-Assistant

**2. Create virtual environment**
python -m venv venv

venv\Scripts\activate

**3. Install dependencies**
pip install -r requirements.txt

**4. Add your Gemini API key**
Create a `.env` file:
GEMINI_API_KEY=your_key_here

**5. Run the server**
uvicorn main:app --reload

**6. Open the app**
- UI: http://localhost:8000/app
- - API docs: http://localhost:8000/docs

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check |
| POST | /upload | Upload a PDF |
| POST | /ask | Ask a question |

## Built By
Vaibhav — Final Year Student
[LinkedIn](linkedin.com/in/vaibhav-srivastava-851927285) | [GitHub](https://github.com/Vaibhav-029)
