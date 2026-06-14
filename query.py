from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def query_paper(question):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    prompt = f"""You are a research paper assistant. 
Use the following context from the paper to answer the question.
Always mention which part of the document your answer comes from.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {response.content}")
    print("\nSources used:")
    for i, doc in enumerate(docs):
        print(f"\n--- Source {i+1} ---")
        print(doc.page_content[:200])

if __name__ == "__main__":
    query_paper("What is pydantic and how do I install it?")