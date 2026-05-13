import os
import requests
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def setup_vector_store(loader, embeddings):
    """Helper to load, split, and index documents."""
    data = loader.load()
    # Task constraint: 150 chars, 30 overlap
    splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=30, separator=" ")
    chunks = splitter.split_documents(data)
    return FAISS.from_documents(chunks, embeddings)

def main():
    # 1. Configuration
    pdf_url = "https://ieai.mcts.tum.de/wp-content/uploads/2020/04/White-Paper_AI-Ethics-and-Governance-_March-20201.pdf"
    web_url = "https://www.capgemini.com/insights/research-library/top-tech-trends-of-2026/"
    
    print("Downloading PDF...")
    response = requests.get(pdf_url)
    with open("ai_ethics.pdf", "wb") as f:
        f.write(response.content)

    # 2. Initialize Models
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-small",
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0
    )

    # 3. Create Separate Vector Stores
    print("Indexing PDF and Webpage...")
    pdf_store = setup_vector_store(PyPDFLoader("ai_ethics.pdf"), embeddings)
    web_store = setup_vector_store(WebBaseLoader(web_url), embeddings)

    # 4. Define the Chain
    prompt = ChatPromptTemplate.from_template(
        "Summarize the retrieved context regarding 'AI challenges' in 3 sentences:\n\n{context}"
    )
    summarizer = prompt | llm | StrOutputParser()

    # 5. Query and Compare
    query = "AI challenges"
    
    print(f"\n--- PDF Source Summary ({query}) ---")
    pdf_context = pdf_store.as_retriever().invoke(query)
    pdf_summary = summarizer.invoke({"context": "\n".join([d.page_content for d in pdf_context])})
    print(pdf_summary)

    print(f"\n--- Web Source Summary ({query}) ---")
    web_context = web_store.as_retriever().invoke(query)
    web_summary = summarizer.invoke({"context": "\n".join([d.page_content for d in web_context])})
    print(web_summary)

if __name__ == "__main__":
    main()