import os
from dotenv import load_dotenv

# Document Loading & Splitting
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Vector Store & Embeddings
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS

# LCEL Components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def main():
    load_dotenv()

    # 1. Load the Text File
    loader = TextLoader("ai_intro.txt")
    documents = loader.load()

    # 2. Split into Chunks
    # Task requirement: 200 chars size, 20 chars overlap
    text_splitter = CharacterTextSplitter(
        chunk_size=200, 
        chunk_overlap=20,
        separator=" " # Ensures we don't split in the middle of words
    )
    docs = text_splitter.split_documents(documents)

    # 3. Initialize Embeddings & Vector Store
    # Note: Use your "text-embedding-3-small" deployment here
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-small", 
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )
    
    # Create the in-memory FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 4. Build the Retriever
    retriever = vectorstore.as_retriever()

    # 5. Build the Summarization Chain (from Task 2)
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        "Summarize the following retrieved context into exactly 3 sentences:\n\n{context}"
    )

    # The RAG Chain: 
    # - Retrieve context based on the question
    # - Pass context to prompt
    # - Pass prompt to LLM
    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs))}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Query and Run
    query = "AI milestones"
    print(f"--- Summary based on retrieval for: '{query}' ---")
    print(rag_chain.invoke(query))

if __name__ == "__main__":
    main()