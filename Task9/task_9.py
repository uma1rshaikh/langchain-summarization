import os
import logging
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# THE SPECIFIC IMPORT THAT WORKS IN V0.3+
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# Enable logging to see the AI generating the 3 variations
logging.basicConfig()
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)

load_dotenv()

def main():
    # 1. Models
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-small", 
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    # 2. Document Setup
    if not os.path.exists("ai_intro.txt"):
        print("ai_intro.txt not found!")
        return

    loader = TextLoader("ai_intro.txt")
    docs = loader.load()
    # Using a separator for better chunking
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50, separator=" ")
    documents = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 3. Multi-Query Retriever Logic
    # This transforms 'AI advancements' into 3 separate search queries
    retriever_multi = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), 
        llm=llm
    )

    # 4. Summarization Chain
    prompt = ChatPromptTemplate.from_template(
        "Based on the following context, summarize the AI advancements in 3 sentences:\n\n{context}"
    )
    chain = prompt | llm | StrOutputParser()

    # 5. Run the experiment
    query = "AI advancements"
    print(f"\n--- Running Task 9: Multi-Query Retrieval ---")
    
    # This generates the queries and fetches chunks for all of them
    retrieved_docs = retriever_multi.invoke(query)
    
    context_text = "\n\n".join([d.page_content for d in retrieved_docs])
    summary = chain.invoke({"context": context_text})
    
    print("-" * 40)
    print(f"Total Unique Chunks Retrieved: {len(retrieved_docs)}")
    print("-" * 40)
    print(f"Summary:\n{summary}")
    print("-" * 40)

if __name__ == "__main__":
    main()