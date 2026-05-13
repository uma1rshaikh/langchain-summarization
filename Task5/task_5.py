import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

# Document & Vector Store
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS

# LangGraph & Tools
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# SETUP CORE MODELS 
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small", 
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# SETUP RETRIEVER (From Task 3)
loader = TextLoader("ai_intro.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator=" ")
docs = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# DEFINE THE THREE TOOLS

@tool
def info_retriever(query: str):
    """Retrieves relevant text fragments about AI history and breakthroughs from the local document database."""
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

@tool
def text_summarizer(text: str):
    """Summarizes a block of text into exactly 3 concise sentences."""
    prompt = ChatPromptTemplate.from_template("Summarize the following into 3 sentences:\n\n{text}")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})

@tool
def word_counter(text: str):
    """Counts the number of words in a given text string and returns the count."""
    count = len(text.split())
    return f"The summary contains {count} words."

tools = [info_retriever, text_summarizer, word_counter]
llm_with_tools = llm.bind_tools(tools)

# LANGGRAPH LOGIC

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def call_model(state: State):
    return {"messages": [llm_with_tools.invoke(state['messages'])]}

def should_continue(state: State):
    last_message = state['messages'][-1]
    return "action" if last_message.tool_calls else END

workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("action", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("action", "agent")
app = workflow.compile()

# EXECUTION
if __name__ == "__main__":
    query = "Find and summarize text about AI breakthroughs from the document, then tell me the word count."
    print(f"--- Running Complex Multi-Tool Task ---")
    
    inputs = {"messages": [HumanMessage(content=query)]}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"\n[Node: {key}]")
            # Print a snippet of the message to keep console clean
            msg = value['messages'][-1]
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                print(f"Tool Calls: {[t['name'] for t in msg.tool_calls]}")
            else:
                print(f"Content: {msg.content[:200]}...")