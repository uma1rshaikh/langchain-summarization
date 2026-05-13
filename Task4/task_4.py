import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Union
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Setup State and Tools
class State(TypedDict):
    # The 'add_messages' part tells LangGraph to APPEND new messages
    # to the existing list instead of replacing the whole list.
    messages: Annotated[list[BaseMessage], add_messages]

load_dotenv()
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# Custom Tool using the @tool decorator
@tool
def text_summarizer(text: str):
    """Summarizes a block of text into exactly 3 concise sentences."""
    # We can nest our Task 2 logic inside here
    prompt = ChatPromptTemplate.from_template("Summarize into 3 sentences:\n\n{text}")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})

tools = [text_summarizer]
llm_with_tools = llm.bind_tools(tools)

# Logic Nodes
def call_model(state: State):
    messages = state['messages']
    # Ensure the model sees the FULL history including the tool call request
    response = llm_with_tools.invoke(messages)
    
    # We return a list so that LangGraph's Annotated[list, add_messages] 
    # (if used) or our manual logic appends it correctly.
    return {"messages": [response]}

def should_continue(state: State):
    last_message = state['messages'][-1]
    # If the LLM called a tool, go to the "action" node
    if last_message.tool_calls:
        return "action"
    # We are done
    return END

# 4. Build the Graph
workflow = StateGraph(State)

# Add nodes 
workflow.add_node("agent", call_model)
workflow.add_node("action", ToolNode(tools))

# Add edges
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("action", "agent") # After tool use, go back to agent to summarize results

app = workflow.compile()

def run_test(query):
    print(f"\n--- Query: {query} ---")
    inputs = {"messages": [HumanMessage(content=query)]}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Output from Node '{key}':")
            print(value)

# Test 1: Healthcare
healthcare_text = "Artificial Intelligence is fundamentally transforming the healthcare industry by enabling faster and more accurate diagnoses through advanced medical imaging algorithms. Machine learning models can now predict patient outcomes and suggest personalized treatment plans by analyzing vast repositories of electronic health records. Additionally, AI-powered robotics are increasingly assisting surgeons in complex procedures, significantly reducing human error and patient recovery times. Wearable devices and IoT sensors track real-time health data, allowing for proactive medical intervention before emergencies occur. Overall, the integration of AI in medicine is improving operational efficiency, reducing systemic costs, and ultimately saving lives across the global healthcare landscape"
run_test(f"Summarize this impact of AI on healthcare: {healthcare_text}")

# Test 2: Vague
run_test("Summarize something interesting.")