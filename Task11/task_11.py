import os
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

# --- 1. Define the Tools ---

@tool
def get_current_date(query: str) -> str:
    """Returns today's current date and time. Use this when the user asks for the date."""
    return f"Today's date is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

@tool
def mock_web_search(query: str) -> str:
    """Performs a web search for the latest news or updates. Use this for 'recent' or 'live' info."""
    return (
        "MOCK SEARCH RESULT: Recent AI trends in 2026 show a massive shift toward 'Small Language Models' (SLMs) "
        "that run locally on edge devices. There is also significant regulatory progress globally, "
        "with new standards for AI-generated content watermarking becoming mandatory across major platforms."
    )

tools = [get_current_date, mock_web_search]

# --- 2. Setup the Agent ---

def main():
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0
    )

    # Creating a prompt that tells the agent how to use tools
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant with access to tools. Be concise."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Construct the OpenAI Functions agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # The Executor handles the loop: LLM -> Tool -> LLM -> Final Answer
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # --- 3. Run Experiments ---

    # Test 1: Summary + Date
    ai_text = (
        "Artificial Intelligence is no longer just a futuristic concept; it is integrated into daily life. "
        "From smart assistants in our homes to advanced diagnostic tools in healthcare, AI improves efficiency. "
        "However, as it scales, concerns regarding bias, privacy, and the environmental cost of training large "
        "models remain at the forefront of the global conversation."
    )

    print("\n--- TEST 1: SUMMARY & DATE ---")
    agent_executor.invoke({
        "input": f"Summarize this text in 2 sentences and tell me today's date: {ai_text}"
    })

    print("\n--- TEST 2: TRENDS & MOCK SEARCH ---")
    agent_executor.invoke({
        "input": "Summarize general AI trends and search for recent updates."
    })

if __name__ == "__main__":
    main()