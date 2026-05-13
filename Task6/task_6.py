import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate

# NEW: Import from langchain_classic to avoid the ModuleNotFoundError in v0.3
from langchain_classic.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_classic.chains import LLMChain

# Load environment variables
load_dotenv()

def main():
    # 1. Initialize the Azure OpenAI Model
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )

    # 2. Define the Prompt Template
    # The {chat_history} variable is where the memory will be injected
    template = """
    You are an expert AI researcher.
    
    Current Conversation History:
    {chat_history}
    
    New Task: {input}
    
    Response:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "input"], 
        template=template
    )

    # 3. Sample Data (100 words approx each)
    ml_text = """
    Machine learning (ML) is a core subfield of artificial intelligence that focuses on the development 
    of algorithms capable of learning from and making predictions on data. Instead of following 
    strictly static program instructions, ML models use statistical techniques like linear regression, 
    support vector machines, and decision trees to identify complex patterns. It is widely applied in 
    diverse areas such as email spam filtering, financial fraud detection, and personalized 
    recommendation engines. As these systems process more information, their performance typically 
    improves, allowing them to adapt to new data without manual intervention from developers.
    """

    dl_text = """
    Deep learning (DL) is a specialized subset of machine learning that is inspired by the structure 
    and function of the human brain's neural networks. It utilizes multiple layers of artificial 
    neurons—hence the term 'deep'—to extract higher-level features progressively from raw input data. 
    This architecture makes it exceptionally powerful for processing unstructured data like images, 
    video, and natural language. Unlike traditional machine learning, which often requires manual 
    feature engineering by experts, deep learning can automatically learn representations through 
    the process of backpropagation, making it the driving force behind modern AI breakthroughs.
    """

    # 4. Function to execute the test loop
    def run_experiment(memory_type, name):
        print(f"\n{'='*20} {name} {'='*20}")
        
        # Initialize the chain with the specific memory type
        chain = LLMChain(
            llm=llm, 
            prompt=prompt, 
            memory=memory_type, 
            verbose=False
        )

        # Interaction 1: Summarize ML
        print("\n[Step 1: Summarizing Machine Learning]")
        res1 = chain.predict(input=f"Summarize this text about Machine Learning in 3 sentences: {ml_text}")
        print(res1)

        # Interaction 2: Summarize DL with context
        print("\n[Step 2: Summarizing Deep Learning with Contextual Link]")
        query = f"""
        Summarize this text about Deep Learning in 3 sentences. 
        Then, explain how this compares to the Machine Learning summary we just discussed above: 
        {dl_text}
        """
        res2 = chain.predict(input=query)
        print(res2)

    # --- EXECUTION ---

    # Test A: Buffer Memory (The "Literal" Memory)
    # This stores the exact text of previous messages.
    buffer_mem = ConversationBufferMemory(memory_key="chat_history")
    run_experiment(buffer_mem, "ConversationBufferMemory")

    # Test B: Summary Memory (The "Conceptual" Memory)
    # This uses an LLM to summarize the history as it goes to save space/tokens.
    summary_mem = ConversationSummaryMemory(llm=llm, memory_key="chat_history")
    run_experiment(summary_mem, "ConversationSummaryMemory")

if __name__ == "__main__":
    main()