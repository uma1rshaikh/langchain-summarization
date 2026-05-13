import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Use legacy import for the Task-specific LLMChain requirement
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate

load_dotenv()

def main():
    # 1. Initialize Model
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0
    )

    # 2. Load the Original Document (Full Text)
    if not os.path.exists("ai_intro.txt"):
        print("Error: ai_intro.txt not found!")
        return
        
    with open("ai_intro.txt", "r") as f:
        full_text = f.read()

    # 3. Create the Summary (Knowledge Distillation)
    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize the following text into one concise paragraph:\n\n{text}"
    )
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    
    # Store summary in a variable as per task
    ai_summary = summarize_chain.invoke({"text": full_text})
    
    print("--- TASK 10: QA EVALUATION ---")
    print(f"\n[Generated Summary]:\n{ai_summary}")

    # 4. Setup the QA Chain
    qa_template = """Use the following text to answer the question. 
        If the answer is not in the text, say you don't know.

        Text: {text}

        Question: {question}

        Final Answer:"""
    
    qa_prompt = PromptTemplate(input_variables=["text", "question"], template=qa_template)
    qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

    # 5. Run the Experiment
    question = "What’s the key event mentioned?"

    # Test A: QA on Summary
    res_summary = qa_chain.invoke({"text": ai_summary, "question": question})
    
    # Test B: QA on Full Document
    res_full = qa_chain.invoke({"text": full_text, "question": question})

    print("\n" + "="*50)
    print(f"QUESTION: {question}")
    print("="*50)
    print(f"RESPONSE (From Summary):\n{res_summary['text']}")
    print("-" * 50)
    print(f"RESPONSE (From Full Text):\n{res_full['text']}")
    print("="*50)

if __name__ == "__main__":
    main()