import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# 1. Define your schema using Pydantic
class SummarizationOutput(BaseModel):
    summary: str = Field(description="The 3-sentence summary of the input text.")
    length: int = Field(description="The total character count of the summary string.")

def main():
    load_dotenv()

    # 2. Setup the Parser
    parser = PydanticOutputParser(pydantic_object=SummarizationOutput)

    # 3. Setup the Model
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0 
    )

    # 4. Create the Prompt with format instructions
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text into exactly 3 sentences.\n{format_instructions}\n\n{text}"
    )
    
    prompt_with_instructions = prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    # 5. Build the Chain
    chain = prompt_with_instructions | llm | parser

    # 6. Test Data
    ai_apps_text = """
    Artificial intelligence is rapidly expanding across various sectors, significantly impacting how we live and work. 
    In the automotive industry, AI powers self-driving technologies and optimizes traffic flow in smart cities. 
    The financial sector utilizes AI for high-frequency trading and sophisticated fraud detection systems that analyze millions of transactions in real-time. 
    In agriculture, computer vision helps farmers monitor crop health and automate harvesting, leading to higher yields and reduced waste. 
    Retailers use predictive analytics to personalize shopping experiences and manage complex global supply chains. 
    Furthermore, AI is revolutionizing language translation and accessibility, breaking down communication barriers across the globe. 
    As these applications become more sophisticated, the focus is shifting toward creating ethical AI frameworks that ensure safety and fairness 
    for all users while continuing to drive technological innovation.
    """

    print("--- Running Structured Output Task ---")
    
    try:
        # STEP A: Call the LLM exactly once
        result = chain.invoke({"text": ai_apps_text})
        
        print("\n[Raw LLM Output]")
        print(f"Summary: {result.summary}")
        print(f"LLM-Guessed Length: {result.length}")

        # STEP B: Apply the Python correction to the existing object
        # This ensures the length matches the EXACT summary generated above
        result.length = len(result.summary)
        
        print("\n[Final Corrected JSON Object]")
        print(f"Summary: {result.summary}")
        print(f"Actual Length: {result.length}")
        
        # Final Verification
        print(f"\nVerification Success: {result.length == len(result.summary)}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()