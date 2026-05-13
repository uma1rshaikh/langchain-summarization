import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def main():
    load_dotenv()
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0 
    )

    # 3 Sentences
    prompt_3_sentences = ChatPromptTemplate.from_template(
        "Summarize the following text into exactly 3 sentences:\n\n{text}"
    )

    # Create the Chain (new format -> using LCEL: Prompt | Model | Parser)
    chain_3 = prompt_3_sentences | llm | StrOutputParser()

    # Sample text for testing
    ai_paragraph = """
    Artificial Intelligence (AI) represents a branch of computer science dedicated to creating systems 
    capable of performing tasks that typically require human intelligence. This encompasses a vast array 
    of technologies, from machine learning algorithms that identify patterns in big data to natural 
    language processing systems that power modern virtual assistants. In recent years, generative AI 
    has gained significant traction, enabling machines to create original content, such as images, 
    music, and text, by learning from massive datasets. While the potential for AI to revolutionize 
    industries like healthcare, finance, and education is immense, it also raises ethical concerns 
    regarding data privacy, algorithmic bias, and job displacement. Engineers and researchers are 
    now tasked with not only advancing the technical capabilities of these models but also ensuring 
    they are transparent and aligned with human values. As we move forward, the integration of AI 
    into daily life is expected to deepen, fundamentally altering how we interact with technology 
    and each other.
    """

    print("3-Sentence Summary")
    result_3 = chain_3.invoke({"text": ai_paragraph})
    print(result_3)

    # 4. Modify for 1-Sentence Summary
    print("1-Sentence Summary")
    prompt_1_sentence = ChatPromptTemplate.from_template(
        "Summarize the following text into exactly 1 sentence:\n\n{text}"
    )
    chain_1 = prompt_1_sentence | llm | StrOutputParser()
    
    result_1 = chain_1.invoke({"text": ai_paragraph})
    print(result_1)

if __name__ == "__main__":
    main()