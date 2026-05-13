import os
from dotenv import load_dotenv

def verify_environment():
    # Load variables from .env into the environment
    load_dotenv()

    # Retrieve the variables
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    version = os.getenv("AZURE_OPENAI_API_VERSION")

    print("--- Environment Variable Check ---")
    print(f"API Key Loaded: {'Yes' if api_key else 'No'}")
    print(f"Endpoint: {endpoint}")
    print(f"Deployment: {deployment}")
    print(f"API Version: {version}")

if __name__ == "__main__":
    verify_environment()