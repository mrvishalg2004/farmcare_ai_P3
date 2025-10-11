import google.generativeai as genai
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up the API key
API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

def test_gemini_api():
    try:
        print("Testing Gemini API connection...")
        print(f"Using API key: {API_KEY[:5]}...{API_KEY[-4:]}")
        
        # Try to list available models
        print("\nListing available models:")
        models = genai.list_models()
        for model in models:
            if "generateContent" in model.supported_generation_methods:
                print(f"- {model.name} (supports text generation)")
        
        # Try gemini-pro model
        print("\nTesting gemini-pro model:")
        model_pro = genai.GenerativeModel('gemini-pro')
        response_pro = model_pro.generate_content("Hello, what can you tell me about plant diseases?")
        print(f"gemini-pro response: {response_pro.text[:100]}...")
        
        print("\nAPI test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nAPI Error: {type(e).__name__}: {str(e)}")
        return False

if __name__ == "__main__":
    test_gemini_api()