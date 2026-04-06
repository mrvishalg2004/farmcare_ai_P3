import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using API key (first 5 chars): {api_key[:5]}...")

# Configure the Gemini AI
genai.configure(api_key=api_key)

try:
    # Try listing available models
    print("Listing available models...")
    models = genai.list_models()
    for model in models:
        if "generateContent" in model.supported_generation_methods:
            print(f"- {model.name}")
    
    # Try a simple query with gemini-flash-latest
    print("\nTesting gemini-flash-latest model...")
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content("What can you tell me about plant diseases?")
        print(f"Response: {response.text[:100]}...")
    except Exception as e:
        print(f"Error with gemini-flash-latest: {e}")
        
        # Try with gemini-pro
        print("\nTrying gemini-pro model...")
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("What can you tell me about plant diseases?")
        print(f"Response: {response.text[:100]}...")
    
    print("\nAPI test completed")
except Exception as e:
    print(f"Error: {e}")