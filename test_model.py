import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up the API key
API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

try:
    # Try the gemini-pro-latest model
    print("Testing gemini-pro-latest model...")
    model = genai.GenerativeModel('gemini-pro-latest')
    response = model.generate_content("Hello, tell me about plant diseases")
    print(f"Response from API: {response.text[:100]}...")
    print("API working correctly.")
except Exception as e:
    print(f"API Error: {type(e).__name__}: {str(e)}")
    
    # Try alternative models
    try:
        print("\nTrying gemini-flash-latest model...")
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content("Hello, tell me about plant diseases")
        print(f"Response from API: {response.text[:100]}...")
        print("API working correctly.")
    except Exception as e:
        print(f"API Error: {type(e).__name__}: {str(e)}")