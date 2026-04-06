import google.generativeai as genai
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up the API key
API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# Print environment info
print(f"Python version: {sys.version}")
print(f"Google Generative AI version: {genai.__version__}")
print(f"API Key (masked): {API_KEY[:6]}...{API_KEY[-4:]}")
print("\n---\n")

def test_models():
    # List of models to try
    models_to_try = [
        "gemini-flash-latest", 
        "gemini-pro-latest",
        "gemini-2.0-pro-exp"
    ]
    
    successes = []
    failures = []
    
    for model_name in models_to_try:
        print(f"\nTesting model: {model_name}")
        try:
            # Create the model instance
            model = genai.GenerativeModel(model_name)
            
            # Test with a simple prompt
            prompt = "Write a short paragraph about plant diseases."
            print("  Sending request...")
            
            # Try with safety settings disabled for testing
            response = model.generate_content(
                prompt,
                safety_settings={
                    "HARASSMENT": "block_none",
                    "HATE": "block_none",
                    "SEXUAL": "block_none",
                    "DANGEROUS": "block_none",
                }
            )
            
            print(f"  ✅ Success! Response received.")
            print(f"  First 100 chars: {response.text[:100]}...")
            successes.append(model_name)
        except Exception as e:
            print(f"  ❌ Error: {type(e).__name__}: {str(e)}")
            failures.append((model_name, str(e)))
    
    print("\n---\nSummary:")
    print(f"Successful models: {', '.join(successes) if successes else 'None'}")
    print(f"Failed models: {len(failures)}")
    for model_name, error in failures:
        print(f"  - {model_name}: {error}")
    
    # If any model succeeded, provide implementation code
    if successes:
        print("\n---\nImplementation code:")
        print(f"""
# Add this to your app.py:

def init_gemini_model():
    \"\"\"Initialize and return the Gemini AI model\"\"\"
    model = genai.GenerativeModel('{successes[0]}')
    return model

def generate_gemini_response(user_input):
    \"\"\"Generate a response from Gemini AI model\"\"\"
    model = init_gemini_model()
    
    try:
        # Simple direct request without system prompt
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        print(f"Error: {{type(e).__name__}}: {{str(e)}}")
        return "I'm sorry, I couldn't generate a response at this time."
""")

if __name__ == "__main__":
    test_models()