import os
import re
import time
from typing import List, Dict

import streamlit as st
import streamlit.components.v1 as components
from langdetect import detect, DetectorFactory
import google.generativeai as genai

# Ensure deterministic language detection
DetectorFactory.seed = 0


def detect_language(text: str) -> str:
    """Detect the ISO 639-1 language code from input text; default to 'en'."""
    try:
        code = detect(text)
        return code or "en"
    except Exception:
        return "en"


def get_system_prompt() -> str:
    """Return the FarmCare AI system prompt guiding tone and behavior."""
    return (
        "You are FarmCare AI, a friendly agricultural assistant that responds instantly and clearly. "
        "Your users are farmers and agri-enthusiasts seeking crop guidance, disease prevention, and simple, practical tips. "
        "Keep answers concise, step-by-step, and actionable. Prefer low-cost, locally doable solutions first. "
        "Avoid medical claims. Do not invent facts; if unsure, say so briefly and suggest consulting a local expert. "
        "Always reply in the user's language and maintain a warm, supportive tone."
    )


def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)


def check_api_key(api_key: str, show_status: bool = True) -> Dict:
    """
    Check if API key is valid with minimal API usage
    Uses the standalone api_key_check.py or falls back to direct check
    """
    try:
        # Try to import the standalone checker first
        try:
            from api_key_check import check_gemini_api_key
            result = check_gemini_api_key(api_key)
            if show_status and result["success"]:
                st.sidebar.success(f"🔑 API Key validated successfully!")
            return result
        except ImportError:
            # Fall back to direct check if standalone checker not available
            genai.configure(api_key=api_key)
            
            try:
                # Try to list models - lightest API call
                models = list(genai.list_models())
                if show_status:
                    st.sidebar.success(f"🔑 API Key validated successfully! ({len(models)} models available)")
                return {"success": True, "models_count": len(models)}
            except Exception as list_err:
                # Try to get a model if listing fails
                model = genai.GenerativeModel('gemini-pro')
                if show_status:
                    st.sidebar.success("🔑 API Key validated successfully!")
                return {"success": True}
    except Exception as e:
        error_message = str(e)
        error_type = type(e).__name__
        
        if show_status:
            st.sidebar.error(f"❌ API Key Invalid: {error_type}")
        
        # Return detailed error information
        return {
            "success": False, 
            "error": error_message,
            "error_type": error_type
        }


@st.cache_resource(show_spinner=False)
def get_gemini_model(model_name: str):
    """Get Gemini model by name with improved error handling"""
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        # More detailed error information
        error_type = type(e).__name__
        if "API_KEY_INVALID" in str(e):
            st.warning("Your API key appears to be invalid. Please check it in the sidebar.")
        elif "not found" in str(e).lower():
            st.warning(f"Model '{model_name}' not found. Try selecting a different model.")
        return None


def format_chat_history(history: List[Dict]) -> List[Dict]:
    """Convert Streamlit chat history to Gemini-compatible contents."""
    contents: List[Dict] = []
    for item in history:
        role = "user" if item.get("role") == "user" else "model"
        text = item.get("content", "")
        contents.append({"role": role, "parts": [{"text": text}]})
    return contents


def main():
    st.set_page_config(page_title="FarmCare AI - Multilingual Chatbot", page_icon="🌾", layout="centered")

    st.markdown("""
    <style>
    .stChatFloatingInputContainer { bottom: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🌾 FarmCare AI")
    st.caption("Instant crop guidance, disease tips, and best practices — in your language.")

    # API key handling (secrets/env/sidebar)
    with st.sidebar:
        st.subheader("Settings")
        model_name = st.selectbox(
            "Model",
            [
                "gemini-pro",  # Most reliable model
                "gemini-1.0-pro",
                "gemini-1.5-pro", 
                "gemini-1.5-flash"
            ],
            index=0,
        )
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1)
        st.markdown("---")
        # Key sources: Streamlit secrets -> ENV -> sidebar input (handle missing secrets safely)
        try:
            secrets_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            secrets_key = None
        env_key = os.environ.get("GEMINI_API_KEY")
        
        # Clear input field and get fresh key
        input_key = st.text_input(
            "Gemini API Key (required)", 
            value="",
            type="password",
            placeholder="Paste your API key here (starts with AIza...)",
            help="Get your free API key from Google AI Studio"
        )
        
        st.caption("🔑 The app will use: Secrets > Environment > Your Input")
        
        # Show API key instructions
        with st.expander("💡 How to get a Gemini API Key", expanded=not input_key):
            st.markdown("""
            1. **Visit**: [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. **Sign in** with your Google account
            3. **Click** "Create API Key" 
            4. **Copy** the key (starts with `AIza`)
            5. **Paste** it in the field above
            
            ⚠️ **Note**: Keep your API key private and secure!
            """)
        
        # Debug info (only show key length for security)
        if input_key:
            st.success(f"✅ API Key entered (length: {len(input_key)} chars)")
        elif secrets_key:
            st.info(f"🔐 Using secrets key (length: {len(secrets_key)} chars)")
        elif env_key:
            st.info(f"🌍 Using environment key (length: {len(env_key)} chars)")
        else:
            st.warning("⚠️ No API key provided yet")

    # Resolve effective key with better priority
    effective_key = input_key or secrets_key or env_key
    
    if not effective_key:
        st.error("🚨 **Please provide your Gemini API Key**")
        st.stop()

    # Clean and validate API key
    effective_key = effective_key.strip()
    
    # Validate API key format
    if not effective_key.startswith('AIza'):
        st.error("🚨 **Invalid API Key Format**")
        st.markdown("""
        **Your API key should:**
        - Start with `AIza`
        - Be about 39 characters long
        - Look like: `AIzaSyB1234567890abcdefghijklmnopqrstuv`
        
        Please get a new key from [Google AI Studio](https://makersuite.google.com/app/apikey)
        """)
        st.stop()
    
    if len(effective_key) < 35:
        st.error("🚨 **API Key too short**")
        st.markdown("Your API key seems incomplete. Please copy the full key from Google AI Studio.")
        st.stop()

    # API Key Management - Store and switch between multiple keys
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = []
        st.session_state.current_key_index = 0
    
    # Add new key if not already in list
    if effective_key not in st.session_state.api_keys:
        st.session_state.api_keys.append(effective_key)
        st.session_state.current_key_index = len(st.session_state.api_keys) - 1
    
    # If multiple keys available, show key rotation option
    if len(st.session_state.api_keys) > 1:
        with st.sidebar.expander("🔑 API Key Management", expanded=False):
            st.info(f"You have {len(st.session_state.api_keys)} API keys available")
            st.markdown("If you hit quota limits, you can switch keys:")
            if st.button("↪️ Rotate to Next API Key"):
                st.session_state.current_key_index = (st.session_state.current_key_index + 1) % len(st.session_state.api_keys)
                effective_key = st.session_state.api_keys[st.session_state.current_key_index]
                st.success(f"✅ Switched to key #{st.session_state.current_key_index + 1}")
                st.experimental_rerun()
    
    # Show which key is being used
    if input_key:
        st.sidebar.success(f"✅ Using your entered API key (#{st.session_state.current_key_index + 1})")
    elif secrets_key:
        st.sidebar.info("🔐 Using secrets API key")
    elif env_key:
        st.sidebar.info("🌍 Using environment API key")
        
    # Add conversation management options
    st.sidebar.divider()
    st.sidebar.subheader("💬 Conversation")
    
    # Add clear conversation button
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        if st.sidebar.button("🗑️ Clear Conversation", key="clear_convo_sidebar"):
            st.session_state.messages = []
            st.sidebar.success("✅ Conversation cleared!")
            time.sleep(0.5)  # Short pause to show success message
            st.experimental_rerun()
    else:
        st.sidebar.info("Start a new conversation by typing a message")

    configure_gemini(effective_key)
    
    # Test API key with a simple call
    try:
        # Use most reliable model for validation - no actual content generation
        test_model = genai.GenerativeModel('gemini-pro')
        
        # Try to list models first - much lighter API call
        try:
            # Get model list to verify key works (no content generation)
            models_response = genai.list_models()
            model_count = len(list(models_response))
            if model_count > 0:
                st.sidebar.success(f"🔑 API Key validated successfully! ({model_count} models available)")
            else:
                st.sidebar.warning("⚠️ API key works but no models are accessible")
        except Exception as list_error:
            # Warn but don't fail if model listing fails
            st.sidebar.warning("⚠️ API Key accepted but model listing failed")
            st.sidebar.info(f"Continuing with default models - error: {type(list_error).__name__}")
            
    except Exception as e:
        error_message = str(e)
        error_type = type(e).__name__
        
        st.error(f"🚨 **API Key Issue: {error_type}**")
        
        if "API_KEY_INVALID" in error_message:
            st.markdown("""
            **Your API key is not working. Please:**
            1. Check if you copied the full key correctly
            2. Generate a new key at [Google AI Studio](https://makersuite.google.com/app/apikey)  
            3. Make sure you're signed into the correct Google account
            4. Verify the API key hasn't expired
            """)
        elif "403" in error_message:
            st.markdown("""
            **Access denied. Please:**
            1. Enable the Generative AI API in Google Cloud Console
            2. Make sure billing is set up (free tier available)
            3. Check API quotas and limits
            """)
        elif "429" in error_message and "quota exceeded" in error_message.lower():
            st.markdown("""
            **API quota exceeded. Please:**
            1. Wait a few minutes before trying again
            2. Create a new API key
            3. Set up billing for higher quotas
            """)
        else:
            with st.expander("Error Details"):
                st.code(error_message)
            
        st.stop()
    
    # Preflight: validate key and API enablement by listing models  
    try:
        models_response = genai.list_models()
        
        # List available generation models
        generation_models = []
        for model in models_response:
            # Check if model supports text generation
            if hasattr(model, 'supported_generation_methods') and 'generateContent' in model.supported_generation_methods:
                model_id = model.name.split('/')[-1]
                generation_models.append(model_id)
        
        if not generation_models:
            # If no generation models are found, use safe defaults
            generation_models = ["gemini-pro"]
            st.sidebar.warning("⚠️ No generation models found, using default models")
        
        # Check if selected model is available for generation
        if model_name not in generation_models:
            # Find the best available alternative
            preferred_models = ["gemini-pro", "gemini-1.0-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
            for preferred in preferred_models:
                if preferred in generation_models:
                    model_name = preferred
                    st.sidebar.warning(f"⚠️ Selected model not available. Using {model_name} instead.")
                    break
            else:
                # Use the first available generation model
                model_name = generation_models[0]
                st.sidebar.warning(f"⚠️ Using available model: {model_name}")
        
        st.sidebar.info(f"📡 Connected! Using model: {model_name}")
        
        # Show available models for debugging
        with st.sidebar.expander("🔍 Available Text Generation Models", expanded=False):
            for model in generation_models:
                st.write(f"• {model}")
        
        # Add quota and pricing info
        with st.sidebar.expander("ℹ️ About API Quotas & Pricing", expanded=False):
            st.markdown("""
            **Free Tier Limits:**
            - 60 queries per minute
            - Limited daily requests per model
            - [Full quota details](https://ai.google.dev/gemini-api/docs/rate-limits)
            
            **Common Quota Errors:**
            - **429 Error**: Rate limit exceeded
            - **"Quota exceeded"**: Daily limit reached
            
            **Solutions for Quota Errors:**
            1. Wait a few minutes between requests
            2. Try multiple API keys (add a new one above)
            3. Set up billing in Google Cloud for paid tier
            
            [See pricing details](https://ai.google.dev/pricing)
            """)
                
    except Exception as e:
        st.error(f"⚠️ **Connection Error:** {str(e)}")
        st.markdown(f"```{str(e)}```")
        st.stop()

    model = get_gemini_model(model_name)

    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {role: 'user'|'assistant', content: str}

    # Render past messages
    for msg in st.session_state.messages:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["content"]) 

    # Chat input
    user_input = st.chat_input("Type your question (Marathi, Hindi, English, etc.)…")
    if user_input:
        # Detect language and prepend a brief instruction for the model
        lang_code = detect_language(user_input)
        sys_prompt = get_system_prompt()

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.write("Thinking…")

            # Build contents with system guidance + chat history + user turn
            contents = [
                {"role": "user", "parts": [{"text": f"SYSTEM:\n{sys_prompt}\nLANGUAGE:{lang_code}"}]} 
            ]
            formatted_history = format_chat_history(st.session_state.messages)
            
            # Check for empty payload before sending to API
            if not formatted_history or len(formatted_history) == 0:
                # Add at least one user message to prevent empty payload error
                contents.append({"role": "user", "parts": [{"text": user_input}]})
                st.sidebar.warning("⚠️ Chat history empty - using direct input")
            else:
                contents += formatted_history
                
            # Debug display
            if st.sidebar.checkbox("Show API payload for debugging", False):
                with st.sidebar.expander("API Request Payload"):
                    st.write("Content items:", len(contents))
                    st.json(contents)

            try:
                max_retries = 2
                retry_count = 0
                response = None
                
                while retry_count <= max_retries:
                    try:
                        # If it's not the first try, show retry message
                        if retry_count > 0:
                            placeholder.info(f"🔄 Retry attempt {retry_count}/{max_retries}...")
                        
                        response = model.generate_content(
                            contents,
                            generation_config=genai.types.GenerationConfig(
                                temperature=temperature,
                                max_output_tokens=512,
                                top_p=0.9,
                            ),
                        )
                        
                        # Success - break the retry loop
                        break
                    
                    except Exception as retry_error:
                        error_message = str(retry_error).lower()
                        
                        # More precise quota error detection - must be exact match
                        is_quota_error = False
                        
                        # Only consider it a quota error if specifically mentioned
                        if "429" in error_message and "quota exceeded" in error_message:
                            is_quota_error = True
                        
                        if retry_count < max_retries and is_quota_error:
                            retry_count += 1
                            placeholder.warning(f"⚠️ Rate limit reached. Waiting before retry...")
                            
                            # If we have multiple keys, try rotating to another one
                            if hasattr(st.session_state, 'api_keys') and len(st.session_state.api_keys) > 1:
                                prev_key_index = st.session_state.current_key_index
                                st.session_state.current_key_index = (st.session_state.current_key_index + 1) % len(st.session_state.api_keys)
                                effective_key = st.session_state.api_keys[st.session_state.current_key_index]
                                configure_gemini(effective_key)
                                model = get_gemini_model(model_name)
                                placeholder.info(f"🔑 Switched to API key #{st.session_state.current_key_index + 1}")
                            
                            # Wait before retry (with increasing backoff)
                            time.sleep(2 * retry_count)
                        else:
                            # Not a quota error or max retries reached, re-raise with more details
                            placeholder.error(f"⚠️ Error: {type(retry_error).__name__}")
                            raise
                
                # Check if we got a response after retries
                if response:
                    text = (response.text or "") if hasattr(response, "text") else ""
                    if not text:
                        text = "I'm sorry, I couldn't generate a response right now. Please try again."

                    placeholder.write(text)
                    st.session_state.messages.append({"role": "assistant", "content": text})
                else:
                    raise Exception("Failed to get response after retries")
                    
            except Exception as e:
                # Fallback to alternate models if model not found/unsupported
                fallback_models = [
                    "gemini-pro",  # Most reliable model
                    "gemini-1.0-pro",
                    "gemini-1.5-pro", 
                    "gemini-1.5-flash"
                ]
                tried = [model_name]
                for alt in fallback_models:
                    if alt in tried or alt == model_name:
                        continue
                    # Only try fallbacks if models were checked
                    if 'generation_models' in locals() and alt not in generation_models:
                        tried.append(alt)
                        continue
                    try:
                        alt_model = get_gemini_model(alt)
                        response = alt_model.generate_content(
                            contents,
                            generation_config=genai.types.GenerationConfig(
                                temperature=temperature,
                                max_output_tokens=512,
                                top_p=0.9,
                            ),
                        )
                        text = (response.text or "") if hasattr(response, "text") else ""
                        if text:
                            placeholder.write(text)
                            st.session_state.messages.append({"role": "assistant", "content": text})
                            return
                    except Exception:
                        tried.append(alt)

                # Initialize error message variable first to avoid UnboundLocalError
                err = f"Connection issue: {str(e)}. Tried models: {', '.join(tried)}."
                
                # Check the exact error type and message
                error_message = str(e).lower()
                error_type = type(e).__name__
                
                # More precise error categorization
                if "429" in error_message and "quota exceeded" in error_message:
                    # Extract retry delay if available
                    retry_seconds = 60  # Default
                    retry_match = re.search(r"retry in (\d+\.\d+)s", str(e))
                    if retry_match:
                        retry_seconds = int(float(retry_match.group(1))) + 1
                    
                    # Create a more helpful message that includes the wait time
                    wait_minutes = max(1, retry_seconds // 60)
                    
                    quota_message = f"""
                    ### ⌛ **API Quota Limit Reached**
                    
                    You've reached Google's rate limit for API requests. This is normal and happens when:
                    
                    1. **Too many requests** sent in a short time period
                    2. **Daily free quota limit** has been reached
                    
                    **The system will automatically retry in {wait_minutes} minute(s).**
                    
                    Other options:
                    1. **Wait {wait_minutes} minute(s)** and then send a new message
                    2. **Try a different API key** (add one in the sidebar)
                    3. **Upgrade to paid tier** for higher limits
                    
                    *This is not an error - it's just Google's way of managing free tier usage.*
                    """
                    placeholder.markdown(quota_message)
                    
                    # Use a more user-friendly error message
                    err = f"💬 I'll respond to your question shortly. Google's API needs a short break (quota limit reached). Please wait about {wait_minutes} minute(s) or try a different API key."
                    
                    # Add countdown for better user experience
                    countdown = st.empty()
                    countdown_seconds = min(retry_seconds, 60)  # Limit to 1 minute max display
                    for i in range(countdown_seconds, 0, -1):
                        remaining_min = i // 60
                        remaining_sec = i % 60
                        time_format = f"{remaining_min}m {remaining_sec}s" if remaining_min > 0 else f"{remaining_sec}s"
                        countdown.info(f"⏱️ API cooldown: {time_format} remaining...")
                        time.sleep(1)
                    countdown.success("✅ You can try again now!")
                    
                elif "404" in error_message and "not found" in error_message:
                    quota_message = """
                    ### ⚠️ **Model Not Found Error**
                    
                    The requested model couldn't be found or isn't available with your API key:
                    
                    **Solutions:**
                    
                    1. **Use a different model** from the dropdown above
                    2. **Create a new API key** with access to more models
                    3. Check if you need to **enable the Generative AI API** in Google Cloud Console
                    """
                    placeholder.markdown(quota_message)
                    err = f"Model not found. Please try a different model. Tried models: {', '.join(tried)}."
                elif "request payload is empty" in error_message or "400" in error_message:
                    quota_message = """
                    ### ⚠️ **Empty Request Error**
                    
                    The API received an empty payload. This can happen when:
                    
                    1. The conversation history is corrupted
                    2. There's a formatting issue with your message
                    
                    **Solutions:**
                    
                    1. **Click "Clear Conversation"** in the sidebar
                    2. **Try a shorter, simpler message**
                    3. **Restart the app** if the problem persists
                    """
                    placeholder.markdown(quota_message)
                    err = "I couldn't process that message. Please try clearing the conversation and starting over with a new message."
                    
                    # Add a clear conversation button for convenience
                    if st.button("🗑️ Clear Conversation History"):
                        st.session_state.messages = []
                        st.experimental_rerun()
                else:
                    # Generic API error
                    quota_message = f"""
                    ### ⚠️ **API Error: {error_type}**
                    
                    There was an issue with the Gemini API:
                    
                    **Error details:** 
                    ```
                    {str(e)[:300]}
                    ```
                    
                    **Solutions:**
                    1. **Try a different model** from the dropdown above
                    2. **Create a new API key** in Google AI Studio
                    3. **Wait a few minutes** and try again
                    """
                    placeholder.markdown(quota_message)
                
                # Display the error message
                placeholder.write(err)
            
            # Add the error message to session state
            st.session_state.messages.append({"role": "assistant", "content": err})

    # Tawk widget intentionally not injected on this page per user request

if __name__ == "__main__":
    main()


