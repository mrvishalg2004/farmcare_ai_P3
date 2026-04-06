import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
import requests
import time
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
warnings.filterwarnings('ignore')
import streamlit.components.v1 as components
import google.generativeai as genai
import json
from langdetect import detect, LangDetectException
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# 🌐 Multi-Language Support
# -----------------------------
LANGUAGES = {
    'English': 'en',
    'Español': 'es',
    'Français': 'fr',
    'Deutsch': 'de',
    'Italiano': 'it',
    'Português': 'pt',
    '中文': 'zh',
    '日本語': 'ja',
    '한국어': 'ko',
    'हिंदी': 'hi',
    'मराठी': 'mr',
    'العربية': 'ar',
    'Русский': 'ru'
}

# -----------------------------
# 🤖 Gemini AI Chatbot Configuration
# -----------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Language translations dictionary
TRANSLATIONS = {
    'en': {
        'title': '🌱 Plant Disease Detection Tool',
        'subtitle': 'Upload a plant leaf image to detect disease and get treatment suggestions',
        'upload_text': '📂 Drag & drop or select a leaf image',
        'uploaded_image': 'Uploaded Leaf Image',
        'analysis_progress': '🔍 ML Analysis in Progress...',
        'preprocessing': 'Preprocessing image...',
        'extracting': 'Extracting features...',
        'detecting': 'Running disease detection...',
        'calculating': 'Calculating confidence scores...',
        'visualizing': 'Generating visualizations...',
        'prediction_results': '🎯 Prediction Results',
        'predicted_disease': '🏆 Predicted Disease',
        'confidence_score': '📊 Confidence Score',
        'high_confidence': '🎯 High Confidence',
        'moderate_confidence': '⚠️ Moderate Confidence',
        'low_confidence': '❓ Low Confidence - Consider retaking the image',
        'treatment_plan': '💡 Suggested Treatment Plan',
        'generating_treatment': 'Generating personalized treatment recommendations...',
        'advanced_analysis': '🔬 Advanced Model Analysis & Visualizations',
        'gradcam_tab': '🎯 GradCAM Heatmap',
        'probabilities_tab': '📊 Detailed Probabilities',
        'radar_tab': '🕸️ Confidence Radar',
        'plant_analysis_tab': '🌿 Plant Analysis',
        'model_insights_tab': '⚡ Model Insights',
        'gradcam_title': '🔥 GradCAM Visualization - What the ML Sees',
        'gradcam_description': 'This heatmap shows which parts of the leaf the ML model focused on for its prediction.',
        'original_image': '🖼️ Original Image',
        'attention_heatmap': '🔥 Attention Heatmap',
        'ai_focus_overlay': '🎯 ML Focus Overlay',
        'language_select': '🌐 Select Language',
        'invalid_image': '⚠️ Invalid image! Please upload a valid JPG/PNG file.',
        'healthy_plant': 'Your {plant_type} leaf is healthy! No treatment needed. Maintain good agricultural practices.',
        'heatmap_success': '✅ **Heatmap Generated Successfully!**',
        'how_to_read': '🔍 **How to Read This**:',
        'red_areas': '🔴 **Red/Hot areas**: Most important regions for ML decision',
        'yellow_areas': '🟡 **Yellow/Warm areas**: Moderately important regions',
        'blue_areas': '🔵 **Blue/Cool areas**: Less relevant regions',
        'focus_explanation': '🎯 **Focus**: The ML concentrated on the highlighted areas to make its prediction',
        'peak_attention': '🎯 Peak Attention Score',
        'attention_center': '📍 **Attention Center**: Row {row}, Column {col}',
        'comprehensive_analysis': '📈 Comprehensive Probability Analysis',
        'top_predictions': '🏆 Top 10 Predictions',
        'rank': 'Rank',
        'disease': 'Disease',
        'confidence_percent': 'Confidence (%)',
    }
}

# Your existing model loading and prediction functions would go here...

# -----------------------------
# 🤖 Gemini AI Chatbot Functions
# -----------------------------
def init_gemini_model():
    """Initialize and return the Gemini AI model"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model

# Initialize chat history in session state if not already there
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize chat visibility state
if 'chat_visible' not in st.session_state:
    st.session_state.chat_visible = False

def toggle_chat():
    """Toggle chat window visibility"""
    st.session_state.chat_visible = not st.session_state.chat_visible
    st.rerun()  # Use st.rerun() which is the modern replacement for experimental_rerun

# Check URL parameters to show chat if requested via URL
params = st.query_params
if "show_chat" in params and params["show_chat"] == "true":
    st.session_state.chat_visible = True
    # Remove parameter after processing
    new_params = dict(params)
    if "show_chat" in new_params:
        del new_params["show_chat"]
    st.query_params.update(**new_params)
    
def detect_language_safely(text):
    """Detect language of input text with fallback to English"""
    try:
        lang_code = detect(text)
        return lang_code
    except LangDetectException:
        return 'en'

def generate_gemini_response(user_input):
    """Generate a response from Gemini AI model"""
    model = init_gemini_model()
    
    # Detect user input language for personalized response
    detected_lang = detect_language_safely(user_input)
    
    # Craft a prompt for Gemini that includes context about plant diseases
    system_prompt = """
    You are Farmcare AI, a helpful virtual assistant specializing in plant diseases, agriculture, and plant care.
    - Be friendly, concise, and informative
    - Use emojis occasionally to make responses engaging 🌱
    - If asked about plant diseases, provide helpful information on symptoms, causes, and treatments
    - For plant care questions, give practical advice for home gardeners and farmers
    - If you're uncertain about specific plant diseases, acknowledge limitations and suggest consulting local agricultural experts
    - Keep responses focused on agriculture, plants, gardening, and related topics
    - Format responses in easy-to-read paragraphs with bullet points for steps/lists
    - Respond in the same language as the user's query
    """
    
    try:
        response = model.generate_content(
            [system_prompt, user_input],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=800,
                top_p=0.95,
            )
        )
        return response.text
    except Exception as e:
        return f"I'm having trouble connecting to my knowledge base. Please try again. Error: {str(e)}"

# --------------------------
# 📱 Streamlit App
# --------------------------
def main():
    st.title("🌱 Plant Disease Detection Tool")
    st.write("Upload a plant leaf image to detect disease and get treatment suggestions")
    
    # Here would go your plant disease detection functionality
    # For brevity, this part is omitted since we're focusing on the chat functionality
    
    # Add CSS for chat button and chat window
    st.markdown("""
    <style>
    /* Chat button styling */
    .chat-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background-color: #4CAF50;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        z-index: 9999;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    /* Chat container styling */
    .chat-container {
        position: fixed;
        bottom: 90px;
        right: 20px;
        width: 350px;
        height: 500px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        z-index: 1000;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    /* Chat header styling */
    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #4CAF50;
        color: white;
        padding: 10px 15px;
        font-weight: bold;
    }
    
    /* Message bubbles */
    .message-bubble {
        padding: 8px 12px;
        border-radius: 15px;
        max-width: 80%;
        word-wrap: break-word;
        margin-bottom: 10px;
    }
    
    .user-bubble {
        background-color: #4CAF50;
        color: white;
        border-radius: 15px 15px 0 15px;
        align-self: flex-end;
    }
    
    .bot-bubble {
        background-color: white;
        color: black;
        border-radius: 15px 15px 15px 0;
        border: 1px solid #e0e0e0;
        align-self: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display chat button when chat is not visible
    if not st.session_state.chat_visible:
        chat_button_html = """
        <div class="chat-button" onclick="toggleChat()" style="box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
            <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="white">
                <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
                <path d="M7 9h2v2H7zm4 0h2v2h-2zm4 0h2v2h-2z"/>
            </svg>
        </div>
        
        <script>
            function toggleChat() {
                window.location.href = window.location.pathname + "?show_chat=true";
            }
        </script>
        """
        components.html(chat_button_html, height=70)
    
    # Display chat interface when active
    if st.session_state.chat_visible:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Chat header with title and close button
        st.markdown("""
            <div class="chat-header">
                <div>💬 Farmcare AI Assistant</div>
                <div style="cursor: pointer;" onclick="closeChat()">❌</div>
            </div>
            <script>
                function closeChat() {
                    window.location.href = window.location.pathname;
                }
            </script>
        """, unsafe_allow_html=True)
        
        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f"""<div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
                            <div class="message-bubble user-bubble">
                                {msg["content"]}
                            </div>
                        </div>""", 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""<div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
                            <div class="message-bubble bot-bubble">
                                {msg["content"]}
                            </div>
                        </div>""",
                        unsafe_allow_html=True
                    )
            
            # Chat input form
            with st.form(key="chat_form", clear_on_submit=True):
                cols = st.columns([0.85, 0.15])
                with cols[0]:
                    user_input = st.text_input("Type a message:", key="user_input")
                with cols[1]:
                    submit_button = st.form_submit_button("📤")
                
                if submit_button and user_input:
                    # Add user message to chat history
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    
                    # Generate AI response
                    ai_response = generate_gemini_response(user_input)
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    
                    # Rerun to update UI
                    st.rerun()

# Run the app
if __name__ == "__main__":
    main()