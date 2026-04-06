import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import plotly.express as px
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import google.generativeai as genai
from langdetect import detect, LangDetectException
import os
import io
import time
import random
import base64
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

# Configure the Gemini AI
genai.configure(api_key=GEMINI_API_KEY)

# Initialize model
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Translation function
def translate_text(text, target_language):
    try:
        if target_language == 'en' or not text.strip():  # Skip if English or empty
            return text
            
        # Use Gemini for translation
        model = genai.GenerativeModel('gemini-flash-latest')
        prompt = f"Translate the following text from {detect(text)} to {target_language}:\n\n{text}"
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# Load the model
@st.cache_resource
def load_model():
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=10)
    checkpoint = torch.load("best_plant_disease_model.pth", map_location=device)
    
    # Handle potential state_dict format differences
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    return model

# Get class names (adjust as needed based on your model)
def get_class_names():
    return [
        "Apple - Black Rot",
        "Apple - Healthy",
        "Cherry - Powdery Mildew",
        "Corn - Common Rust",
        "Grape - Black Rot",
        "Peach - Bacterial Spot",
        "Pepper - Bacterial Spot",
        "Potato - Early Blight",
        "Strawberry - Healthy",
        "Tomato - Mosaic Virus"
    ]

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Generate prediction
def predict(img, model):
    img_tensor = preprocess_image(img).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    class_names = get_class_names()
    probs_list = probabilities.cpu().numpy()
    
    # Return top 3 predictions
    top_3_idx = np.argsort(probs_list)[-3:][::-1]
    top_3_classes = [class_names[i] for i in top_3_idx]
    top_3_probs = [probs_list[i] for i in top_3_idx]
    
    return list(zip(top_3_classes, top_3_probs))

# Generate Grad-CAM visualization
def generate_gradcam(img, model, target_class_idx):
    # Prepare the image
    img_array = np.array(img.resize((224, 224))) / 255.0
    
    # Define the target layer - for ViT models we typically use the last attention block
    target_layer = model.blocks[-1].norm1
    
    # Create GradCAM object
    cam = GradCAM(model=model, target_layer=target_layer)
    
    # Generate CAM
    targets = [ClassifierOutputTarget(target_class_idx)]
    grayscale_cam = cam(input_tensor=preprocess_image(img).to(device), targets=targets)
    grayscale_cam = grayscale_cam[0]
    
    # Overlay CAM on image
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    
    return visualization

# Get treatment recommendations using Gemini AI
def get_treatment_recommendations(disease_name, language_code='en'):
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        
        prompt = f"""
        You are a plant disease expert. Provide detailed information about {disease_name} including:
        1. Disease overview
        2. Symptoms and identification
        3. Causes and conditions
        4. Treatment options
        5. Prevention strategies
        
        Format the information in a clear, concise, and organized manner suitable for farmers.
        """
        
        response = model.generate_content(prompt)
        treatment_text = response.text
        
        # Translate if needed
        if language_code != 'en':
            treatment_text = translate_text(treatment_text, language_code)
            
        return treatment_text
        
    except Exception as e:
        return f"Error generating treatment recommendations: {e}"

# Set up the Streamlit app
# Function to add a particle background
def add_bg_animation():
    # CSS for particle animation background
    particle_css = '''
    <style>
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(5deg); }
        100% { transform: translateY(0px) rotate(0deg); }
    }
    
    .floating-leaf {
        position: fixed;
        opacity: 0.15;
        z-index: -1;
        pointer-events: none;
        animation: float 15s ease-in-out infinite;
    }
    
    .gradient-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 50%, rgba(255,255,255,0) 40%, rgba(255,255,255,0.9) 100%);
        z-index: -1;
        pointer-events: none;
    }
    </style>
    
    <div class="gradient-overlay"></div>
    '''
    
    # Generate 10 random floating leaves with random positions
    for i in range(10):
        left = random.randint(0, 100)
        top = random.randint(0, 100)
        size = random.randint(40, 100)
        delay = random.randint(0, 15)
        
        # Choose a leaf icon randomly (could be replaced with actual leaf SVGs)
        leaf_type = random.choice(['🍃', '🌿', '🍂', '🌱'])
        
        particle_css += f'''
        <div class="floating-leaf" style="left: {left}vw; top: {top}vh; font-size: {size}px; animation-delay: {delay}s; opacity: 0.1;">
            {leaf_type}
        </div>
        '''
    
    st.markdown(particle_css, unsafe_allow_html=True)

def main():
    # Set page configuration
    st.set_page_config(
        page_title="FarmCare AI - Plant Disease Detection",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add animated background
    add_bg_animation()
    
    # Load model
    model = load_model()
    
    # Custom CSS for modern UI styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    body {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, rgba(255,255,255,0.8), rgba(248,250,252,0.8));
    }
    
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Card styling for sections */
    .stMarkdown, .stImage, .stFileUploader, .stButton {
        animation: fadeIn 0.6s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .stHeader {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(79, 70, 229, 0.1);
        padding: 1rem 0;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    /* Heading styles */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h2 {
        font-size: 1.8rem;
        color: #1E293B;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.3rem;
        color: #334155;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        border: none;
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        color: white;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.3);
    }
    
    /* File uploader styling */
    .stFileUploader {
        padding: 1.5rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(79, 70, 229, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        box-shadow: 0 6px 24px rgba(79, 70, 229, 0.15);
    }
    
    /* Image styling */
    img {
        border-radius: 16px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    img:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
    }
    
    /* Progress bar animation */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        background-size: 200% 200%;
        animation: gradientShift 2s linear infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        animation: fadeIn 0.5s ease-out;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .chat-message:hover {
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    .chat-message-content {
        margin-left: 1rem;
        line-height: 1.6;
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.1), rgba(124, 58, 237, 0.1));
        border-left: 4px solid #4F46E5;
    }
    
    .chat-message.bot {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-left: 4px solid #10B981;
    }
    
    .user-avatar, .bot-avatar {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        object-fit: cover;
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
    }
    
    .bot-avatar {
        background: linear-gradient(135deg, #059669, #10B981);
    }
    </style>
    """, unsafe_allow_html=True)

    # Enhanced sidebar styling
    st.markdown("""
    <style>
    .css-1d391kg, .css-1lcbmhc {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-right: 1px solid rgba(79, 70, 229, 0.1);
    }
    
    .css-1v3fvcr {
        background: transparent;
    }
    
    /* Improve sidebar styling */
    [data-testid="stSidebar"] > div:first-child {
        background: rgba(249, 250, 251, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-right: 1px solid rgba(79, 70, 229, 0.1);
        padding: 2rem 1rem;
    }
    
    .sidebar .sidebar-content {
        background-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar for language selection with enhanced UI
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <img src="https://i.ibb.co/0GG6Gx5/plant-icon.png" width="80" style="margin-bottom: 1rem; border-radius: 50%; padding: 10px; background: linear-gradient(135deg, rgba(79, 70, 229, 0.1), rgba(124, 58, 237, 0.1)); box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);">
            <h2 style="font-family: 'Inter', sans-serif; font-weight: 700; margin: 0; background: linear-gradient(135deg, #4F46E5, #7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">FarmCare AI</h2>
            <p style="color: #64748B; font-size: 0.9rem; margin-top: 5px;">Plant Disease Detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🌐 Language")
        selected_language_name = st.selectbox("Select your preferred language", list(LANGUAGES.keys()), index=0, label_visibility="collapsed")
        selected_language_code = LANGUAGES[selected_language_name]
        
        st.markdown("---")
        st.markdown("### 📱 App Information")
        
        # Styled info box
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.6); border-radius: 16px; padding: 1.2rem; border: 1px solid rgba(79, 70, 229, 0.2); box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05); margin-top: 1rem;">
            <p style="margin-top: 0; color: #334155; font-size: 0.95rem;">
                This app uses a Vision Transformer (ViT) model to detect plant diseases from images.
            </p>
            <p style="color: #334155; font-size: 0.9rem;">
                Upload an image of a plant leaf to get:
            </p>
            <ul style="color: #334155; font-size: 0.9rem; padding-left: 1.2rem; margin-top: 0.5rem;">
                <li>Disease identification</li>
                <li>Confidence levels</li>
                <li>Treatment recommendations</li>
                <li>Visual explanation of the detection</li>
            </ul>
            <p style="color: #334155; font-size: 0.9rem; margin-bottom: 0;">
                You can also chat with an AI assistant for more information.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Main content
    # Enhanced header with modern design
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem; background: rgba(255, 255, 255, 0.7); padding: 1.5rem; border-radius: 16px; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); box-shadow: 0 4px 24px rgba(0, 0, 0, 0.08); border: 1px solid rgba(255, 255, 255, 0.3);">
        <img src="https://i.ibb.co/0GG6Gx5/plant-icon.png" width="60" style="margin-right: 1rem; animation: pulse 2s infinite;">
        <div>
            <h1 style="margin: 0; font-size: 2.6rem; background: linear-gradient(135deg, #4F46E5, #7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">FarmCare AI</h1>
            <p style="margin: 0; color: #64748B; font-size: 1.2rem;">Advanced Plant Disease Detection & Treatment Assistant</p>
        </div>
    </div>
    
    <style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("Upload an image of a plant leaf to detect diseases and get treatment recommendations.")
    
    # Image upload section
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image and processing information
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
        # Process the image
        with st.spinner('Analyzing image...'):
            # Make prediction
            predictions = predict(image, model)
            top_disease, confidence = predictions[0]
            class_idx = get_class_names().index(top_disease)
            
            # Generate Grad-CAM visualization
            gradcam_img = generate_gradcam(image, model, class_idx)
            
        # Display results
        with col2:
            st.subheader("Diagnosis Results:")
            
            # Format confidence as percentage
            confidence_pct = f"{confidence * 100:.2f}%"
            
            # Display top prediction with large text and color
            st.markdown(f"<h3 style='color:#2c3e50;'>Disease: {top_disease}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color:#2980b9;'>Confidence: {confidence_pct}</h4>", unsafe_allow_html=True)
            
            # Display top 3 predictions as a table
            results_df = pd.DataFrame(
                [(disease, f"{prob * 100:.2f}%") for disease, prob in predictions],
                columns=["Disease", "Confidence"]
            )
            st.dataframe(results_df, use_container_width=True)
            
            # Show Grad-CAM visualization
            st.subheader("Visual Explanation (Grad-CAM)")
            st.image(gradcam_img, caption="Highlighted areas show features that influenced the diagnosis", use_column_width=True)
        
        # Treatment recommendations
        st.markdown("---")
        st.subheader("📋 Treatment Recommendations")
        
        with st.spinner("Generating treatment recommendations..."):
            treatment_text = get_treatment_recommendations(top_disease, selected_language_code)
            st.markdown(treatment_text)
        
        # Add CSS for enhancing Streamlit's native chat component
        st.markdown("""
        <style>
        /* Modern Chat UI Styling */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        
        /* Message animation keyframes */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulseHighlight {
            0% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0.2); }
            70% { box-shadow: 0 0 0 10px rgba(79, 70, 229, 0); }
            100% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0); }
        }
        
        @keyframes typing {
            0% { opacity: 0.3; }
            50% { opacity: 1; }
            100% { opacity: 0.3; }
        }
        
        /* Style the message container */
        .stChatMessageContent {
            border-radius: 18px !important;
            padding: 14px 18px !important;
            font-family: 'Inter', sans-serif !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08) !important;
            line-height: 1.6 !important;
            animation: fadeInUp 0.4s ease forwards !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            transform-origin: bottom !important;
        }
        
        /* User message styling */
        .stChatMessage.user .stChatMessageContent {
            background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
            color: white !important;
            margin-left: 60px !important;
            margin-right: 12px !important;
        }
        
        /* Assistant message styling */
        .stChatMessage.assistant .stChatMessageContent {
            background: rgba(255, 255, 255, 0.9) !important; 
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            border: 1px solid rgba(79, 70, 229, 0.2) !important;
            color: #1E293B !important;
            margin-right: 60px !important;
            margin-left: 12px !important;
        }
        
        /* Style the chat avatars */
        .stChatMessage .stChatMessageAvatar {
            background: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(10px) !important;
            -webkit-backdrop-filter: blur(10px) !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.4) !important;
            overflow: hidden !important;
            transition: all 0.3s ease !important;
        }
        
        .stChatMessage:hover .stChatMessageAvatar {
            transform: scale(1.05) !important;
        }
        
        /* Style chat input area */
        .stChatInputContainer {
            padding-top: 16px !important;
            border-top: 1px solid rgba(79, 70, 229, 0.15) !important;
        }
        
        .stChatInput {
            border-radius: 20px !important;
            border: 1px solid rgba(79, 70, 229, 0.3) !important;
            background: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(8px) !important;
            -webkit-backdrop-filter: blur(8px) !important;
            padding: 14px 16px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
            transition: all 0.3s ease !important;
        }
        
        .stChatInput:focus {
            border-color: rgba(79, 70, 229, 0.6) !important;
            box-shadow: 0 4px 16px rgba(79, 70, 229, 0.15) !important;
            transform: translateY(-2px) !important;
        }
        
        /* Add animation to new messages */
        .stChatMessage:last-child {
            animation: fadeInUp 0.4s ease-out !important;
        }
        
        /* Chat input button styling */
        .stChatInputContainer button {
            border-radius: 12px !important;
            background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
            transition: all 0.3s ease !important;
        }
        
        .stChatInputContainer button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4) !important;
        }
        
        /* Chat title styling */
        .stChatFloatingInputContainer h3 {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            color: #1E293B !important;
            margin-bottom: 20px !important;
            border-bottom: 1px solid rgba(79, 70, 229, 0.15) !important;
            padding-bottom: 12px !important;
        }
        
        /* Overall chat container styling */
        .stChatContainer {
            border-radius: 20px !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            background: rgba(249, 250, 251, 0.7) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            box-shadow: 0 8px 30px rgba(0,0,0,0.12) !important;
            overflow: hidden !important;
        }
        
        /* Markdown content styling */
        .stChatMessageContent p {
            margin-bottom: 8px !important;
        }
        
        .stChatMessageContent code {
            background: rgba(79, 70, 229, 0.1) !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-size: 0.9em !important;
            font-family: 'Source Code Pro', monospace !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Initialize chat
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": f"👋 Hello! I'm your FarmCare AI assistant. I can help you with more information about {top_disease} or any other plant disease questions you may have."}
            ]

        # Chat interface with custom styling
        st.markdown("---")
        st.markdown("### 💬 Ask FarmCare AI Assistant")
        
        # Display chat messages with enhanced styling
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # User input with enhanced placeholder
        if prompt := st.chat_input("Ask me about plant diseases, treatments, or prevention..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message in chat
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response using Gemini with typing indicator animation
            with st.chat_message("assistant"):
                with st.spinner("💭 Thinking..."):
                    try:
                        # Show typing animation
                        typing_placeholder = st.empty()
                        typing_placeholder.markdown("*Generating response...*")
                        
                        # Generate response
                        model = genai.GenerativeModel('gemini-flash-latest')
                        context = f"Current detected plant disease: {top_disease}" if 'top_disease' in locals() else ""
                        
                        full_prompt = f"""
                        You are a knowledgeable plant disease expert helping a user. Be informative, helpful, and concise.
                        Provide practical advice and explain concepts in a clear way.
                        Context: {context}
                        
                        User question: {prompt}
                        """
                        
                        response = model.generate_content(full_prompt)
                        response_text = response.text
                        
                        # Translate response if needed
                        if selected_language_code != 'en':
                            response_text = translate_text(response_text, selected_language_code)
                        
                        # Remove typing indicator and show response
                        typing_placeholder.empty()
                        st.write(response_text)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                        
                    except Exception as e:
                        error_msg = f"⚠️ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    else:
        # Display example images or instructions
        st.info("Please upload an image to begin analysis")
        
        # Sample images display
        st.markdown("### Sample Images")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("Apple - Black Rot")
            st.image("https://plantvillage.psu.edu/images/pests/apple_black_rot.jpg", use_column_width=True)
            
        with col2:
            st.markdown("Tomato - Mosaic Virus")
            st.image("https://plantvillage.psu.edu/images/pests/tomato_mosaic_virus.jpg", use_column_width=True)
            
        with col3:
            st.markdown("Corn - Common Rust")
            st.image("https://plantvillage.psu.edu/images/pests/corn_common_rust.jpg", use_column_width=True)

if __name__ == "__main__":
    main()