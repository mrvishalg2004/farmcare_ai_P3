import os
import requests
import streamlit as st

def download_model_if_needed(model_path="best_plant_disease_model.pth"):
    """Download model from Hugging Face Hub if not available locally"""
    if os.path.exists(model_path):
        return True
    
    try:
        # Replace this URL with your actual model URL after uploading to Hugging Face or other storage
        MODEL_URL = "https://huggingface.co/yourusername/farmcare-ai/resolve/main/best_plant_disease_model.pth"
        
        with st.spinner("Downloading model... This may take a minute..."):
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        return True
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return False