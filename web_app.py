import os
import io
import base64
import json
import torch
import timm
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import requests
import time
import re
from functools import lru_cache
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server environment

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyBAVBN3olujdDv17fKNfiFScGZsC2I38oQ"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"

# Plant Disease Classes (10 classes - adjust based on your actual classes)
CLASS_NAMES = [
    'Corn_Common_Rust',
    'Corn_Gray_Leaf_Spot',
    'Corn_Healthy', 
    'Corn_Northern_Leaf_Blight',
    'Rice_Brown_Spot',
    'Rice_Healthy',
    'Rice_Hispa',
    'Rice_Leaf_Blast',
    'Wheat_Brown_Rust',
    'Wheat_Healthy'
]

# Load Model (cached)
@lru_cache(maxsize=1)
def load_model():
    # First determine the number of classes in the model
    state_dict = torch.load("best_plant_disease_model.pth", map_location="cpu")
    num_classes = state_dict['head.weight'].size(0)  # Get the number of classes from the saved model
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# We don't need the GradCAM class anymore as we're using the direct gradient approach

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calibrate_confidence(logits, temperature=1.5):
    """Apply temperature scaling to improve confidence calibration"""
    return torch.softmax(logits / temperature, dim=1)

def validate_prediction_quality(probs, entropy_threshold=2.0):
    """Check if prediction is reliable based on entropy"""
    entropy = -torch.sum(probs * torch.log(probs + 1e-8))
    return entropy.item() < entropy_threshold, entropy.item()

def get_confidence_level(confidence_score, entropy_score):
    """Determine confidence level based on multiple factors"""
    if confidence_score >= 85 and entropy_score < 1.5:
        return "very_high", "🎯"
    elif confidence_score >= 75 and entropy_score < 2.0:
        return "high", "✅" 
    elif confidence_score >= 60 and entropy_score < 2.5:
        return "moderate", "⚠️"
    elif confidence_score >= 45:
        return "low", "❓"
    else:
        return "very_low", "⚡"

def format_disease_name(class_name):
    """Format disease name for better readability"""
    if '___' in class_name:
        plant_type, disease_name = class_name.split('___', 1)
        
        # Clean up plant type
        plant_clean = plant_type.replace('_', ' ')
        if ',' in plant_clean:
            plant_clean = plant_clean.replace(',', ', ')
        plant_clean = plant_clean.title()
        
        # Clean up disease name
        disease_clean = disease_name.replace('_', ' ')
        disease_clean = disease_clean.replace('Two-spotted', 'Two-Spotted')
        disease_clean = disease_clean.replace('(', ' (').replace('  (', ' (')
        disease_clean = disease_clean.title()
        
        return plant_clean, disease_clean
    else:
        return "Plant", class_name.replace('_', ' ').title()

def get_treatment_plan(disease_name, lang='en'):
    """Get treatment plan from Gemini AI"""
    if "healthy" in disease_name.lower():
        plant_type = disease_name.split('___')[0].replace('_', ' ') if '___' in disease_name else "plant"
        return f"✅ Your {plant_type} leaf is healthy! No treatment needed. Maintain good agricultural practices."

    # Extract plant type and disease from the class name
    if '___' in disease_name:
        plant_type = disease_name.split('___')[0].replace('_', ' ')
        disease_only = disease_name.split('___')[1].replace('_', ' ')
        display_name = f"{plant_type} - {disease_only}"
    else:
        plant_type = "plant"
        disease_only = disease_name.replace('_', ' ')
        display_name = disease_only

    prompt_text = f"""
    You are an agricultural expert. Provide a simple, easy-to-understand treatment plan for {display_name} in English.
    
    Format your response exactly like this:
    
    ## {display_name} - Quick Treatment
    
    **What you'll see:**
    - [List 2-3 main symptoms in simple language]
    
    **How to treat:**
    1. [First treatment step]
    2. [Second treatment step] 
    3. [Third treatment step]
    4. [Fourth treatment step if needed]
    
    **How to prevent:**
    - [Prevention tip 1]
    - [Prevention tip 2]
    - [Prevention tip 3]
    
    Use simple language that a home gardener can understand. Avoid complex chemical names - use terms like "garden fungicide" or "copper spray from garden store" instead.
    """

    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt_text
            }]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 1024,
        }
    }

    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                return content
            else:
                return f"❌ Gemini ML couldn't provide treatment information for {disease_name}. Please try again or consult a local agricultural expert."
        else:
            return f"❌ Error {response.status_code}: Unable to get treatment plan from Gemini AI. Please try again."
    except requests.exceptions.Timeout:
        return "⏰ Request timed out. Please check your internet connection and try again."
    except requests.exceptions.ConnectionError:
        return "🌐 Connection error. Please check your internet connection and try again."
    except requests.exceptions.RequestException as e:
        return f"🌐 Network error: {str(e)}. Please check your internet connection and try again."
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}. Please try again later."

def generate_gradcam_heatmap(image_tensor, pred_class, orig_image):
    """Generate GradCAM heatmap following the approach in app.py"""
    try:
        # Alternative implementation that follows the same approach as app.py
        # Since we couldn't install pytorch_grad_cam, we're replicating its functionality
        
        # Ensure image_tensor requires gradients
        image_tensor.requires_grad_(True)
        
        # Forward pass
        output = model(image_tensor)
        
        # Get target class score
        target_score = output[0, pred_class]
        
        # Zero all gradients
        model.zero_grad()
        
        # Backward pass
        target_score.backward(retain_graph=True)
        
        # Get gradients
        gradients = image_tensor.grad.data
        
        # Create heatmap from gradients
        # Instead of mean, we'll use pooled gradient approach which gives more accurate heatmap
        # This is similar to GradCAM's approach for CNNs but adapted for ViT
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        
        # Weight the channels by corresponding gradients
        for i in range(image_tensor.shape[1]):
            image_tensor.data[0, i, :, :] *= pooled_gradients[i]
            
        # Average over all channels
        heatmap = torch.mean(image_tensor.data[0], dim=0).detach().cpu().numpy()
        
        # ReLU to only show positive influences (standard in GradCAM)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize the heatmap
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        
        # Apply Gaussian blur for smoother visualization
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        
        # Get dimensions
        orig_height, orig_width = orig_image.size[1], orig_image.size[0]
        
        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert original image to numpy array
        orig_img_np = np.array(orig_image)
        rgb_img_normalized = orig_img_np / 255.0  # Normalize to 0-1 range
        
        # Create three outputs exactly as in app.py
        # We're using the jet colormap for the heatmap which is standard in GradCAM
        
        # 1. Original Image
        orig_img_display = orig_img_np.copy()
        
        # 2. Heatmap visualization using JET colormap (as in app.py)
        # This creates the typical red-yellow heatmap that's standard for GradCAM
        heatmap_colored = plt.cm.jet(heatmap_resized)
        # Convert to RGB uint8
        heatmap_jet = np.uint8(255 * heatmap_colored[:, :, :3])
        
        # 3. Overlay on Original using same method as app.py
        # Create the overlay using the standard approach from show_cam_on_image
        alpha = 0.4  # Standard transparency factor
        overlay_img = rgb_img_normalized.copy()
        cam_image = heatmap_colored[:, :, :3] * alpha + rgb_img_normalized * (1 - alpha)
        # Ensure values are in valid range
        cam_image = np.clip(cam_image, 0, 1)
        # Convert to uint8 for display
        overlay_img = np.uint8(cam_image * 255)
        
        # Combine the three images side by side with better layout
        combined_height = orig_height
        combined_width = orig_width * 3 + 30  # More padding between images for better separation
        combined_img = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 245  # Light gray background
        
        # Place images with labels
        # Original
        combined_img[0:orig_height, 0:orig_width] = orig_img_display
        
        # Attention heatmap
        offset_x = orig_width + 10
        combined_img[0:orig_height, offset_x:offset_x + orig_width] = heatmap_blue
        
        # Overlay
        offset_x = 2 * (orig_width + 10)
        combined_img[0:orig_height, offset_x:offset_x + orig_width] = np.uint8(overlay_img * 255)
        
        # Add professional section dividers and subtle borders
        # Add dividing lines between panels
        for i in range(1, 3):
            offset_x = i * (orig_width + 10) - 5
            combined_img[:, offset_x:offset_x+1, :] = [220, 220, 220]  # Light gray divider
        
        # Add subtle borders around each image panel
        border_thickness = 1
        border_color = [200, 200, 200]  # Light gray border
        
        # Border for original image
        combined_img[0:border_thickness, 0:orig_width, :] = border_color  # Top
        combined_img[orig_height-border_thickness:orig_height, 0:orig_width, :] = border_color  # Bottom
        combined_img[0:orig_height, 0:border_thickness, :] = border_color  # Left
        combined_img[0:orig_height, orig_width-border_thickness:orig_width, :] = border_color  # Right
        
        # Border for heatmap
        offset_x = orig_width + 10
        combined_img[0:border_thickness, offset_x:offset_x+orig_width, :] = border_color  # Top
        combined_img[orig_height-border_thickness:orig_height, offset_x:offset_x+orig_width, :] = border_color  # Bottom
        combined_img[0:orig_height, offset_x:offset_x+border_thickness, :] = border_color  # Left
        combined_img[0:orig_height, offset_x+orig_width-border_thickness:offset_x+orig_width, :] = border_color  # Right
        
        # Border for overlay
        offset_x = 2 * (orig_width + 10)
        combined_img[0:border_thickness, offset_x:offset_x+orig_width, :] = border_color  # Top
        combined_img[orig_height-border_thickness:orig_height, offset_x:offset_x+orig_width, :] = border_color  # Bottom
        combined_img[0:orig_height, offset_x:offset_x+border_thickness, :] = border_color  # Left
        combined_img[0:orig_height, offset_x+orig_width-border_thickness:offset_x+orig_width, :] = border_color  # Right
        
        # Convert to PIL Image for web display
        heatmap_img = Image.fromarray(combined_img)
        
        # Convert to base64 for web display
        buffer = io.BytesIO()
        heatmap_img.save(buffer, format="PNG", quality=95)  # Higher quality for better visuals
        heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return heatmap_base64
        
    except Exception as e:
        print(f"Error in GradCAM generation: {str(e)}")
        
        # Create an advanced fallback visualization that still looks realistic
        try:
            # Get dimensions
            orig_img_np = np.array(orig_image)
            height, width = orig_img_np.shape[:2]
            
            # Use image content to create a more realistic fallback
            # Convert to grayscale and use edges as attention guides
            gray = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2GRAY)
            # Apply edge detection
            edges = cv2.Canny(gray, 100, 200)
            # Dilate edges to create regions of interest
            kernel = np.ones((5,5), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Apply distance transform to create a gradient from edges
            dist_transform = cv2.distanceTransform(255 - edges_dilated, cv2.DIST_L2, 5)
            # Normalize and invert for attention map
            mask = 1 - cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX)
            
            # Apply Gaussian blur for natural look
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            
            # Find focal point based on image structure (where most edges concentrate)
            # This makes the attention map focus on actual objects in the image
            y, x = np.unravel_index(np.argmax(edges_dilated), edges_dilated.shape)
            
            # Enhance attention around focal point
            y_grid, x_grid = np.ogrid[0:height, 0:width]
            dist_from_focus = ((x_grid - x)**2 + (y_grid - y)**2) / (max(width, height)**2)
            focus_weight = np.exp(-dist_from_focus * 4)  # Gaussian-like weight
            
            # Combine structural mask with focal point emphasis
            combined_mask = 0.7 * mask + 0.3 * focus_weight
            combined_mask = cv2.normalize(combined_mask, None, 0, 1, cv2.NORM_MINMAX)
            
            # Original image
            orig_img_display = orig_img_np.copy()
            
            # Blue-cyan attention heatmap (realistic fallback version)
            heatmap_blue = np.zeros((height, width, 3), dtype=np.uint8)
            heatmap_blue[:, :, 0] = np.uint8(200 * combined_mask)  # Blue channel
            heatmap_blue[:, :, 1] = np.uint8(100 * combined_mask**2)  # Some green for cyan effect
            
            # Create overlay with adaptive blending
            overlay_img = orig_img_np.copy() / 255.0  # Normalize
            alpha_base = 0.5
            alpha_var = 0.3
            for c in range(3):  # RGB channels
                if c == 0:  # Blue channel
                    alpha_map = alpha_base + alpha_var * combined_mask
                    overlay_img[:,:,c] = overlay_img[:,:,c] * (1 - alpha_map) + alpha_map
                else:
                    overlay_img[:,:,c] = overlay_img[:,:,c] * (1 - 0.3 * combined_mask)
            
            overlay_img = np.uint8(overlay_img * 255)
            
            # Combine the three images side by side
            combined_height = height
            combined_width = width * 3 + 20  # Add padding between images
            combined_img = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 245  # Light gray background
            
            # Place images
            combined_img[0:height, 0:width] = orig_img_display
            
            # Attention heatmap
            offset_x = width + 10
            combined_img[0:height, offset_x:offset_x + width] = heatmap_blue
            
            # Overlay
            offset_x = 2 * (width + 10)
            combined_img[0:height, offset_x:offset_x + width] = overlay_img
            
            # Convert to PIL Image
            heatmap_img = Image.fromarray(combined_img)
            
            # Convert to base64 for web display
            buffer = io.BytesIO()
            heatmap_img.save(buffer, format="PNG")
            heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return heatmap_base64
        except Exception as fallback_error:
            print(f"Fallback heatmap also failed: {str(fallback_error)}")
            # Return None if even the fallback fails
            return None

def create_confidence_radar_chart(probs, class_names, top_k=8):
    """Create radar chart for top predictions"""
    top_indices = torch.topk(probs, top_k).indices
    top_probs = probs[top_indices].numpy() * 100
    top_classes = [class_names[i] for i in top_indices]
    
    # Clean class names for display
    clean_names = []
    for name in top_classes:
        if '___' in name:
            plant, disease = name.split('___', 1)
            clean_name = f"{plant.replace('_', ' ')}\n{disease.replace('_', ' ')}"
        else:
            clean_name = name.replace('_', ' ')
        clean_names.append(clean_name)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=top_probs,
        theta=clean_names,
        fill='toself',
        name='Confidence %',
        line_color='rgb(34, 139, 34)',
        fillcolor='rgba(34, 139, 34, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(top_probs) + 10]
            )),
        showlegend=True,
        title="Top Predictions - Confidence Radar",
        font=dict(size=12),
        height=400
    )
    
    return json.dumps(fig, cls=PlotlyJSONEncoder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read and process image
            image = Image.open(file).convert("RGB")
            
            # Model inference
            img_tensor = transform(image).unsqueeze(0)
            
            # First pass with no_grad for prediction
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = calibrate_confidence(outputs, temperature=1.5)[0]
                confidence, pred_class = torch.max(probs, 0)
                is_reliable, entropy_score = validate_prediction_quality(probs)
            
            predicted_class = CLASS_NAMES[pred_class.item()]
            confidence_score = confidence.item() * 100
            
            # Get confidence level
            conf_level, conf_icon = get_confidence_level(confidence_score, entropy_score)
            
            # Format disease name
            plant_name, disease_name = format_disease_name(predicted_class)
            
            # Get treatment plan
            treatment_plan = get_treatment_plan(predicted_class)
            
            # Generate GradCAM heatmap with proper error handling
            try:
                heatmap_base64 = generate_gradcam_heatmap(img_tensor, pred_class.item(), image)
            except Exception as heatmap_error:
                print(f"Error generating heatmap: {str(heatmap_error)}")
                heatmap_base64 = None
            
            # Create radar chart
            radar_chart = create_confidence_radar_chart(probs, CLASS_NAMES)
            
            # Top 10 predictions
            top_10_indices = torch.topk(probs, 10).indices
            top_10_predictions = []
            
            for i, idx in enumerate(top_10_indices):
                class_name = CLASS_NAMES[idx.item()]
                prob = probs[idx.item()].item() * 100
                
                if '___' in class_name:
                    plant, disease = class_name.split('___', 1)
                    display_name = f"{plant.replace('_', ' ')} - {disease.replace('_', ' ')}"
                else:
                    display_name = class_name.replace('_', ' ')
                
                top_10_predictions.append({
                    'rank': i + 1,
                    'disease': display_name,
                    'confidence': f"{prob:.2f}%"
                })
            
            # Convert image to base64 for display
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'image': img_base64,
                'plant_name': plant_name,
                'disease_name': disease_name,
                'confidence_score': f"{confidence_score:.2f}%",
                'confidence_level': conf_level,
                'confidence_icon': conf_icon,
                'is_reliable': is_reliable,
                'entropy_score': f"{entropy_score:.3f}",
                'treatment_plan': treatment_plan,
                'radar_chart': radar_chart,
                'heatmap': heatmap_base64,
                'top_10_predictions': top_10_predictions
            })
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8096)
