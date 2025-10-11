#!/usr/bin/env python3
"""
Plant Disease Model Evaluation Script
=====================================
This script provides comprehensive evaluation of the plant disease detection model including:
- Model accuracy and performance metrics
- Confusion matrix visualization
- Classification report
- GradCAM visualization for model interpretability
- Per-class performance analysis
"""

import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import pandas as pd
from PIL import Image
import os
# GradCAM imports - simplified approach
GRADCAM_AVAILABLE = False
try:
    import pytorch_grad_cam
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
    print("✅ GradCAM library loaded successfully")
except ImportError:
    print("⚠️  Advanced GradCAM not available. Using basic gradient visualization.")
import cv2
import warnings
warnings.filterwarnings('ignore')

# Plant Disease Classes (38 classes)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model_path="plant_disease_model_final.pth"):
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Image preprocessing (using ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}...")
        
        # Create model architecture
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASS_NAMES))
        
        # Load state dict
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"📊 Model Info:")
        print(f"   - Architecture: Vision Transformer (ViT) Base")
        print(f"   - Total Parameters: {total_params:,}")
        print(f"   - Trainable Parameters: {trainable_params:,}")
        print(f"   - Number of Classes: {len(CLASS_NAMES)}")
        
    def create_sample_dataset(self, sample_images_dir=None):
        """
        Create or load a sample dataset for evaluation
        If no directory provided, creates synthetic data for demonstration
        """
        if sample_images_dir and os.path.exists(sample_images_dir):
            print(f"Loading images from {sample_images_dir}...")
            dataset = datasets.ImageFolder(sample_images_dir, transform=self.transform)
            return DataLoader(dataset, batch_size=32, shuffle=False)
        else:
            print("Creating synthetic sample data for demonstration...")
            # Create synthetic data for demonstration
            sample_data = []
            sample_labels = []
            
            # Generate synthetic images for each class (demonstration purposes)
            for class_idx in range(min(len(CLASS_NAMES), 10)):  # Limit to 10 classes for demo
                for _ in range(5):  # 5 samples per class
                    # Create random image data
                    synthetic_img = torch.randn(3, 224, 224)
                    sample_data.append(synthetic_img)
                    sample_labels.append(class_idx)
            
            return list(zip(sample_data, sample_labels))
    
    def evaluate_single_image(self, image_path):
        """Evaluate a single image and return predictions"""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
            
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            confidence, pred_class = torch.max(probs, 0)
        
        return {
            'image': image,
            'predicted_class': CLASS_NAMES[pred_class.item()],
            'confidence': confidence.item() * 100,
            'probabilities': probs.cpu().numpy(),
            'raw_outputs': outputs.cpu()
        }
    
    def generate_confusion_matrix(self, test_data):
        """Generate and visualize confusion matrix"""
        print("🔄 Generating confusion matrix...")
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        self.model.eval()
        with torch.no_grad():
            if isinstance(test_data, DataLoader):
                for batch_idx, (images, labels) in enumerate(test_data):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            else:
                # Handle list of tuples (synthetic data)
                for img_tensor, label in test_data:
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)
                    outputs = self.model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.append(label)
                    all_probs.extend(probs.cpu().numpy())
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Create visualization
        plt.figure(figsize=(15, 12))
        
        # Use subset of classes for better visualization if too many classes
        if len(np.unique(all_labels)) > 15:
            unique_labels = np.unique(all_labels)[:15]  # Show first 15 classes
            class_names_subset = [CLASS_NAMES[i] for i in unique_labels]
        else:
            unique_labels = np.unique(all_labels)
            class_names_subset = [CLASS_NAMES[i] for i in unique_labels]
        
        # Filter confusion matrix for subset
        cm_subset = cm[np.ix_(unique_labels, unique_labels)]
        
        # Create heatmap
        sns.heatmap(cm_subset, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=[name.replace('___', '\n').replace('_', ' ') for name in class_names_subset],
                    yticklabels=[name.replace('___', '\n').replace('_', ' ') for name in class_names_subset])
        
        plt.title('Confusion Matrix - Plant Disease Classification', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Classes', fontsize=12)
        plt.ylabel('True Classes', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm, all_predictions, all_labels, np.array(all_probs)
    
    def calculate_metrics(self, y_true, y_pred, y_probs):
        """Calculate comprehensive evaluation metrics"""
        print("📊 Calculating evaluation metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # Create comprehensive metrics report
        metrics_dict = {
            'Overall Accuracy': accuracy,
            'Weighted Precision': precision,
            'Weighted Recall': recall,
            'Weighted F1-Score': f1,
            'Total Samples': len(y_true)
        }
        
        print("\n🎯 OVERALL MODEL PERFORMANCE")
        print("=" * 50)
        for metric, value in metrics_dict.items():
            if 'Total' in metric:
                print(f"{metric}: {value}")
            else:
                print(f"{metric}: {value:.4f}")
        
        # Per-class performance
        unique_classes = np.unique(y_true)
        per_class_metrics = []
        
        for i, class_idx in enumerate(unique_classes):
            class_name = CLASS_NAMES[class_idx]
            per_class_metrics.append({
                'Class': class_name.replace('___', ' - ').replace('_', ' '),
                'Precision': precision_per_class[i] if i < len(precision_per_class) else 0,
                'Recall': recall_per_class[i] if i < len(recall_per_class) else 0,
                'F1-Score': f1_per_class[i] if i < len(f1_per_class) else 0,
                'Support': support_per_class[i] if i < len(support_per_class) else 0
            })
        
        # Create DataFrame for better visualization
        df_metrics = pd.DataFrame(per_class_metrics)
        
        # Save metrics to CSV
        df_metrics.to_csv('model_evaluation_metrics.csv', index=False)
        
        print("\n📋 PER-CLASS PERFORMANCE (Top 10)")
        print("=" * 80)
        print(df_metrics.head(10).to_string(index=False, float_format='%.4f'))
        
        return metrics_dict, df_metrics
    
    def visualize_performance_charts(self, df_metrics):
        """Create comprehensive performance visualization charts"""
        print("📈 Creating performance visualization charts...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle('Model Performance Analysis', fontsize=20, fontweight='bold')
        
        # 1. Top 10 Classes by F1-Score
        top_f1 = df_metrics.nlargest(10, 'F1-Score')
        axes[0,0].barh(range(len(top_f1)), top_f1['F1-Score'], color='skyblue')
        axes[0,0].set_yticks(range(len(top_f1)))
        axes[0,0].set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in top_f1['Class']])
        axes[0,0].set_xlabel('F1-Score')
        axes[0,0].set_title('Top 10 Classes by F1-Score')
        axes[0,0].grid(axis='x', alpha=0.3)
        
        # 2. Precision vs Recall Scatter Plot
        axes[0,1].scatter(df_metrics['Recall'], df_metrics['Precision'], 
                         alpha=0.6, s=df_metrics['Support']*2, c='coral')
        axes[0,1].set_xlabel('Recall')
        axes[0,1].set_ylabel('Precision')
        axes[0,1].set_title('Precision vs Recall (Size = Support)')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        # 3. Performance Distribution
        metrics_for_hist = ['Precision', 'Recall', 'F1-Score']
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for i, metric in enumerate(metrics_for_hist):
            axes[1,0].hist(df_metrics[metric], bins=15, alpha=0.7, 
                          label=metric, color=colors[i])
        axes[1,0].set_xlabel('Score')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Distribution of Performance Metrics')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Support Distribution
        axes[1,1].hist(df_metrics['Support'], bins=20, color='gold', alpha=0.7)
        axes[1,1].set_xlabel('Number of Samples')
        axes[1,1].set_ylabel('Number of Classes')
        axes[1,1].set_title('Class Support Distribution')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_gradcam_visualization(self, image_path, save_path="gradcam_analysis.png"):
        """Generate GradCAM visualization for model interpretability"""
        print("🔥 Generating GradCAM visualization...")
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}. Creating sample visualization...")
            # Create a synthetic image for demonstration
            synthetic_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            synthetic_image.save("sample_leaf.jpg")
            image_path = "sample_leaf.jpg"
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            confidence, pred_class = torch.max(probs, 0)
        
        # Setup GradCAM
        target_layers = [self.model.blocks[-1].norm1]  # Last transformer block
        
        try:
            cam = GradCAM(model=self.model, target_layers=target_layers)
            
            # Generate CAM
            targets = [ClassifierOutputTarget(pred_class.item())]
            grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Convert image to RGB array
            rgb_img = np.array(image.resize((224, 224))) / 255.0
            
            # Create CAM visualization
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # Create comprehensive visualization
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('GradCAM Analysis - Plant Disease Detection', fontsize=16, fontweight='bold')
            
            # Original image
            axes[0,0].imshow(image)
            axes[0,0].set_title('Original Image')
            axes[0,0].axis('off')
            
            # Heatmap
            axes[0,1].imshow(grayscale_cam, cmap='jet')
            axes[0,1].set_title('Attention Heatmap')
            axes[0,1].axis('off')
            
            # Overlay
            axes[0,2].imshow(cam_image)
            axes[0,2].set_title('GradCAM Overlay')
            axes[0,2].axis('off')
            
            # Prediction results
            axes[1,0].axis('off')
            pred_text = f"""
PREDICTION RESULTS

🏆 Predicted Class:
{CLASS_NAMES[pred_class.item()].replace('___', ' - ').replace('_', ' ')}

📊 Confidence: {confidence.item()*100:.2f}%

🎯 Class Index: {pred_class.item()}
            """
            axes[1,0].text(0.1, 0.5, pred_text, fontsize=12, 
                          verticalalignment='center', 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            # Top 5 predictions
            top_5_indices = torch.topk(probs, 5).indices
            top_5_probs = probs[top_5_indices] * 100
            
            class_names_short = [CLASS_NAMES[i].split('___')[1] if '___' in CLASS_NAMES[i] 
                               else CLASS_NAMES[i] for i in top_5_indices]
            class_names_short = [name.replace('_', ' ')[:20] for name in class_names_short]
            
            axes[1,1].barh(range(5), top_5_probs.numpy(), color='skyblue')
            axes[1,1].set_yticks(range(5))
            axes[1,1].set_yticklabels(class_names_short)
            axes[1,1].set_xlabel('Confidence (%)')
            axes[1,1].set_title('Top 5 Predictions')
            axes[1,1].grid(axis='x', alpha=0.3)
            
            # Attention statistics
            axes[1,2].axis('off')
            attention_stats = f"""
ATTENTION ANALYSIS

🔥 Max Attention: {grayscale_cam.max():.3f}
📊 Mean Attention: {grayscale_cam.mean():.3f}
📈 Std Attention: {grayscale_cam.std():.3f}

📍 Focus Center:
Row: {np.unravel_index(grayscale_cam.argmax(), grayscale_cam.shape)[0]}
Col: {np.unravel_index(grayscale_cam.argmax(), grayscale_cam.shape)[1]}

🎲 Prediction Entropy:
{-torch.sum(probs * torch.log(probs + 1e-8)).item():.3f}
            """
            axes[1,2].text(0.1, 0.5, attention_stats, fontsize=11,
                          verticalalignment='center',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ GradCAM visualization saved as {save_path}")
            
        except Exception as e:
            print(f"❌ GradCAM generation failed: {str(e)}")
            print("Using alternative visualization method...")
            
            # Simple prediction visualization as fallback
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(image)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            # Top predictions
            top_5_indices = torch.topk(probs, 5).indices
            top_5_probs = probs[top_5_indices] * 100
            class_names_short = [CLASS_NAMES[i].replace('___', ' - ').replace('_', ' ')[:30] 
                               for i in top_5_indices]
            
            axes[1].barh(range(5), top_5_probs.numpy())
            axes[1].set_yticks(range(5))
            axes[1].set_yticklabels(class_names_short)
            axes[1].set_xlabel('Confidence (%)')
            axes[1].set_title('Model Predictions')
            
            plt.tight_layout()
            plt.savefig('simple_prediction_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def run_complete_evaluation(self, test_data_dir=None, sample_image_path=None):
        """Run complete model evaluation pipeline"""
        print("\n" + "="*60)
        print("🚀 STARTING COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        # Load model
        self.load_model()
        
        # Create or load test data
        test_data = self.create_sample_dataset(test_data_dir)
        
        # Generate confusion matrix and get predictions
        cm, predictions, labels, probs = self.generate_confusion_matrix(test_data)
        
        # Calculate comprehensive metrics
        metrics_dict, df_metrics = self.calculate_metrics(labels, predictions, probs)
        
        # Create performance visualizations
        self.visualize_performance_charts(df_metrics)
        
        # Generate GradCAM visualization
        if sample_image_path:
            self.generate_gradcam_visualization(sample_image_path)
        else:
            # Try to find a sample image or create one
            sample_images = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if sample_images:
                self.generate_gradcam_visualization(sample_images[0])
            else:
                print("ℹ️  No sample image found. Creating synthetic image for GradCAM demo...")
                self.generate_gradcam_visualization("nonexistent.jpg")  # Will create synthetic
        
        print("\n" + "="*60)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated Files:")
        print("📊 confusion_matrix.png - Confusion matrix visualization")
        print("📈 model_performance_analysis.png - Performance charts")
        print("🔥 gradcam_analysis.png - GradCAM visualization")
        print("📋 model_evaluation_metrics.csv - Detailed metrics")
        
        return metrics_dict, df_metrics

def main():
    """Main execution function"""
    print("🌱 Plant Disease Model Evaluation Tool")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = ModelEvaluator("plant_disease_model_final.pth")
    
    # Run complete evaluation
    # You can provide paths to your test data and sample image:
    # evaluator.run_complete_evaluation(
    #     test_data_dir="path/to/your/test/data",
    #     sample_image_path="path/to/sample/leaf/image.jpg"
    # )
    
    evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main()
