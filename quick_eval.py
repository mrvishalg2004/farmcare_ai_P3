#!/usr/bin/env python3
"""
Quick Model Evaluation Runner
============================
This script runs a quick evaluation of your plant disease model.
"""

import sys
import os
from model_evaluation import ModelEvaluator

def quick_evaluation():
    """Run a quick model evaluation"""
    print("🚀 Quick Model Evaluation")
    print("=" * 40)
    
    # Check if model file exists
    model_path = "plant_disease_model_final.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the .pth file is in the current directory")
        return
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(model_path)
        
        # Load model and get basic info
        evaluator.load_model()
        
        print("\n📊 Model Analysis Options:")
        print("1. Basic model info ✅ (completed above)")
        print("2. Confusion matrix analysis")
        print("3. Performance metrics calculation") 
        print("4. GradCAM visualization")
        print("5. Complete evaluation suite")
        
        choice = input("\nSelect option (1-5) or press Enter for complete evaluation: ")
        
        if choice == "2":
            # Just confusion matrix
            test_data = evaluator.create_sample_dataset()
            cm, predictions, labels, probs = evaluator.generate_confusion_matrix(test_data)
            print("✅ Confusion matrix saved as 'confusion_matrix.png'")
            
        elif choice == "3":
            # Performance metrics
            test_data = evaluator.create_sample_dataset()
            cm, predictions, labels, probs = evaluator.generate_confusion_matrix(test_data)
            metrics_dict, df_metrics = evaluator.calculate_metrics(labels, predictions, probs)
            evaluator.visualize_performance_charts(df_metrics)
            print("✅ Performance analysis completed!")
            
        elif choice == "4":
            # Just GradCAM
            evaluator.generate_gradcam_visualization("sample_image.jpg")
            print("✅ GradCAM visualization completed!")
            
        else:
            # Complete evaluation (default)
            print("\n🔄 Running complete evaluation...")
            evaluator.run_complete_evaluation()
            
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure all required packages are installed")
        print("2. Check if the model file is compatible")
        print("3. Verify you have sufficient memory")

if __name__ == "__main__":
    quick_evaluation()
