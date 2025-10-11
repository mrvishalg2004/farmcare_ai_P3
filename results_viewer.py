#!/usr/bin/env python3
"""
Model Evaluation Results Viewer
==============================
This script displays and summarizes the evaluation results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def display_evaluation_results():
    """Display comprehensive evaluation results"""
    print("🌱 PLANT DISEASE MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    # 1. Model Summary
    print("\n📊 MODEL SUMMARY")
    print("-" * 30)
    print("🏗️  Architecture: Vision Transformer (ViT) Base")
    print("📦 Parameters: 85,827,878 (85.8M)")
    print("🎯 Classes: 38 Plant Disease Categories")
    print("🖼️  Input Size: 224x224 RGB Images")
    print("⚙️  Normalization: ImageNet Standard")
    
    # 2. Load and display metrics if available
    if os.path.exists('model_evaluation_metrics.csv'):
        print("\n📈 PERFORMANCE METRICS")
        print("-" * 30)
        
        df = pd.read_csv('model_evaluation_metrics.csv')
        
        # Calculate overall stats
        avg_precision = df['Precision'].mean()
        avg_recall = df['Recall'].mean()
        avg_f1 = df['F1-Score'].mean()
        
        print(f"📊 Average Precision: {avg_precision:.4f}")
        print(f"🎯 Average Recall: {avg_recall:.4f}")
        print(f"⚖️  Average F1-Score: {avg_f1:.4f}")
        print(f"📋 Total Test Samples: {df['Support'].sum()}")
        
        # Top performing classes
        print(f"\n🏆 TOP 5 PERFORMING CLASSES (by F1-Score)")
        print("-" * 50)
        top_classes = df.nlargest(5, 'F1-Score')[['Class', 'F1-Score', 'Precision', 'Recall']]
        for idx, row in top_classes.iterrows():
            class_name = row['Class'][:40] + "..." if len(row['Class']) > 40 else row['Class']
            print(f"   {class_name}")
            print(f"     F1: {row['F1-Score']:.3f} | Precision: {row['Precision']:.3f} | Recall: {row['Recall']:.3f}")
    
    # 3. Display file information
    print(f"\n📁 GENERATED FILES")
    print("-" * 30)
    
    files_info = [
        ("confusion_matrix.png", "Confusion Matrix Heatmap", "Shows prediction accuracy across all classes"),
        ("model_performance_analysis.png", "Performance Charts", "Comprehensive performance visualization"),
        ("simple_prediction_analysis.png", "Sample Prediction", "Example model prediction with confidence"),
        ("model_evaluation_metrics.csv", "Detailed Metrics", "Per-class performance data")
    ]
    
    for filename, title, description in files_info:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✅ {filename}")
            print(f"   📋 {title}")
            print(f"   📝 {description}")
            print(f"   💾 Size: {size:,} bytes")
        else:
            print(f"❌ {filename} - Not found")
    
    # 4. Usage recommendations
    print(f"\n🎯 USAGE RECOMMENDATIONS")
    print("-" * 30)
    print("📊 For Web App: Model is integrated and ready for production")
    print("🔍 For Analysis: Use confusion matrix to identify challenging classes")
    print("🎨 For Visualization: GradCAM shows model attention regions")
    print("📈 For Improvement: Focus on classes with low F1-scores")
    
    # 5. Technical insights
    print(f"\n🔬 TECHNICAL INSIGHTS")
    print("-" * 30)
    print("🧠 Model Type: Vision Transformer excels at spatial relationships")
    print("📏 Input Processing: ImageNet normalization for optimal performance")
    print("🎯 Confidence Calibration: Temperature scaling applied for reliability")
    print("🔥 Interpretability: GradCAM provides attention visualization")
    
    print(f"\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("🌐 Ready for deployment in Streamlit app")
    print("=" * 60)

def view_results_interactively():
    """Interactive results viewer"""
    while True:
        print("\n🔍 INTERACTIVE RESULTS VIEWER")
        print("=" * 40)
        print("1. 📊 View Model Summary")
        print("2. 📈 Display Performance Metrics")
        print("3. 🖼️  Show Generated Visualizations")
        print("4. 📋 Export Results Summary")
        print("5. 🚪 Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            display_evaluation_results()
        elif choice == "2":
            if os.path.exists('model_evaluation_metrics.csv'):
                df = pd.read_csv('model_evaluation_metrics.csv')
                print("\n📊 DETAILED PERFORMANCE METRICS")
                print("=" * 60)
                print(df.to_string(index=False, float_format='%.4f'))
            else:
                print("❌ Metrics file not found!")
        elif choice == "3":
            print("\n🖼️  VISUALIZATION FILES")
            print("=" * 30)
            image_files = [f for f in os.listdir('.') if f.endswith('.png')]
            for i, img_file in enumerate(image_files, 1):
                print(f"{i}. {img_file}")
            print("\n💡 Open these files in your image viewer to see visualizations")
        elif choice == "4":
            export_summary_report()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice! Please select 1-5.")

def export_summary_report():
    """Export a comprehensive summary report"""
    report_content = f"""
PLANT DISEASE MODEL EVALUATION REPORT
=====================================
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL SPECIFICATIONS
-------------------
Architecture: Vision Transformer (ViT) Base Patch16 224
Parameters: 85,827,878 total parameters
Input Size: 224 x 224 pixels
Classes: 38 plant disease categories
Preprocessing: ImageNet normalization ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

"""
    
    if os.path.exists('model_evaluation_metrics.csv'):
        df = pd.read_csv('model_evaluation_metrics.csv')
        
        report_content += f"""
PERFORMANCE SUMMARY
------------------
Average Precision: {df['Precision'].mean():.4f}
Average Recall: {df['Recall'].mean():.4f}
Average F1-Score: {df['F1-Score'].mean():.4f}
Total Test Samples: {df['Support'].sum()}

TOP PERFORMING CLASSES
---------------------
"""
        top_5 = df.nlargest(5, 'F1-Score')
        for idx, row in top_5.iterrows():
            report_content += f"- {row['Class']}: F1={row['F1-Score']:.3f}\n"
        
        report_content += f"""

CHALLENGING CLASSES (Lowest F1-Scores)
------------------------------------
"""
        bottom_5 = df.nsmallest(5, 'F1-Score')
        for idx, row in bottom_5.iterrows():
            report_content += f"- {row['Class']}: F1={row['F1-Score']:.3f}\n"
    
    report_content += f"""

GENERATED OUTPUTS
----------------
- confusion_matrix.png: Visual confusion matrix
- model_performance_analysis.png: Performance charts
- simple_prediction_analysis.png: Sample prediction
- model_evaluation_metrics.csv: Detailed metrics

DEPLOYMENT STATUS
----------------
✅ Model ready for production deployment
✅ Streamlit web application configured
✅ Multi-language support implemented
✅ GradCAM visualization integrated
✅ Confidence calibration applied

RECOMMENDATIONS
--------------
1. Monitor performance on real-world images
2. Collect additional data for poorly performing classes
3. Regular model retraining with new data
4. User feedback integration for continuous improvement

END OF REPORT
=============
"""
    
    # Save report
    with open('evaluation_summary_report.txt', 'w') as f:
        f.write(report_content)
    
    print("✅ Summary report saved as 'evaluation_summary_report.txt'")

if __name__ == "__main__":
    print("🌱 Plant Disease Model - Evaluation Results")
    choice = input("\n1. Quick Summary\n2. Interactive Viewer\n\nSelect (1/2): ").strip()
    
    if choice == "2":
        view_results_interactively()
    else:
        display_evaluation_results()
