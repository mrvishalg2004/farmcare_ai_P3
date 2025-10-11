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
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import warnings
warnings.filterwarnings('ignore')
import streamlit.components.v1 as components
import google.generativeai as genai
import json
from langdetect import detect, LangDetectException

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
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)  # Using default configuration

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
        'confidence_radar_desc': 'Radar visualization of top predictions showing confidence distribution.',
        'plant_wise_analysis': '🌿 Plant-wise Disease Analysis',
        'model_performance': '⚡ Model Performance Insights',
        'total_classes': '🎯 Total Classes',
        'prediction_entropy': '🎲 Prediction Entropy',
        'top3_confidence': '🏆 Top-3 Confidence',
        'prediction_spread': '📊 Prediction Spread',
        'generating_heatmap': 'Generating GradCAM heatmap...',
        'heatmap_failed': '❌ Unable to generate heatmap visualization',
        'possible_solutions': '💡 **Possible solutions:**',
        'try_different_image': '- Try a different image with better lighting',
        'ensure_leaf_visible': '- Ensure the leaf is clearly visible and centered',
        'check_format': '- Check if the image format is supported (JPG, PNG)',
        'feature_importance': '🎯 Input Feature Importance',
        'confidence_distribution': '📈 Confidence Distribution Analysis',
        'very_high': 'Very High (80-100%)',
        'high': 'High (60-80%)',
        'medium': 'Medium (40-60%)',
        'low': 'Low (20-40%)',
        'very_low': 'Very Low (0-20%)',
        'prediction_entropy': '🎲 Prediction Entropy',
        'unreliable_prediction': '⚠️ **Prediction may be unreliable**',
        'high_entropy_warning': 'High entropy',
        'recommendations': '💡 **Recommendations**',
        'try_clearer_image': 'Try a clearer image with better lighting and focus',
        'excellent_confidence': '**Excellent Confidence**',
        'very_reliable': 'Very reliable prediction',
        'reliable_prediction': 'Reliable prediction',
        'consider_validation': 'Consider additional validation',
        'use_with_caution': 'Use with caution',
        'consider_retaking': 'Consider retaking image',
        'close_competition': '📊 **Close Competition**',
        'gap_with_second': 'Gap with 2nd prediction is only',
        'detailed_confidence': '🔍 Detailed Confidence Analysis',
        'prediction_gap': '🎯 Prediction Gap',
        'gap_help': 'Difference between top 2 predictions',
        'model_certainty': '📈 Model Certainty',
        'entropy_level': '🧠 Entropy Level'
    },
    'mr': {
        'title': '🌱 वनस्पती रोग ओळख साधन',
        'subtitle': 'रोग ओळखण्यासाठी आणि उपचार सूचना मिळवण्यासाठी वनस्पतीच्या पानाचा फोटो अपलोड करा',
        'upload_text': '📂 पानाचा फोटो खेचा आणि सोडा किंवा निवडा',
        'uploaded_image': 'अपलोड केलेला पान फोटो',
        'analysis_progress': '🔍 ML विश्लेषण सुरू आहे...',
        'preprocessing': 'फोटो प्रक्रिया करत आहे...',
        'extracting': 'वैशिष्ट्ये काढत आहे...',
        'detecting': 'रोग ओळख चालू आहे...',
        'calculating': 'विश्वास गुण मोजत आहे...',
        'visualizing': 'दृश्य तयार करत आहे...',
        'prediction_results': '🎯 अंदाज परिणाम',
        'predicted_disease': '🏆 अंदाजित रोग',
        'confidence_score': '📊 विश्वास गुण',
        'high_confidence': '🎯 उच्च विश्वास',
        'moderate_confidence': '⚠️ मध्यम विश्वास',
        'low_confidence': '❓ कमी विश्वास - फोटो पुन्हा घेण्याचा विचार करा',
        'treatment_plan': '💡 सुचवलेली उपचार योजना',
        'generating_treatment': 'वैयक्तिक उपचार शिफारसी तयार करत आहे...',
        'advanced_analysis': '🔬 प्रगत मॉडेल विश्लेषण आणि दृश्ये',
        'gradcam_tab': '🎯 GradCAM हीट मॅप',
        'probabilities_tab': '📊 तपशीलवार संभावना',
        'radar_tab': '🕸️ विश्वास रडार',
        'plant_analysis_tab': '🌿 वनस्पती विश्लेषण',
        'model_insights_tab': '⚡ मॉडेल अंतर्दृष्टी',
        'gradcam_title': '🔥 GradCAM दृश्य - ML काय पाहते',
        'gradcam_description': 'हा हीट मॅप दाखवतो की AI मॉडेलने आपल्या अंदाजासाठी पानाच्या कोणत्या भागांवर लक्ष केंद्रित केले.',
        'original_image': '🖼️ मूळ फोटो',
        'attention_heatmap': '🔥 लक्ष हीट मॅप',
        'ai_focus_overlay': '🎯 AI फोकस ओव्हरले',
        'language_select': '🌐 भाषा निवडा',
        'invalid_image': '⚠️ चुकीचा फोटो! कृपया वैध JPG/PNG फाईल अपलोड करा.',
        'healthy_plant': 'तुमच्या {plant_type} चे पान निरोगी आहे! उपचाराची गरज नाही. चांगल्या शेतकरी पद्धती पाळा.',
        'heatmap_success': '✅ **हीट मॅप यशस्वीरित्या तयार झाला!**',
        'how_to_read': '🔍 **हे कसे वाचावे**:',
        'red_areas': '🔴 **लाल/गरम भाग**: AI निर्णयासाठी सर्वात महत्वाचे क्षेत्र',
        'yellow_areas': '🟡 **पिवळे/उबदार भाग**: मध्यम महत्वाचे क्षेत्र',
        'blue_areas': '🔵 **निळे/थंड भाग**: कमी संबंधित क्षेत्र',
        'focus_explanation': '🎯 **फोकस**: AI ने आपल्या अंदाजासाठी हायलाइट केलेल्या भागांवर लक्ष केंद्रित केले',
        'peak_attention': '🎯 सर्वोच्च लक्ष गुण',
        'attention_center': '📍 **लक्ष केंद्र**: पंक्ती {row}, स्तंभ {col}',
        'comprehensive_analysis': '📈 सर्वसमावेशक संभाव्यता विश्लेषण',
        'top_predictions': '🏆 टॉप 10 अंदाज',
        'rank': 'स्थान',
        'disease': 'रोग',
        'confidence_percent': 'विश्वास (%)',
        'confidence_radar_desc': 'टॉप अंदाजांचे रडार दृश्य जे विश्वास वितरण दाखवते.',
        'plant_wise_analysis': '🌿 वनस्पती-वार विश्लेषण',
        'model_performance': '⚡ मॉडेल कार्यक्षमता अंतर्दृष्टी',
        'total_classes': '🎯 एकूण वर्ग',
        'prediction_entropy': '🎲 अंदाज एन्ट्रॉपी',
        'top3_confidence': '🏆 टॉप-3 विश्वास',
        'prediction_spread': '📊 अंदाज प्रसार',
        'generating_heatmap': 'GradCAM हीट मॅप तयार करत आहे...',
        'heatmap_failed': '❌ हीट मॅप दृश्य तयार करू शकत नाही',
        'possible_solutions': '💡 **शक्य उपाय:**',
        'try_different_image': '- चांगल्या प्रकाशासह वेगळा फोटो वापरून पहा',
        'ensure_leaf_visible': '- पान स्पष्टपणे दिसणारे आणि मध्यभागी असल्याची खात्री करा',
        'check_format': '- फोटो फॉर्मॅट समर्थित आहे का तपासा (JPG, PNG)',
        'feature_importance': '🎯 इनपुट वैशिष्ट्य महत्व',
        'confidence_distribution': '📈 विश्वास वितरण विश्लेषण',
        'very_high': 'अतिउच्च (80-100%)',
        'high': 'उच्च (60-80%)',
        'medium': 'मध्यम (40-60%)',
        'low': 'कमी (20-40%)',
        'very_low': 'अतिकमी (0-20%)'
    },
    'es': {
        'title': '🌱 Herramienta de Detección de Enfermedades de Plantas',
        'subtitle': 'Sube una imagen de hoja de planta para detectar enfermedades y obtener sugerencias de tratamiento',
        'upload_text': '📂 Arrastra y suelta o selecciona una imagen de hoja',
        'uploaded_image': 'Imagen de Hoja Subida',
        'analysis_progress': '🔍 Análisis de IA en Progreso...',
        'preprocessing': 'Preprocesando imagen...',
        'extracting': 'Extrayendo características...',
        'detecting': 'Ejecutando detección de enfermedades...',
        'calculating': 'Calculando puntuaciones de confianza...',
        'visualizing': 'Generando visualizaciones...',
        'prediction_results': '🎯 Resultados de Predicción',
        'predicted_disease': '🏆 Enfermedad Predicha',
        'confidence_score': '📊 Puntuación de Confianza',
        'high_confidence': '🎯 Alta Confianza',
        'moderate_confidence': '⚠️ Confianza Moderada',
        'low_confidence': '❓ Baja Confianza - Considera volver a tomar la imagen',
        'treatment_plan': '💡 Plan de Tratamiento Sugerido',
        'generating_treatment': 'Generando recomendaciones de tratamiento personalizadas...',
        'advanced_analysis': '🔬 Análisis Avanzado del Modelo y Visualizaciones',
        'gradcam_tab': '🎯 Mapa de Calor GradCAM',
        'probabilities_tab': '📊 Probabilidades Detalladas',
        'radar_tab': '🕸️ Radar de Confianza',
        'plant_analysis_tab': '🌿 Análisis de Plantas',
        'model_insights_tab': '⚡ Perspectivas del Modelo',
        'gradcam_title': '🔥 Visualización GradCAM - Lo que Ve la IA',
        'gradcam_description': 'Este mapa de calor muestra en qué partes de la hoja se enfocó el modelo de IA para su predicción.',
        'original_image': '🖼️ Imagen Original',
        'attention_heatmap': '🔥 Mapa de Calor de Atención',
        'ai_focus_overlay': '🎯 Superposición de Enfoque de IA',
        'language_select': '🌐 Seleccionar Idioma',
        'invalid_image': '⚠️ ¡Imagen inválida! Por favor sube un archivo JPG/PNG válido.',
        'healthy_plant': '¡Tu hoja de {plant_type} está sana! No se necesita tratamiento. Mantén buenas prácticas agrícolas.',
        'heatmap_success': '✅ **¡Mapa de Calor Generado con Éxito!**',
        'how_to_read': '🔍 **Cómo Leer Esto**:',
        'red_areas': '🔴 **Áreas Rojas/Calientes**: Regiones más importantes para la decisión de IA',
        'yellow_areas': '🟡 **Áreas Amarillas/Cálidas**: Regiones moderadamente importantes',
        'blue_areas': '🔵 **Áreas Azules/Frías**: Regiones menos relevantes',
        'focus_explanation': '🎯 **Enfoque**: La IA se concentró en las áreas resaltadas para hacer su predicción',
        'peak_attention': '🎯 Puntuación de Atención Máxima',
        'attention_center': '📍 **Centro de Atención**: Fila {row}, Columna {col}',
        'comprehensive_analysis': '📈 Análisis Integral de Probabilidades',
        'top_predictions': '🏆 Top 10 Predicciones',
        'rank': 'Rango',
        'disease': 'Enfermedad',
        'confidence_percent': 'Confianza (%)',
        'confidence_radar_desc': 'Visualización de radar de las mejores predicciones mostrando distribución de confianza.',
        'plant_wise_analysis': '🌿 Análisis por Tipo de Planta',
        'model_performance': '⚡ Perspectivas de Rendimiento del Modelo',
        'total_classes': '🎯 Clases Totales',
        'prediction_entropy': '🎲 Entropía de Predicción',
        'top3_confidence': '🏆 Confianza Top-3',
        'prediction_spread': '📊 Dispersión de Predicción'
    },
    'fr': {
        'title': '🌱 Outil de Détection des Maladies des Plantes',
        'subtitle': 'Téléchargez une image de feuille de plante pour détecter les maladies et obtenir des suggestions de traitement',
        'upload_text': '📂 Glissez-déposez ou sélectionnez une image de feuille',
        'uploaded_image': 'Image de Feuille Téléchargée',
        'analysis_progress': '🔍 Analyse IA en Cours...',
        'preprocessing': 'Prétraitement de l\'image...',
        'extracting': 'Extraction des caractéristiques...',
        'detecting': 'Exécution de la détection des maladies...',
        'calculating': 'Calcul des scores de confiance...',
        'visualizing': 'Génération des visualisations...',
        'prediction_results': '🎯 Résultats de Prédiction',
        'predicted_disease': '🏆 Maladie Prédite',
        'confidence_score': '📊 Score de Confiance',
        'high_confidence': '🎯 Haute Confiance',
        'moderate_confidence': '⚠️ Confiance Modérée',
        'low_confidence': '❓ Faible Confiance - Considérez reprendre l\'image',
        'treatment_plan': '💡 Plan de Traitement Suggéré',
        'generating_treatment': 'Génération de recommandations de traitement personnalisées...',
        'advanced_analysis': '🔬 Analyse Avancée du Modèle et Visualisations',
        'gradcam_tab': '🎯 Carte de Chaleur GradCAM',
        'probabilities_tab': '📊 Probabilités Détaillées',
        'radar_tab': '🕸️ Radar de Confiance',
        'plant_analysis_tab': '🌿 Analyse des Plantes',
        'model_insights_tab': '⚡ Insights du Modèle',
        'gradcam_title': '🔥 Visualisation GradCAM - Ce que Voit l\'IA',
        'gradcam_description': 'Cette carte de chaleur montre sur quelles parties de la feuille le modèle IA s\'est concentré pour sa prédiction.',
        'original_image': '🖼️ Image Originale',
        'attention_heatmap': '🔥 Carte de Chaleur d\'Attention',
        'ai_focus_overlay': '🎯 Superposition de Focus IA',
        'language_select': '🌐 Sélectionner la Langue',
        'invalid_image': '⚠️ Image invalide ! Veuillez télécharger un fichier JPG/PNG valide.',
        'healthy_plant': 'Votre feuille de {plant_type} est saine ! Aucun traitement nécessaire. Maintenez de bonnes pratiques agricoles.',
        'heatmap_success': '✅ **Carte de Chaleur Générée avec Succès !**',
        'how_to_read': '🔍 **Comment Lire Ceci** :',
        'red_areas': '🔴 **Zones Rouges/Chaudes** : Régions les plus importantes pour la décision IA',
        'yellow_areas': '🟡 **Zones Jaunes/Tièdes** : Régions modérément importantes',
        'blue_areas': '🔵 **Zones Bleues/Froides** : Régions moins pertinentes',
        'focus_explanation': '🎯 **Focus** : L\'IA s\'est concentrée sur les zones surlignées pour faire sa prédiction',
        'peak_attention': '🎯 Score d\'Attention Maximale',
        'attention_center': '📍 **Centre d\'Attention** : Ligne {row}, Colonne {col}',
        'comprehensive_analysis': '📈 Analyse Complète des Probabilités',
        'top_predictions': '🏆 Top 10 Prédictions',
        'rank': 'Rang',
        'disease': 'Maladie',
        'confidence_percent': 'Confiance (%)',
        'confidence_radar_desc': 'Visualisation radar des meilleures prédictions montrant la distribution de confiance.',
        'plant_wise_analysis': '🌿 Analyse par Type de Plante',
        'model_performance': '⚡ Insights de Performance du Modèle',
        'total_classes': '🎯 Classes Totales',
        'prediction_entropy': '🎲 Entropie de Prédiction',
        'top3_confidence': '🏆 Confiance Top-3',
        'prediction_spread': '📊 Répartition des Prédictions'
    },
    'de': {
        'title': '🌱 Pflanzenkrankheits-Erkennungstool',
        'subtitle': 'Laden Sie ein Pflanzenblattbild hoch, um Krankheiten zu erkennen und Behandlungsvorschläge zu erhalten',
        'upload_text': '📂 Blattbild ziehen und ablegen oder auswählen',
        'uploaded_image': 'Hochgeladenes Blattbild',
        'analysis_progress': '🔍 KI-Analyse läuft...',
        'preprocessing': 'Bildvorverarbeitung...',
        'extracting': 'Merkmale extrahieren...',
        'detecting': 'Krankheitserkennung läuft...',
        'calculating': 'Konfidenzwerte berechnen...',
        'visualizing': 'Visualisierungen generieren...',
        'prediction_results': '🎯 Vorhersageergebnisse',
        'predicted_disease': '🏆 Vorhergesagte Krankheit',
        'confidence_score': '📊 Konfidenzwert',
        'high_confidence': '🎯 Hohe Konfidenz',
        'moderate_confidence': '⚠️ Mittlere Konfidenz',
        'low_confidence': '❓ Niedrige Konfidenz - Erwägen Sie, das Bild erneut aufzunehmen',
        'treatment_plan': '💡 Vorgeschlagener Behandlungsplan',
        'generating_treatment': 'Personalisierte Behandlungsempfehlungen generieren...',
        'advanced_analysis': '🔬 Erweiterte Modellanalyse und Visualisierungen',
        'gradcam_tab': '🎯 GradCAM Heatmap',
        'probabilities_tab': '📊 Detaillierte Wahrscheinlichkeiten',
        'radar_tab': '🕸️ Konfidenz-Radar',
        'plant_analysis_tab': '🌿 Pflanzenanalyse',
        'model_insights_tab': '⚡ Modell-Einblicke',
        'gradcam_title': '🔥 GradCAM Visualisierung - Was die KI sieht',
        'gradcam_description': 'Diese Heatmap zeigt, auf welche Teile des Blattes sich das KI-Modell für seine Vorhersage konzentriert hat.',
        'original_image': '🖼️ Originalbild',
        'attention_heatmap': '🔥 Aufmerksamkeits-Heatmap',
        'ai_focus_overlay': '🎯 KI-Fokus-Overlay',
        'language_select': '🌐 Sprache auswählen',
        'invalid_image': '⚠️ Ungültiges Bild! Bitte laden Sie eine gültige JPG/PNG-Datei hoch.',
        'healthy_plant': 'Ihr {plant_type}-Blatt ist gesund! Keine Behandlung erforderlich. Behalten Sie gute landwirtschaftliche Praktiken bei.',
        'heatmap_success': '✅ **Heatmap erfolgreich generiert!**',
        'how_to_read': '🔍 **So lesen Sie dies**:',
        'red_areas': '🔴 **Rote/Heiße Bereiche**: Wichtigste Regionen für KI-Entscheidung',
        'yellow_areas': '🟡 **Gelbe/Warme Bereiche**: Mäßig wichtige Regionen',
        'blue_areas': '🔵 **Blaue/Kühle Bereiche**: Weniger relevante Regionen',
        'focus_explanation': '🎯 **Fokus**: Die KI konzentrierte sich auf die hervorgehobenen Bereiche für ihre Vorhersage',
        'peak_attention': '🎯 Höchste Aufmerksamkeitswert',
        'attention_center': '📍 **Aufmerksamkeitszentrum**: Zeile {row}, Spalte {col}',
        'comprehensive_analysis': '📈 Umfassende Wahrscheinlichkeitsanalyse',
        'top_predictions': '🏆 Top 10 Vorhersagen',
        'rank': 'Rang',
        'disease': 'Krankheit',
        'confidence_percent': 'Konfidenz (%)',
        'confidence_radar_desc': 'Radar-Visualisierung der Top-Vorhersagen mit Konfidenzverteilung.',
        'plant_wise_analysis': '🌿 Pflanzenweise Analyse',
        'model_performance': '⚡ Modell-Leistungs-Einblicke',
        'total_classes': '🎯 Gesamtklassen',
        'prediction_entropy': '🎲 Vorhersage-Entropie',
        'top3_confidence': '🏆 Top-3 Konfidenz',
        'prediction_spread': '📊 Vorhersage-Streuung'
    },
    'hi': {
        'title': '🌱 पौधों की बीमारी की पहचान उपकरण',
        'subtitle': 'बीमारी का पता लगाने और उपचार सुझाव प्राप्त करने के लिए पौधे की पत्ती की छवि अपलोड करें',
        'upload_text': '📂 पत्ती की छवि खींचें और छोड़ें या चुनें',
        'uploaded_image': 'अपलोड की गई पत्ती की छवि',
        'analysis_progress': '🔍 एआई विश्लेषण प्रगति में...',
        'preprocessing': 'छवि पूर्व-प्रसंस्करण...',
        'extracting': 'विशेषताएं निकालना...',
        'detecting': 'बीमारी का पता लगाना चल रहा है...',
        'calculating': 'विश्वास स्कोर की गणना...',
        'visualizing': 'विज़ुअलाइज़ेशन जेनरेट करना...',
        'prediction_results': '🎯 भविष्यवाणी परिणाम',
        'predicted_disease': '🏆 भविष्यवाणी की गई बीमारी',
        'confidence_score': '📊 विश्वास स्कोर',
        'high_confidence': '🎯 उच्च विश्वास',
        'moderate_confidence': '⚠️ मध्यम विश्वास',
        'low_confidence': '❓ कम विश्वास - छवि दोबारा लेने पर विचार करें',
        'treatment_plan': '💡 सुझाई गई उपचार योजना',
        'generating_treatment': 'व्यक्तिगत उपचार सिफारिशें जेनरेट कर रहे हैं...',
        'advanced_analysis': '🔬 उन्नत मॉडल विश्लेषण और विज़ुअलाइज़ेशन',
        'gradcam_tab': '🎯 GradCAM हीटमैप',
        'probabilities_tab': '📊 विस्तृत संभावनाएं',
        'radar_tab': '🕸️ विश्वास रडार',
        'plant_analysis_tab': '🌿 पौधे का विश्लेषण',
        'model_insights_tab': '⚡ मॉडल अंतर्दृष्टि',
        'gradcam_title': '🔥 GradCAM विज़ुअलाइज़ेशन - एआई क्या देखता है',
        'gradcam_description': 'यह हीटमैप दिखाता है कि एआई मॉडल ने अपनी भविष्यवाणी के लिए पत्ती के किन हिस्सों पर ध्यान केंद्रित किया।',
        'original_image': '🖼️ मूल छवि',
        'attention_heatmap': '🔥 ध्यान हीटमैप',
        'ai_focus_overlay': '🎯 एआई फोकस ओवरले',
        'language_select': '🌐 भाषा चुनें',
        'invalid_image': '⚠️ अमान्य छवि! कृपया एक वैध JPG/PNG फ़ाइल अपलोड करें।',
        'healthy_plant': 'आपकी {plant_type} की पत्ती स्वस्थ है! कोई उपचार की आवश्यकता नहीं। अच्छी कृषि प्रथाओं को बनाए रखें।',
        'heatmap_success': '✅ **हीटमैप सफलतापूर्वक जेनरेट हुआ!**',
        'how_to_read': '🔍 **इसे कैसे पढ़ें**:',
        'red_areas': '🔴 **लाल/गर्म क्षेत्र**: एआई निर्णय के लिए सबसे महत्वपूर्ण क्षेत्र',
        'yellow_areas': '🟡 **पीले/गर्म क्षेत्र**: मध्यम महत्वपूर्ण क्षेत्र',
        'blue_areas': '🔵 **नीले/ठंडे क्षेत्र**: कम प्रासंगिक क्षेत्र',
        'focus_explanation': '🎯 **फोकस**: एआई ने अपनी भविष्यवाणी के लिए हाइलाइट किए गए क्षेत्रों पर ध्यान केंद्रित किया',
        'peak_attention': '🎯 अधिकतम ध्यान स्कोर',
        'attention_center': '📍 **ध्यान केंद्र**: पंक्ति {row}, स्तंभ {col}',
        'comprehensive_analysis': '📈 व्यापक संभावना विश्लेषण',
        'top_predictions': '🏆 शीर्ष 10 भविष्यवाणियां',
        'rank': 'रैंक',
        'disease': 'बीमारी',
        'confidence_percent': 'विश्वास (%)',
        'confidence_radar_desc': 'शीर्ष भविष्यवाणियों का रडार विज़ुअलाइज़ेशन जो विश्वास वितरण दिखाता है।',
        'plant_wise_analysis': '🌿 पौधे-वार विश्लेषण',
        'model_performance': '⚡ मॉडल प्रदर्शन अंतर्दृष्टि',
        'total_classes': '🎯 कुल वर्ग',
        'prediction_entropy': '🎲 भविष्यवाणी एन्ट्रॉपी',
        'top3_confidence': '🏆 शीर्ष-3 विश्वास',
        'prediction_spread': '📊 भविष्यवाणी प्रसार'
    },
    'zh': {
        'title': '🌱 植物病害检测工具',
        'subtitle': '上传植物叶片图像以检测病害并获取治疗建议',
        'upload_text': '📂 拖放或选择叶片图像',
        'uploaded_image': '已上传的叶片图像',
        'analysis_progress': '🔍 AI分析进行中...',
        'preprocessing': '图像预处理中...',
        'extracting': '提取特征中...',
        'detecting': '运行病害检测中...',
        'calculating': '计算置信度分数中...',
        'visualizing': '生成可视化中...',
        'prediction_results': '🎯 预测结果',
        'predicted_disease': '🏆 预测病害',
        'confidence_score': '📊 置信度分数',
        'high_confidence': '🎯 高置信度',
        'moderate_confidence': '⚠️ 中等置信度',
        'low_confidence': '❓ 低置信度 - 考虑重新拍摄图像',
        'treatment_plan': '💡 建议治疗方案',
        'generating_treatment': '正在生成个性化治疗建议...',
        'advanced_analysis': '🔬 高级模型分析和可视化',
        'gradcam_tab': '🎯 GradCAM热图',
        'probabilities_tab': '📊 详细概率',
        'radar_tab': '🕸️ 置信度雷达',
        'plant_analysis_tab': '🌿 植物分析',
        'model_insights_tab': '⚡ 模型洞察',
        'gradcam_title': '🔥 GradCAM可视化 - AI看到的内容',
        'gradcam_description': '此热图显示AI模型在进行预测时关注叶片的哪些部分。',
        'original_image': '🖼️ 原始图像',
        'attention_heatmap': '🔥 注意力热图',
        'ai_focus_overlay': '🎯 AI焦点叠加',
        'language_select': '🌐 选择语言',
        'invalid_image': '⚠️ 无效图像！请上传有效的JPG/PNG文件。',
        'healthy_plant': '您的{plant_type}叶片是健康的！无需治疗。保持良好的农业实践。',
        'heatmap_success': '✅ **热图生成成功！**',
        'how_to_read': '🔍 **如何阅读**：',
        'red_areas': '🔴 **红色/热点区域**：AI决策最重要的区域',
        'yellow_areas': '🟡 **黄色/温暖区域**：中等重要的区域',
        'blue_areas': '🔵 **蓝色/冷点区域**：相关性较低的区域',
        'focus_explanation': '🎯 **焦点**：AI专注于突出显示的区域进行预测',
        'peak_attention': '🎯 峰值注意力分数',
        'attention_center': '📍 **注意力中心**：行 {row}，列 {col}',
        'comprehensive_analysis': '📈 综合概率分析',
        'top_predictions': '🏆 前10预测',
        'rank': '排名',
        'disease': '病害',
        'confidence_percent': '置信度 (%)',
        'confidence_radar_desc': '顶级预测的雷达可视化，显示置信度分布。',
        'plant_wise_analysis': '🌿 按植物类型分析',
        'model_performance': '⚡ 模型性能洞察',
        'total_classes': '🎯 总类别数',
        'prediction_entropy': '🎲 预测熵',
        'top3_confidence': '🏆 前3置信度',
        'prediction_spread': '📊 预测分布',
        'generating_heatmap': '正在生成GradCAM热图...',
        'heatmap_failed': '❌ 无法生成热图可视化',
        'possible_solutions': '💡 **可能的解决方案：**',
        'try_different_image': '- 尝试使用光线更好的不同图像',
        'ensure_leaf_visible': '- 确保叶片清晰可见且居中',
        'check_format': '- 检查图像格式是否受支持（JPG、PNG）',
        'feature_importance': '🎯 输入特征重要性',
        'confidence_distribution': '📈 置信度分布分析',
        'very_high': '极高 (80-100%)',
        'high': '高 (60-80%)',
        'medium': '中等 (40-60%)',
        'low': '低 (20-40%)',
        'very_low': '极低 (0-20%)'
    }
}

def get_text(key, lang='en', **kwargs):
    """Get translated text for the current language"""
    text = TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, TRANSLATIONS['en'].get(key, key))
    if kwargs:
        try:
            return text.format(**kwargs)
        except:
            return text
    return text

# -----------------------------
# Gemini AI API Configuration
# -----------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_URL = os.environ.get("GEMINI_API_URL")

# -----------------------------
# Plant Disease Classes (38 classes)
# -----------------------------
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASS_NAMES))
    # Ensure the model file 'plant_disease_model_final.pth' is in the same directory
    state_dict = torch.load("plant_disease_model_final.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# -----------------------------
# Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet standards
])

# -----------------------------
# Confidence Calibration Functions
# -----------------------------
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

# -----------------------------
# Advanced Visualization Functions
# -----------------------------

def generate_gradcam_heatmap(model, image_tensor, target_class_idx, image_rgb):
    """Generate GradCAM heatmap for model interpretation"""
    try:
        possible_target_layers = [
            [model.head],
            [model.blocks[-1].norm2],
            [model.blocks[-1].mlp],
        ]
        
        for target_layers in possible_target_layers:
            try:
                cam = GradCAM(model=model, target_layers=target_layers)
                targets = [ClassifierOutputTarget(target_class_idx)]
                grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                
                rgb_img = np.array(image_rgb) / 255.0
                rgb_img_resized = cv2.resize(rgb_img, (224, 224))
                
                cam_image = show_cam_on_image(rgb_img_resized, grayscale_cam, use_rgb=True)
                
                return cam_image, grayscale_cam
                
            except Exception as layer_error:
                continue
        
        st.info("🔄 Using alternative gradient-based visualization...")
        return generate_alternative_heatmap(model, image_tensor, target_class_idx, image_rgb)
        
    except Exception as e:
        st.warning(f"Heatmap generation failed: {str(e)}")
        return generate_alternative_heatmap(model, image_tensor, target_class_idx, image_rgb)

def generate_alternative_heatmap(model, image_tensor, target_class_idx, image_rgb):
    """Generate alternative heatmap using gradient-based method"""
    try:
        image_tensor.requires_grad_(True)
        output = model(image_tensor)
        target_score = output[0, target_class_idx]
        
        model.zero_grad()
        target_score.backward(retain_graph=True)
        
        gradients = image_tensor.grad.data
        heatmap = torch.abs(gradients).mean(dim=1).squeeze().cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = plt.cm.jet(heatmap_resized)[:,:,:3]
        
        rgb_img = np.array(image_rgb) / 255.0
        rgb_img_resized = cv2.resize(rgb_img, (224, 224))
        
        alpha = 0.4
        cam_image = alpha * heatmap_colored + (1 - alpha) * rgb_img_resized
        cam_image = np.clip(cam_image, 0, 1)
        
        return cam_image, heatmap_resized
        
    except Exception as e:
        st.error(f"Alternative heatmap generation failed: {str(e)}")
        return None, None

def create_confidence_radar_chart(probs, class_names, top_k=8):
    """Create radar chart for top predictions"""
    top_indices = torch.topk(probs, top_k).indices
    top_probs = probs[top_indices].numpy() * 100
    top_classes = [class_names[i] for i in top_indices]
    
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
        font=dict(size=12)
    )
    
    return fig

def create_probability_distribution_chart(probs, class_names):
    """Create detailed probability distribution chart"""
    prob_data = pd.DataFrame({
        'Class': class_names,
        'Probability': probs.numpy() * 100,
        'Plant_Type': [name.split('___')[0].replace('_', ' ') if '___' in name else 'Unknown' for name in class_names],
        'Disease': [name.split('___')[1].replace('_', ' ') if '___' in name else name.replace('_', ' ') for name in class_names]
    })
    
    prob_data = prob_data.sort_values('Probability', ascending=True)
    
    fig = px.bar(
        prob_data.tail(15),  # Show top 15
        x='Probability',
        y='Class',
        color='Plant_Type',
        title='Top 15 Disease Predictions with Confidence Levels',
        labels={'Probability': 'Confidence (%)', 'Class': 'Disease Class'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        showlegend=True
    )
    
    return fig

def create_plant_wise_summary(probs, class_names):
    """Create plant-wise disease probability summary"""
    plant_summary = {}
    
    for i, class_name in enumerate(class_names):
        if '___' in class_name:
            plant, disease = class_name.split('___', 1)
            plant = plant.replace('_', ' ')
            disease = disease.replace('_', ' ')
            
            if plant not in plant_summary:
                plant_summary[plant] = []
            
            plant_summary[plant].append({
                'disease': disease,
                'probability': probs[i].item() * 100
            })
    
    return plant_summary

def create_feature_importance_chart(model, image_tensor):
    """Create feature importance visualization using gradients"""
    try:
        image_tensor.requires_grad_(True)
        
        output = model(image_tensor)
        
        pred_class = output.argmax(dim=1)
        loss = output[0, pred_class]
        loss.backward()
        
        gradients = image_tensor.grad.data
        importance = torch.abs(gradients).mean(dim=1).squeeze().cpu().numpy()
        
        return importance
        
    except Exception as e:
        st.warning(f"Feature importance calculation failed: {str(e)}")
        return None

# -----------------------------
# Function to call Gemini AI for treatment plan
# -----------------------------
def get_treatment_plan(disease_name, lang='en'):
    if "healthy" in disease_name.lower():
        plant_type = disease_name.split('___')[0].replace('_', ' ') if '___' in disease_name else "plant"
        return f"✅ {get_text('healthy_plant', lang, plant_type=plant_type)}"

    if '___' in disease_name:
        plant_type = disease_name.split('___')[0].replace('_', ' ')
        disease_only = disease_name.split('___')[1].replace('_', ' ')
        display_name = f"{plant_type} - {disease_only}"
    else:
        plant_type = "plant"
        disease_only = disease_name.replace('_', ' ')
        display_name = disease_only

    language_prompts = {
        'en': f"""You are an agricultural expert. Provide a simple, easy-to-understand treatment plan for {display_name} in English.""",
        'es': f"""Eres un experto agrícola. Proporciona un plan de tratamiento simple y fácil de entender para {display_name} en español.""",
        'fr': f"""Vous êtes un expert agricole. Fournissez un plan de traitement simple et facile à comprendre pour {display_name} en français.""",
        'de': f"""Sie sind ein Landwirtschaftsexperte. Erstellen Sie einen einfachen, leicht verständlichen Behandlungsplan für {display_name} auf Deutsch.""",
        'hi': f"""आप एक कृषि विशेषज्ञ हैं। {display_name} के लिए हिंदी में एक सरल, समझने योग्य उपचार योजना प्रदान करें।""",
        'mr': f"""तुम्ही एक कृषी तज्ञ आहात। {display_name} साठी मराठीत एक सोपी, समजण्यायोग्य उपचार योजना प्रदान करा।""",
        'zh': f"""您是农业专家。请为{display_name}提供一个简单易懂的中文治疗方案。"""
    }
    
    base_prompt = language_prompts.get(lang, language_prompts['en'])
    
    prompt_text = f"""
    {base_prompt}
    
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

    # Use the imported genai library instead of direct API requests
    try:
        # Configure the API key
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Use the latest Gemini model for better reliability
        model = genai.GenerativeModel('gemini-flash-latest')
        
        # Generate the treatment plan
        response = model.generate_content(prompt_text)
        
        if response and hasattr(response, 'text'):
            return response.text
        else:
            # Fallback to direct API if needed
            return generate_fallback_treatment_plan(display_name)
    except Exception as e:
        # Log the error for debugging
        print(f"Gemini API error: {str(e)}")
        
        # Generate a fallback treatment plan
        return generate_fallback_treatment_plan(display_name)

def generate_fallback_treatment_plan(disease_name):
    """Generate a fallback treatment plan when the API fails"""
    if "Apple" in disease_name and "Black Rot" in disease_name:
        return """
## Apple - Black Rot - Quick Treatment

**What you'll see:**
- Dark circular lesions on fruits with concentric rings
- Rotted areas that become black and leathery
- Mummified fruits that stay attached to the tree

**How to treat:**
1. Remove and destroy all infected fruits, leaves, and cankers
2. Apply fungicide sprays (copper-based or organic) as directed
3. Prune affected branches, sterilizing tools between cuts
4. Continue protective spray program throughout the growing season

**How to prevent:**
- Maintain good orchard sanitation by removing fallen leaves and fruits
- Ensure proper spacing between trees for good air circulation
- Apply dormant sprays before bud break in early spring
- Choose resistant apple varieties when planting new trees
        """
    
    elif "Tomato" in disease_name and "Mosaic Virus" in disease_name:
        return """
## Tomato - Mosaic Virus - Quick Treatment

**What you'll see:**
- Yellow and green mottled pattern on leaves
- Stunted plant growth and malformed leaves
- Reduced fruit production with possible fruit discoloration

**How to treat:**
1. Unfortunately, there's no cure for viral infections in plants
2. Remove and destroy infected plants to prevent spreading
3. Disinfect garden tools with 10% bleach solution after handling
4. Control insect vectors like aphids with organic insecticidal soap

**How to prevent:**
- Purchase certified disease-free seeds and plants
- Wash hands after handling tobacco products before working with plants
- Control weeds that may harbor the virus
- Rotate crops and avoid planting tomatoes in the same location
        """
    
    else:
        return f"""
## {disease_name} - Quick Treatment

**What you'll see:**
- Discoloration and lesions on leaves or fruit
- Possible wilting or stunted growth
- Unusual spots or powdery substances on plant surfaces

**How to treat:**
1. Remove and destroy heavily infected plant parts
2. Apply appropriate fungicide or pesticide based on your location's recommendations
3. Ensure proper watering (avoid overhead watering if possible)
4. Improve air circulation around plants by proper spacing and pruning

**How to prevent:**
- Practice crop rotation to reduce pathogen buildup in soil
- Use disease-resistant varieties when available
- Maintain good garden sanitation by removing plant debris
- Monitor plants regularly for early signs of disease
        """


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="🌱 Plant Disease Detection & Treatment",
    page_icon="🌿",
    layout="wide"
)

# Initialize chat widget state in session
if 'enable_tawk' not in st.session_state:
    st.session_state.enable_tawk = True


# -----------------------------
# Language Selection Sidebar
# -----------------------------
st.sidebar.markdown("### 🌐 Language / भाषा / मराठी / 语言")
selected_language = st.sidebar.selectbox(
    "Choose your language / भाषा निवडा / भाषा निवडा:",
    options=list(LANGUAGES.keys()),
    index=0,
    key='language_selector'
)
current_lang = LANGUAGES[selected_language]

# Chat support toggle and functional Tawk.to widget
st.sidebar.markdown("---")
st.sidebar.markdown("### 💬 Chat Support")
enable_chat = st.sidebar.checkbox("Enable Live Chat Support", key="enable_tawk")

if enable_chat:
    st.sidebar.success("✅ Live chat support is enabled")
    # This correctly loads the widget only when the box is checked
    st.markdown(
    """
    <script type="text/javascript">
    var Tawk_API=Tawk_API||{}, Tawk_LoadStart=new Date();
    (function(){
    var s1=document.createElement("script"),s0=document.getElementsByTagName("script")[0];
    s1.async=true;
    s1.src='https://embed.tawk.to/6727675b4304e3196adc83e8/1iboung7u';
    s1.charset='UTF-8';
    s1.setAttribute('crossorigin','*');
    s0.parentNode.insertBefore(s1,s0);
    })();
    </script>
    """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.info("ℹ️ Chat support is disabled. Check the box to get help.")
    
if current_lang != 'en':
    st.sidebar.success(f"Language set to: {selected_language}")
    # Additional language-specific messages can go here

# Also add language selector to main area for better visibility 
col_lang, col_spacer = st.columns([1, 3])
with col_lang:
    if current_lang == 'mr':
        st.selectbox(
            "🌐 भाषा:",
            options=list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index(selected_language),
            key='main_language_selector',
            disabled=True,
            help="साइडबारमध्ये भाषा बदला"
        )
    else:
        st.selectbox(
            "🌐 Language:",
            options=list(LANGUAGES.keys()),
            index=list(LANGUAGES.keys()).index(selected_language),
            key='main_language_selector',
            disabled=True,
            help="Change language in the sidebar"
        )

# -----------------------------
# Custom CSS for UI
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #f0f9f0, #ffffff);
    font-family: 'Segoe UI', sans-serif;
}
.header-title {
    font-size: 40px !important;
    font-weight: bold;
    color: #2e7d32;
    text-align: center;
    margin-bottom: 5px;
}
.header-subtitle {
    font-size: 18px !important;
    color: #1b5e20;
    text-align: center;
    margin-bottom: 30px;
}
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* Chatbot CSS */
.chat-icon {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background-color: #4CAF50;
    color: white;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    z-index: 1000;
    transition: all 0.3s ease;
}

.chat-icon:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 12px rgba(0,0,0,0.3);
}

.chat-window {
    position: fixed;
    bottom: 90px;
    right: 20px;
    width: 350px;
    height: 500px;
    background-color: white;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    display: flex;
    flex-direction: column;
    z-index: 999;
    overflow: hidden;
}

.chat-header {
    background-color: #4CAF50;
    color: white;
    padding: 15px;
    font-weight: bold;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.chat-messages {
    padding: 15px;
    overflow-y: auto;
    flex-grow: 1;
}

.chat-input-container {
    padding: 10px;
    border-top: 1px solid #e0e0e0;
    display: flex;
}

.chat-input {
    flex-grow: 1;
    padding: 10px;
    border: 1px solid #e0e0e0;
    border-radius: 20px;
    margin-right: 10px;
}

.chat-send {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 20px;
    padding: 10px 15px;
    cursor: pointer;
}

.user-message {
    background: linear-gradient(135deg, #0078FF, #00C6FF);
    color: white;
    border-radius: 18px 18px 0 18px;
    padding: 12px 18px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    animation: fadeIn 0.3s ease forwards;
    margin: 5px 0;
    max-width: 80%;
    align-self: flex-end;
    margin-left: auto;
}

.bot-message {
    background: white;
    color: #333;
    border-radius: 18px 18px 18px 0;
    padding: 12px 18px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    animation: fadeIn 0.3s ease forwards;
    border-left: 3px solid #00cc66;
    margin: 5px 0;
    max-width: 80%;
    align-self: flex-start;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(f'<div class="header-title">{get_text("title", current_lang)}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="header-subtitle">{get_text("subtitle", current_lang)}</div>', unsafe_allow_html=True)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(get_text("upload_text", current_lang), type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except:
        st.error(get_text("invalid_image", current_lang))
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(f'<div class="card"><h4>{get_text("uploaded_image", current_lang)}</h4></div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown(f'<div class="card"><h4>{get_text("analysis_progress", current_lang)}</h4></div>', unsafe_allow_html=True)
        
        analysis_steps = [
            get_text("preprocessing", current_lang),
            get_text("extracting", current_lang),
            get_text("detecting", current_lang),
            get_text("calculating", current_lang),
            get_text("visualizing", current_lang)
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, step in enumerate(analysis_steps):
            status_text.text(step)
            progress_bar.progress((i + 1) / len(analysis_steps))
            time.sleep(0.3)

        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = calibrate_confidence(outputs, temperature=1.5)[0]
            confidence, pred_class = torch.max(probs, 0)
            is_reliable, entropy_score = validate_prediction_quality(probs)

        progress_bar.empty()
        status_text.empty()
        
        st.markdown(f'<div class="card"><h4>{get_text("prediction_results", current_lang)}</h4></div>', unsafe_allow_html=True)
        
        predicted_class = CLASS_NAMES[pred_class.item()]
        confidence_score = confidence.item() * 100
        
        conf_level, conf_icon = get_confidence_level(confidence_score, entropy_score)
        
        def format_disease_name(class_name):
            if '___' in class_name:
                plant_type, disease_name = class_name.split('___', 1)
                plant_clean = plant_type.replace('_', ' ').title()
                disease_clean = disease_name.replace('_', ' ').title()
                return plant_clean, disease_clean
            else:
                return "Plant", class_name.replace('_', ' ').title()
        
        plant_name, disease_name_formatted = format_disease_name(predicted_class)
        
        st.markdown(f"### 🏆 {get_text('predicted_disease', current_lang)}")
        
        st.markdown(f"""
        <div style='
            background: linear-gradient(90deg, #e8f5e8, #f0f9f0);
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #4caf50;
            margin: 10px 0;
        '>
            <div style='font-size: 20px; font-weight: bold; color: #2e7d32; margin-bottom: 5px;'>
                🌱 {plant_name}
            </div>
            <div style='font-size: 18px; color: #1b5e20; font-weight: 600;'>
                🦠 {disease_name_formatted}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col2b, col2c = st.columns(2)
        with col2b:
            st.metric(get_text("confidence_score", current_lang), f"{confidence_score:.2f}%")
        with col2c:
            st.metric("🎲 Entropy", f"{entropy_score:.3f}")
        
        if not is_reliable:
            st.error(f"⚠️ **Prediction may be unreliable** (High entropy: {entropy_score:.3f})")
            st.info("💡 **Recommendations**: Try a clearer image with better lighting and focus")
        
        if conf_level == "very_high":
            st.success(f"{conf_icon} **Excellent Confidence**: {confidence_score:.2f}% - Very reliable prediction")
        elif conf_level == "high":
            st.success(f"{conf_icon} **High Confidence**: {confidence_score:.2f}% - Reliable prediction")
        elif conf_level == "moderate":
            st.warning(f"{conf_icon} **Moderate Confidence**: {confidence_score:.2f}% - Consider additional validation")
        elif conf_level == "low":
            st.warning(f"{conf_icon} **Low Confidence**: {confidence_score:.2f}% - Use with caution")
        else:
            st.error(f"{conf_icon} **Very Low Confidence**: {confidence_score:.2f}% - Consider retaking image")

    # -----------------------------
    # Treatment Plan Generation
    # -----------------------------
    with st.spinner(get_text("generating_treatment", current_lang)):
        treatment_plan = get_treatment_plan(predicted_class, current_lang)

    st.subheader(get_text("treatment_plan", current_lang))
    st.markdown(f"{treatment_plan}")

    # -----------------------------
    # Advanced Visualizations Section
    # -----------------------------
    st.markdown("---")
    st.markdown(f"## {get_text('advanced_analysis', current_lang)}")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        get_text("gradcam_tab", current_lang), 
        get_text("probabilities_tab", current_lang), 
        get_text("radar_tab", current_lang), 
        get_text("plant_analysis_tab", current_lang), 
        get_text("model_insights_tab", current_lang)
    ])
    
    with tab1:
        st.subheader(get_text("gradcam_title", current_lang))
        st.write(get_text("gradcam_description", current_lang))
        
        with st.spinner(get_text('generating_heatmap', current_lang)):
            cam_image, grayscale_cam = generate_gradcam_heatmap(
                model, img_tensor, pred_class.item(), image
            )
        
        if cam_image is not None and isinstance(grayscale_cam, np.ndarray):
            col_grad1, col_grad2, col_grad3 = st.columns(3)

            with col_grad1:
                st.image(image, caption=get_text("original_image", current_lang), use_column_width=True)
            
            with col_grad2:
                heatmap_colored = plt.cm.jet(grayscale_cam)[:,:,:3]
                st.image(heatmap_colored, caption=get_text("attention_heatmap", current_lang), use_column_width=True)
            
            with col_grad3:
                st.image(cam_image, caption=get_text("ai_focus_overlay", current_lang), use_column_width=True)

            st.success(get_text('heatmap_success', current_lang))
            st.info(get_text('how_to_read', current_lang))
            st.markdown(f"""
            - {get_text('red_areas', current_lang)}
            - {get_text('yellow_areas', current_lang)}
            - {get_text('blue_areas', current_lang)}
            """)
        else:
            st.error(get_text('heatmap_failed', current_lang))

    with tab2:
        st.subheader(get_text('comprehensive_analysis', current_lang))
        
        prob_fig = create_probability_distribution_chart(probs, CLASS_NAMES)
        st.plotly_chart(prob_fig, use_container_width=True)
        
        st.subheader(get_text('top_predictions', current_lang))
        top_10_indices = torch.topk(probs, 10).indices
        top_10_data = []
        
        for i, idx in enumerate(top_10_indices):
            class_name = CLASS_NAMES[idx.item()]
            prob = probs[idx.item()].item() * 100
            plant_name_disp, disease_name_disp = format_disease_name(class_name)
            
            top_10_data.append({
                get_text("rank", current_lang): i + 1,
                get_text("disease", current_lang): f"{plant_name_disp} - {disease_name_disp}",
                get_text("confidence_percent", current_lang): f"{prob:.2f}%"
            })
        
        df_top10 = pd.DataFrame(top_10_data)
        st.dataframe(df_top10, use_container_width=True)
        
    with tab3:
        st.subheader(get_text('radar_tab', current_lang))
        st.write(get_text('confidence_radar_desc', current_lang))
        
        radar_fig = create_confidence_radar_chart(probs, CLASS_NAMES, top_k=8)
        st.plotly_chart(radar_fig, use_container_width=True)
        
    with tab4:
        st.subheader(get_text('plant_wise_analysis', current_lang))
        
        plant_summary = create_plant_wise_summary(probs, CLASS_NAMES)
        
        for plant, diseases in plant_summary.items():
            if any(d['probability'] > 5 for d in diseases):
                with st.expander(f"🌱 {plant} - Detailed Analysis"):
                    diseases_df = pd.DataFrame(diseases).sort_values('probability', ascending=False)
                    diseases_df['probability'] = diseases_df['probability'].apply(lambda x: f"{x:.2f}%")
                    st.dataframe(diseases_df, use_container_width=True)
    
    with tab5:
        st.subheader(get_text('model_performance', current_lang))
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.metric(get_text('total_classes', current_lang), len(CLASS_NAMES))
        
        with col_stat2:
            st.metric(get_text('prediction_entropy', current_lang), f"{entropy_score:.3f}")
        
        with col_stat3:
            top_3_sum = torch.topk(probs, 3).values.sum().item() * 100
            st.metric(get_text('top3_confidence', current_lang), f"{top_3_sum:.1f}%")

# -----------------------------
# Gemini AI Chatbot Implementation
# -----------------------------

# Initialize Gemini AI model
@st.cache_resource
def init_gemini_model():
    """Initialize and return the Gemini AI model"""
    # Using gemini-flash-latest which is confirmed to work with this API key
    model = genai.GenerativeModel('gemini-flash-latest')
    return model

# Initialize chat history in session state if not already there
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'chat_visible' not in st.session_state:
    st.session_state.chat_visible = False

def toggle_chat():
    """Toggle chat window visibility with enhanced animation effects"""
    st.session_state.chat_visible = not st.session_state.chat_visible
    st.rerun()  # Force rerun to apply changes immediately

# Check URL parameters to show chat if requested via URL
params = st.query_params
if "show_chat" in params and params["show_chat"] == "true":
    st.session_state.chat_visible = True
    # Remove parameter after processing
    new_params = dict(params)
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
    try:
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
        
        # First try the standard approach with system prompt and user input
        try:
            response = model.generate_content([
                {"text": system_prompt},
                {"text": user_input}
            ])
            return response.text
        except Exception as e:
            # If that fails, try a simple direct request
            try:
                response = model.generate_content(user_input)
                return response.text
            except Exception as e2:
                # If all API calls fail, return a helpful error message
                return "I'm sorry, I'm having trouble connecting to my knowledge base. Please try again in a moment."
    except Exception as e:
        # Handle any other errors
        return "I'm sorry, I'm having trouble responding right now. This might be due to API limits or connection issues. Please try again in a moment or ask a different question about plant care or diseases."

def process_chat_input():
    """Process user chat input and generate a response"""
    user_input = st.session_state.user_input
    if user_input.strip():
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate AI response
        with st.spinner("Thinking..."):
            ai_response = generate_gemini_response(user_input)
        
        # Add AI response to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": ai_response
        })
        
        # Clear the input box
        st.session_state.user_input = ""

# Create the enhanced chat UI with HTML and JavaScript
chat_html = """
<div class="bg-gradient"></div>
<div id="particles-container" class="particles">
    <!-- Particles will be dynamically created -->
</div>

<div id="chat-icon" class="chat-button" onclick="toggleChat()">
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="white">
        <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.17L4 17.17V4h16v12z"/>
        <path d="M7 9h2v2H7zm4 0h2v2h-2zm4 0h2v2h-2z"/>
    </svg>
</div>

<div id="chat-window" class="chat-window" style="display: none;">
    <div class="chat-header">
        <div class="chat-header-content">
            <div class="chat-header-avatar">🌿</div>
            <div class="chat-header-title">
                <div class="chat-header-name">Farmcare AI</div>
                <div class="chat-header-status">Active now</div>
            </div>
        </div>
        <div class="chat-close" onclick="toggleChat()">×</div>
    </div>
    <div id="chat-messages" class="chat-messages">
        <div class="bot-bubble typing-effect">
            <div class="bot-avatar">
                <div class="voice-wave">
                    <span></span>
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
            <div class="message-content">
                👋 Hi there! How can I assist you today with plant care or disease questions?
            </div>
            <div class="time-stamp">Just now</div>
        </div>
    </div>
    <div class="chat-input-area">
        <div class="emoji-button" title="Add emoji" onclick="toggleEmojiPicker()">😊</div>
        <input type="text" id="chat-input" class="chat-input-field" placeholder="Type your message here...">
        <button id="chat-send" class="chat-send-button">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="white">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path>
            </svg>
        </button>
    </div>
</div>

<script>
// Create dynamic particles for background effect
function createParticles() {
    const container = document.getElementById('particles-container');
    const particleCount = 20; // Number of particles
    
    // Clear any existing particles
    container.innerHTML = '';
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.classList.add('particle');
        
        // Random position
        const posX = Math.random() * 100;
        const posY = Math.random() * 100;
        particle.style.left = `${posX}%`;
        particle.style.top = `${posY}%`;
        
        // Random size
        const size = Math.random() * 8 + 2;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        
        // Random opacity
        const opacity = Math.random() * 0.5 + 0.1;
        particle.style.opacity = opacity;
        
        // Random animation duration
        const duration = Math.random() * 15 + 8;
        particle.style.animationDuration = `${duration}s`;
        
        // Random animation delay
        const delay = Math.random() * 5;
        particle.style.animationDelay = `${delay}s`;
        
        // Random rotation
        const rotation = Math.random() * 360;
        particle.style.transform = `rotate(${rotation}deg)`;
        
        // Add subtle shadow for depth
        particle.style.boxShadow = '0 0 5px rgba(124, 58, 237, 0.3)';
        
        // Random color variation
        const hue = Math.random() * 40 + 220; // Blues to purples
        particle.style.backgroundColor = `hsla(${hue}, 70%, 60%, ${opacity})`;
        
        // Add some with blurred effect for depth
        if (Math.random() > 0.7) {
            particle.style.filter = 'blur(1px)';
        }
        
        container.appendChild(particle);
    }
}

// Create enhanced typing indicator with smooth animation
function showTypingIndicator() {
    const messagesContainer = document.getElementById('chat-messages');
    const typingIndicator = document.createElement('div');
    typingIndicator.id = 'typing-indicator';
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = '<span></span><span></span><span></span>';
    
    // Add with animation
    typingIndicator.style.opacity = '0';
    typingIndicator.style.transform = 'scale(0.9)';
    messagesContainer.appendChild(typingIndicator);
    
    setTimeout(() => {
        typingIndicator.style.opacity = '1';
        typingIndicator.style.transform = 'scale(1)';
    }, 10);
    
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return typingIndicator;
}

// Remove typing indicator with fade-out animation
function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.style.opacity = '0';
        typingIndicator.style.transform = 'scale(0.9)';
        
        setTimeout(() => {
            if (typingIndicator.parentNode) {
                typingIndicator.parentNode.removeChild(typingIndicator);
            }
        }, 300);
    }
}

// Advanced typing effect with natural pacing
function addTypingEffect(element, text, speed = 20) {
    // Store original text content if any
    const originalContent = element.innerHTML;
    const messageContent = element.querySelector('.message-content') || element;
    
    // Clear only the message part, keeping avatar if it exists
    if (messageContent !== element) {
        messageContent.textContent = '';
    } else {
        element.textContent = '';
    }
    
    let i = 0;
    let lastCharWasPunctuation = false;
    
    function typeNextChar() {
        if (i < text.length) {
            const char = text.charAt(i);
            
            // Add the character to the element
            messageContent.textContent += char;
            i++;
            
            // Variable speed based on punctuation
            let nextSpeed = speed;
            
            // Slow down after punctuation for more natural rhythm
            if (char === '.' || char === '!' || char === '?' || char === ',') {
                nextSpeed = char === ',' ? speed * 8 : speed * 15;
                lastCharWasPunctuation = true;
            } else if (lastCharWasPunctuation) {
                nextSpeed = speed * 3;
                lastCharWasPunctuation = false;
            } else {
                // Random slight variation in typing speed for natural effect
                nextSpeed = speed + (Math.random() * speed * 0.5);
            }
            
            // Continue typing after appropriate delay
            setTimeout(typeNextChar, nextSpeed);
            
            // Scroll as typing progresses
            const messagesContainer = document.getElementById('chat-messages');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }
    
    // Start the typing animation
    typeNextChar();
}

// Toggle emoji picker (placeholder function)
function toggleEmojiPicker() {
    // In a real implementation, this would toggle an emoji picker
    const input = document.getElementById('chat-input');
    const emojis = ['😊', '👋', '🌱', '🌿', '🌷', '🍀', '🌺', '🌞', '🌧️', '🧪'];
    const randomEmoji = emojis[Math.floor(Math.random() * emojis.length)];
    input.value += randomEmoji;
    input.focus();
}

// Get current time in format HH:MM AM/PM
function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
}

// Toggle chat window with enhanced animations
function toggleChat() {
    const chatWindow = document.getElementById('chat-window');
    const chatButton = document.getElementById('chat-icon');
    
    if (chatWindow.style.display === 'none') {
        // Show chat window with animation
        chatWindow.style.display = 'flex';
        chatWindow.style.opacity = '0';
        chatWindow.style.transform = 'translate(50%, 50%) scale(0.9)';
        
        setTimeout(() => {
            chatWindow.style.opacity = '1';
            chatWindow.style.transform = 'translate(50%, 50%) scale(1)';
        }, 50);
        
        // Add active state to button
        chatButton.classList.add('active');
    } else {
        // Hide chat window with animation
        chatWindow.style.opacity = '0';
        chatWindow.style.transform = 'translate(50%, 50%) scale(0.9)';
        
        setTimeout(() => {
            chatWindow.style.display = 'none';
        }, 300);
        
        // Remove active state from button
        chatButton.classList.remove('active');
    }
    
    // Also notify Streamlit of the change
    if (window.parent.window.streamlitSendBackMsg) {
        window.parent.window.streamlitSendBackMsg({
            type: 'streamlit:componentReady',
            value: { name: 'toggleChat', data: {} }
        });
    }
}

// Set up event listeners with enhanced interaction
document.addEventListener('DOMContentLoaded', function() {
    const chatSendButton = document.getElementById('chat-send');
    const chatInput = document.getElementById('chat-input');
    
    chatSendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
            e.preventDefault();
        }
    });
    
    // Focus input when chat window is opened
    const chatWindow = document.getElementById('chat-window');
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.attributeName === 'style' && 
                chatWindow.style.display !== 'none' && 
                chatWindow.style.opacity !== '0') {
                chatInput.focus();
            }
        });
    });
    
    observer.observe(chatWindow, { attributes: true });
    
    // Add subtle interaction effects
    chatInput.addEventListener('focus', function() {
        this.parentElement.style.boxShadow = '0 0 0 2px rgba(124, 58, 237, 0.2)';
    });
    
    chatInput.addEventListener('blur', function() {
        this.parentElement.style.boxShadow = 'none';
    });
});

// Enhanced send message function with modern UI and animations
function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    
    if (message) {
        // Add user message to chat with animation
        const messagesContainer = document.getElementById('chat-messages');
        const userBubble = document.createElement('div');
        userBubble.className = 'user-bubble';
        
        // Create message content and timestamp
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = message;
        
        const timeStamp = document.createElement('div');
        timeStamp.className = 'time-stamp';
        timeStamp.textContent = getCurrentTime();
        
        // Add content to bubble
        userBubble.appendChild(messageContent);
        userBubble.appendChild(timeStamp);
        
        // Set initial state for animation
        userBubble.style.opacity = '0';
        userBubble.style.transform = 'translateY(10px) scale(0.95)';
        
        // Add to container
        messagesContainer.appendChild(userBubble);
        
        // Trigger animation
        setTimeout(() => {
            userBubble.style.opacity = '1';
            userBubble.style.transform = 'translateY(0) scale(1)';
        }, 10);
        
        // Clear input
        input.value = '';
        input.focus();
        
        // Scroll to bottom with smooth animation
        smoothScrollToBottom(messagesContainer);
        
        // Show typing indicator with delay
        setTimeout(() => {
            showTypingIndicator();
        }, 500);
        
        // Send message to Streamlit
        if (window.parent.window.streamlitSendBackMsg) {
            window.parent.window.streamlitSendBackMsg({
                type: 'streamlit:componentReady',
                value: { name: 'chatMessage', data: message }
            });
        }
    }
}

// Smooth scroll to bottom function
function smoothScrollToBottom(element) {
    const targetPosition = element.scrollHeight;
    const startPosition = element.scrollTop;
    const distance = targetPosition - startPosition;
    const duration = 300;
    let startTime = null;
    
    function animation(currentTime) {
        if (startTime === null) startTime = currentTime;
        const timeElapsed = currentTime - startTime;
        const scrollY = easeInOutQuad(timeElapsed, startPosition, distance, duration);
        element.scrollTop = scrollY;
        if (timeElapsed < duration) requestAnimationFrame(animation);
    }
    
    // Easing function
    function easeInOutQuad(t, b, c, d) {
        t /= d/2;
        if (t < 1) return c/2*t*t + b;
        t--;
        return -c/2 * (t*(t-2) - 1) + b;
    }
    
    requestAnimationFrame(animation);
}

// Enhanced bot message function with avatar and voice wave animation
window.addBotMessage = function(message) {
    // Remove typing indicator with animation
    removeTypingIndicator();
    
    const messagesContainer = document.getElementById('chat-messages');
    const botBubble = document.createElement('div');
    botBubble.className = 'bot-bubble';
    
    // Create bot avatar with voice wave animation
    const botAvatar = document.createElement('div');
    botAvatar.className = 'bot-avatar';
    botAvatar.innerHTML = `
        <div class="voice-wave">
            <span></span>
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    
    // Create message content element
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    // Create timestamp
    const timeStamp = document.createElement('div');
    timeStamp.className = 'time-stamp';
    timeStamp.textContent = getCurrentTime();
    
    // Add elements to bubble
    botBubble.appendChild(botAvatar);
    botBubble.appendChild(messageContent);
    botBubble.appendChild(timeStamp);
    
    // Set initial state for animation
    botBubble.style.opacity = '0';
    botBubble.style.transform = 'translateY(10px) scale(0.95)';
    
    // Add to container
    messagesContainer.appendChild(botBubble);
    
    // Trigger appearance animation
    setTimeout(() => {
        botBubble.style.opacity = '1';
        botBubble.style.transform = 'translateY(0) scale(1)';
    }, 10);
    
    // Add typing effect after appearance animation
    setTimeout(() => {
        addTypingEffect(botBubble, message, 8);
        
        // Remove voice wave animation when typing is complete
        const typingDuration = message.length * 8 + 500;
        setTimeout(() => {
            const voiceWave = botAvatar.querySelector('.voice-wave');
            if (voiceWave) {
                voiceWave.style.opacity = '0';
                setTimeout(() => {
                    botAvatar.innerHTML = '🌿';
                }, 300);
            }
        }, typingDuration);
    }, 300);
    
    // Scroll to bottom
    smoothScrollToBottom(messagesContainer);
};

// Enhanced chat visibility update with smooth animations
window.updateChatVisibility = function(isVisible) {
    const chatWindow = document.getElementById('chat-window');
    const chatButton = document.getElementById('chat-icon');
    
    if (isVisible) {
        // Show chat with animation
        chatWindow.style.display = 'flex';
        chatWindow.style.opacity = '0';
        chatWindow.style.transform = 'translate(50%, 50%) scale(0.9)';
        
        setTimeout(() => {
            chatWindow.style.opacity = '1';
            chatWindow.style.transform = 'translate(50%, 50%) scale(1)';
            // Focus input
            document.getElementById('chat-input').focus();
        }, 50);
        
        chatButton.classList.add('active');
    } else {
        // Hide chat with animation
        chatWindow.style.opacity = '0';
        chatWindow.style.transform = 'translate(50%, 50%) scale(0.9)';
        
        setTimeout(() => {
            chatWindow.style.display = 'none';
        }, 300);
        
        chatButton.classList.remove('active');
    }
};

// Enhanced chat history update with staged animations
window.updateChatHistory = function(history) {
    const messagesContainer = document.getElementById('chat-messages');
    messagesContainer.innerHTML = '';
    
    if (history.length === 0) {
        // Create welcome message with avatar and typing effect
        const welcomeBubble = document.createElement('div');
        welcomeBubble.className = 'bot-bubble';
        
        // Create bot avatar with voice wave
        const botAvatar = document.createElement('div');
        botAvatar.className = 'bot-avatar';
        botAvatar.innerHTML = `
            <div class="voice-wave">
                <span></span>
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        
        // Create message content and timestamp
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const timeStamp = document.createElement('div');
        timeStamp.className = 'time-stamp';
        timeStamp.textContent = getCurrentTime();
        
        // Add elements to bubble
        welcomeBubble.appendChild(botAvatar);
        welcomeBubble.appendChild(messageContent);
        welcomeBubble.appendChild(timeStamp);
        
        messagesContainer.appendChild(welcomeBubble);
        
        // Add typing effect
        setTimeout(() => {
            const welcomeText = '👋 Hi there! How can I assist you today with plant care or disease questions?';
            addTypingEffect(welcomeBubble, welcomeText, 8);
            
            // Remove voice wave animation when typing is complete
            const typingDuration = welcomeText.length * 8 + 500;
            setTimeout(() => {
                const voiceWave = botAvatar.querySelector('.voice-wave');
                if (voiceWave) {
                    voiceWave.style.opacity = '0';
                    setTimeout(() => {
                        botAvatar.innerHTML = '🌿';
                    }, 300);
                }
            }, typingDuration);
        }, 500);
        
        return;
    }
    
    // Add messages with staged animation
    let delay = 0;
    const delayIncrement = 200;
    
    for (const msg of history) {
        setTimeout(() => {
            if (msg.role === 'user') {
                // Create user message
                const userBubble = document.createElement('div');
                userBubble.className = 'user-bubble';
                
                // Create message content and timestamp
                const messageContent = document.createElement('div');
                messageContent.className = 'message-content';
                messageContent.textContent = msg.content;
                
                const timeStamp = document.createElement('div');
                timeStamp.className = 'time-stamp';
                timeStamp.textContent = getCurrentTime();
                
                // Add content to bubble
                userBubble.appendChild(messageContent);
                userBubble.appendChild(timeStamp);
                
                // Set initial state for animation
                userBubble.style.opacity = '0';
                userBubble.style.transform = 'translateY(10px) scale(0.95)';
                
                // Add to container
                messagesContainer.appendChild(userBubble);
                
                // Trigger animation
                setTimeout(() => {
                    userBubble.style.opacity = '1';
                    userBubble.style.transform = 'translateY(0) scale(1)';
                }, 10);
                
            } else if (msg.role === 'assistant') {
                // Create bot message with avatar
                const botBubble = document.createElement('div');
                botBubble.className = 'bot-bubble';
                
                // Create bot avatar
                const botAvatar = document.createElement('div');
                botAvatar.className = 'bot-avatar';
                botAvatar.innerHTML = '🌿';
                
                // Create message content and timestamp
                const messageContent = document.createElement('div');
                messageContent.className = 'message-content';
                messageContent.textContent = msg.content;
                
                const timeStamp = document.createElement('div');
                timeStamp.className = 'time-stamp';
                timeStamp.textContent = getCurrentTime();
                
                // Add elements to bubble
                botBubble.appendChild(botAvatar);
                botBubble.appendChild(messageContent);
                botBubble.appendChild(timeStamp);
                
                // Set initial state for animation
                botBubble.style.opacity = '0';
                botBubble.style.transform = 'translateY(10px) scale(0.95)';
                
                // Add to container
                messagesContainer.appendChild(botBubble);
                
                // Trigger animation
                setTimeout(() => {
                    botBubble.style.opacity = '1';
                    botBubble.style.transform = 'translateY(0) scale(1)';
                }, 10);
            }
            
            // Scroll to the last message
            smoothScrollToBottom(messagesContainer);
        }, delay);
        
        delay += delayIncrement;
    }
};

// Initialize particles and other effects
document.addEventListener('DOMContentLoaded', function() {
    // Create particles
    createParticles();
    
    // Set up transitions
    const chatWindow = document.getElementById('chat-window');
    chatWindow.style.opacity = '0';
    chatWindow.style.transform = 'translate(50%, 50%) scale(0.9)';
    chatWindow.style.transition = 'opacity 0.4s cubic-bezier(0.34, 1.56, 0.64, 1), transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)';
    
    // Apply initial typing effect to welcome message with delay
    const welcomeMessage = document.querySelector('.bot-bubble.typing-effect');
    if (welcomeMessage) {
        const messageContent = welcomeMessage.querySelector('.message-content');
        const messageText = messageContent.textContent;
        messageContent.textContent = '';
        
        setTimeout(() => {
            addTypingEffect(welcomeMessage, messageText, 8);
            
            // Remove voice wave animation when typing is complete
            const typingDuration = messageText.length * 8 + 500;
            setTimeout(() => {
                const botAvatar = welcomeMessage.querySelector('.bot-avatar');
                const voiceWave = botAvatar.querySelector('.voice-wave');
                if (voiceWave) {
                    voiceWave.style.opacity = '0';
                    setTimeout(() => {
                        botAvatar.innerHTML = '🌿';
                    }, 300);
                }
            }, typingDuration);
        }, 800);
    }
    
    // Recreate particles every minute for dynamic effect
    setInterval(createParticles, 60000);
});
</script>
"""

# Render the chat UI
components.html(chat_html, height=0)

# Create a container for the chat interface
chat_container = st.container()

# Create simpler chat interaction components
# This will use Streamlit's native state management instead of complex bidirectional communication

# Note: We're keeping only one instance of the toggle_chat() function
# The first definition is used, this one is removed to avoid duplication

# Function to handle chat button clicks
if 'chat_toggle_button' not in st.session_state:
    st.session_state.chat_toggle_button = False

# Initialize the chat interface state
if 'chat_message' not in st.session_state:
    st.session_state.chat_message = ""

# Add CSS for enhanced chatbot components with animations and modern UI
st.markdown("""
<style>
/* Modern Ultra-Enhanced Chatbot styling with futuristic design */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* Keyframes for animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.9); }
    to { opacity: 1; transform: scale(1); }
}

@keyframes pulse {
    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.7); }
    70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(33, 150, 243, 0); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(33, 150, 243, 0); }
}

@keyframes typing {
    from { width: 0 }
    to { width: 100% }
}

@keyframes blink {
    50% { border-color: transparent }
}

@keyframes wave {
    0% { transform: translateY(0px); }
    25% { transform: translateY(-3px); }
    50% { transform: translateY(0px); }
    75% { transform: translateY(3px); }
    100% { transform: translateY(0px); }
}

@keyframes glow {
    0% { box-shadow: 0 0 5px rgba(33, 150, 243, 0.5); }
    50% { box-shadow: 0 0 20px rgba(33, 150, 243, 0.8); }
    100% { box-shadow: 0 0 5px rgba(33, 150, 243, 0.5); }
}

@keyframes float {
    0% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-10px) rotate(3deg); }
    100% { transform: translateY(0px) rotate(0deg); }
}

@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes borderGlow {
    0% { border-color: #4F46E5; }
    50% { border-color: #A78BFA; }
    100% { border-color: #4F46E5; }
}

@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    40% { transform: translateY(-15px); }
    60% { transform: translateY(-7px); }
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* Advanced particles background with dynamic effects */
.particles {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    overflow: hidden;
    pointer-events: none;
}

.particle {
    position: absolute;
    width: 6px;
    height: 6px;
    background-color: rgba(33, 150, 243, 0.15);
    border-radius: 50%;
    animation: float 10s infinite ease-in-out;
    opacity: 0.7;
    filter: blur(1px);
}

/* Gradient background overlay */
.bg-gradient {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(125deg, #a864fd33, #29cdff33, #78ff9233, #ff718d33);
    background-size: 400% 400%;
    animation: gradientFlow 15s ease infinite;
    z-index: -2;
    pointer-events: none;
    opacity: 0.5;
}

/* Main container */
.chatbot-container {
    position: fixed;
    bottom: 25px;
    right: 25px;
    z-index: 9999;
    font-family: 'Inter', sans-serif;
}

/* Floating button with modern design */
.chat-button {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    border: none;
    box-shadow: 0 5px 15px rgba(124, 58, 237, 0.4), 0 0 0 0 rgba(124, 58, 237, 0.8);
    cursor: pointer;
    display: flex;
    justify-content: center;
    align-items: center;
    transition: all 0.4s cubic-bezier(0.68, -0.55, 0.27, 1.55);
    animation: pulse 2.5s infinite;
    position: relative;
    overflow: hidden;
}

.chat-button:before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: rgba(255, 255, 255, 0.1);
    transform: rotate(45deg);
    transition: all 0.6s ease;
    opacity: 0;
}

.chat-button:hover {
    transform: scale(1.08);
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.5);
}

.chat-button:hover:before {
    opacity: 0.3;
    animation: spin 2s linear infinite;
}

.chat-button svg {
    width: 26px;
    height: 26px;
    fill: white;
    animation: wave 3s ease infinite;
    filter: drop-shadow(0 2px 3px rgba(0, 0, 0, 0.2));
}

/* Modern, centered chat window with glassmorphism */
.chat-window {
    position: fixed;
    bottom: 50%;
    right: 50%;
    transform: translate(50%, 50%) scale(1);
    width: 400px;
    height: 550px;
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.25);
    display: flex;
    flex-direction: column;
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    animation: scaleIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    font-family: 'Inter', sans-serif;
    transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

/* Modern futuristic header with animated gradient */
.chat-header {
    background: linear-gradient(-45deg, #4F46E5, #7C3AED, #A78BFA, #4F46E5);
    background-size: 300% 300%;
    animation: gradientFlow 6s ease infinite;
    color: white;
    padding: 20px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 500;
    border-bottom: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
    position: relative;
}

.chat-header:after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(to right, transparent, rgba(255, 255, 255, 0.5), transparent);
}

.chat-header-content {
    display: flex;
    align-items: center;
    gap: 12px;
}

.chat-header-avatar {
    width: 42px;
    height: 42px;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.15);
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.2);
}

.chat-header-avatar:after {
    content: "";
    position: absolute;
    top: -10px;
    left: -10px;
    width: calc(100% + 20px);
    height: calc(100% + 20px);
    background: radial-gradient(circle, rgba(255, 255, 255, 0.2) 0%, transparent 70%);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.chat-header-avatar:hover:after {
    opacity: 1;
}

.chat-header-title {
    display: flex;
    flex-direction: column;
}

.chat-header-name {
    font-weight: 600;
    font-size: 17px;
    letter-spacing: 0.2px;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.chat-header-status {
    font-size: 13px;
    opacity: 0.85;
    display: flex;
    align-items: center;
    gap: 5px;
}

.chat-header-status:before {
    content: "";
    display: block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #4ADE80;
    box-shadow: 0 0 10px #4ADE80;
    animation: pulse 2s infinite;
}

.chat-close {
    cursor: pointer;
    color: white;
    width: 36px;
    height: 36px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.15);
    display: flex;
    justify-content: center;
    align-items: center;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    font-size: 20px;
    backdrop-filter: blur(5px);
}

.chat-close:hover {
    background: rgba(255, 255, 255, 0.25);
    transform: scale(1.1) rotate(90deg);
    box-shadow: 0 0 15px rgba(255, 255, 255, 0.3);
}

/* Advanced messages container with elegant scrolling */
.chat-messages {
    flex-grow: 1;
    padding: 24px;
    overflow-y: auto;
    background: rgba(248, 250, 252, 0.3);
    display: flex;
    flex-direction: column;
    gap: 18px;
    scrollbar-width: thin;
    scrollbar-color: rgba(124, 58, 237, 0.5) transparent;
    background-image: 
        radial-gradient(circle at 25% 25%, rgba(124, 58, 237, 0.03) 0%, transparent 50%),
        radial-gradient(circle at 75% 75%, rgba(79, 70, 229, 0.03) 0%, transparent 50%);
    position: relative;
}

.chat-messages::-webkit-scrollbar {
    width: 5px;
}

.chat-messages::-webkit-scrollbar-track {
    background: transparent;
}

.chat-messages::-webkit-scrollbar-thumb {
    background-color: rgba(124, 58, 237, 0.3);
    border-radius: 20px;
    transition: background-color 0.3s ease;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
    background-color: rgba(124, 58, 237, 0.5);
}

/* Futuristic input area with dynamic effects */
.chat-input-area {
    padding: 18px 22px;
    border-top: 1px solid rgba(124, 58, 237, 0.1);
    display: flex;
    background: rgba(255, 255, 255, 0.7);
    align-items: center;
    gap: 12px;
    position: relative;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

.chat-input-area:before {
    content: "";
    position: absolute;
    top: -2px;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(to right, rgba(124, 58, 237, 0), rgba(124, 58, 237, 0.5), rgba(124, 58, 237, 0));
}

.chat-input-field {
    flex-grow: 1;
    border: 2px solid rgba(124, 58, 237, 0.2);
    border-radius: 16px;
    padding: 14px 20px;
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03), inset 0 2px 4px rgba(0, 0, 0, 0.02);
    transition: all 0.3s ease;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: #374151;
    letter-spacing: 0.2px;
    position: relative;
    z-index: 1;
}

.chat-input-field:focus {
    outline: none;
    border-color: #7C3AED;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.25), inset 0 2px 4px rgba(0, 0, 0, 0.01);
}

.chat-input-field::placeholder {
    color: #9CA3AF;
    opacity: 0.8;
}

.chat-send-button {
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    color: white;
    border: none;
    border-radius: 14px;
    width: 48px;
    height: 48px;
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    box-shadow: 0 4px 10px rgba(124, 58, 237, 0.25);
    position: relative;
    overflow: hidden;
}

.chat-send-button:before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(transparent, rgba(255, 255, 255, 0.2), transparent);
    transform: rotate(45deg);
    transition: transform 0.5s ease;
    transform: translateX(-100%);
}

.chat-send-button:hover {
    transform: scale(1.08) translateY(-2px);
    box-shadow: 0 6px 15px rgba(124, 58, 237, 0.35);
}

.chat-send-button:hover:before {
    transform: translateX(100%);
    transition: transform 0.8s ease;
}

.chat-send-button:active {
    transform: scale(0.95);
}

.chat-send-button svg {
    width: 20px;
    height: 20px;
    fill: white;
    filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.2));
}

/* Advanced modern chat bubbles with animations */
.user-bubble {
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    color: white;
    padding: 14px 20px;
    border-radius: 16px 16px 0 16px;
    margin: 5px 0;
    max-width: 85%;
    align-self: flex-end;
    margin-left: auto;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.25);
    animation: scaleIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    word-wrap: break-word;
    position: relative;
    z-index: 1;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    font-weight: 400;
    letter-spacing: 0.2px;
    line-height: 1.5;
    transform-origin: bottom right;
}

.user-bubble:before {
    content: "";
    position: absolute;
    bottom: -5px;
    right: 15px;
    width: 10px;
    height: 10px;
    background: linear-gradient(135deg, transparent 50%, #7C3AED 50%);
    transform: rotate(45deg);
    z-index: -1;
}

.user-bubble:after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 16px 16px 0 16px;
    padding: 2px;
    background: linear-gradient(135deg, rgba(255,255,255,0.2), rgba(255,255,255,0.05));
    mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    mask-composite: exclude;
    -webkit-mask-composite: destination-out;
    pointer-events: none;
}

.bot-bubble {
    background: rgba(255, 255, 255, 0.85);
    padding: 14px 20px;
    border-radius: 16px 16px 16px 0;
    margin: 5px 0;
    max-width: 85%;
    align-self: flex-start;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    animation: scaleIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
    word-wrap: break-word;
    border-left: 3px solid #7C3AED;
    position: relative;
    z-index: 1;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    color: #1F2937;
    font-weight: 400;
    letter-spacing: 0.2px;
    line-height: 1.5;
    transform-origin: top left;
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.bot-bubble:before {
    content: "";
    position: absolute;
    bottom: -5px;
    left: 15px;
    width: 10px;
    height: 10px;
    background: linear-gradient(-45deg, transparent 50%, rgba(255, 255, 255, 0.85) 50%);
    transform: rotate(45deg);
    z-index: -1;
}

.bot-bubble:hover {
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
    transform: translateY(-1px);
    transition: all 0.3s ease;
}

/* Advanced typing indicator with wave animation */
.typing-indicator {
    display: flex;
    align-items: center;
    padding: 12px 18px;
    background: rgba(255, 255, 255, 0.7);
    border-radius: 16px 16px 16px 0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    width: fit-content;
    margin-top: 5px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-left: 3px solid #7C3AED;
    position: relative;
    animation: scaleIn 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
}

.typing-indicator:before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, rgba(124, 58, 237, 0), rgba(124, 58, 237, 0.05));
    border-radius: 16px 16px 16px 0;
    z-index: -1;
}

.typing-indicator span {
    height: 9px;
    width: 9px;
    margin: 0 3px;
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 5px rgba(124, 58, 237, 0.3);
}

.typing-indicator span:nth-child(1) {
    animation: bounce 1.2s ease-in-out infinite;
}

.typing-indicator span:nth-child(2) {
    animation: bounce 1.2s ease-in-out 0.15s infinite;
}

.typing-indicator span:nth-child(3) {
    animation: bounce 1.2s ease-in-out 0.3s infinite;
}

/* Bot avatar animation for voice wave visualization */
.bot-avatar {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    margin-right: 8px;
    position: relative;
    overflow: hidden;
}

.voice-wave {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    width: 100%;
}

.voice-wave span {
    display: inline-block;
    width: 3px;
    height: 10px;
    margin: 0 1px;
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
}

.voice-wave span:nth-child(1) {
    animation: wave 0.6s ease-in-out infinite;
    height: 8px;
}
.voice-wave span:nth-child(2) {
    animation: wave 0.7s ease-in-out 0.1s infinite;
    height: 16px;
}
.voice-wave span:nth-child(3) {
    animation: wave 0.8s ease-in-out 0.2s infinite;
    height: 10px;
}
.voice-wave span:nth-child(4) {
    animation: wave 0.7s ease-in-out 0.3s infinite;
    height: 14px;
}

.chat-popup {
    visibility: hidden;
}

/* To make the Streamlit sidebar look like a floating chat window */
.streamlit-expanderHeader, .stExpander {
    display: none !important;
}

/* Additional UI enhancements */
.emoji-button {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    border-radius: 12px;
    background: rgba(124, 58, 237, 0.1);
    cursor: pointer;
    transition: all 0.3s ease;
    border: 1px solid rgba(124, 58, 237, 0.2);
    color: #7C3AED;
    font-size: 18px;
}

.emoji-button:hover {
    background: rgba(124, 58, 237, 0.15);
    transform: scale(1.05);
}

.time-stamp {
    font-size: 10px;
    opacity: 0.7;
    margin-top: 2px;
    align-self: flex-end;
    font-weight: 400;
}

/* Media queries for responsive design */
@media screen and (max-width: 768px) {
    .chat-window {
        width: 90%;
        height: 70vh;
        max-height: none;
        bottom: 50%;
        right: 50%;
        border-radius: 18px;
    }
    
    .chat-button {
        bottom: 15px;
        right: 15px;
    }
    
    .chat-input-area {
        padding: 15px;
    }
    
    .chat-input-field {
        padding: 12px 16px;
    }
    
    .user-bubble, .bot-bubble {
        max-width: 90%;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .chat-window {
        background: rgba(30, 30, 46, 0.85);
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    .chat-input-field {
        background: rgba(30, 30, 46, 0.9);
        color: #e2e8f0;
        border-color: rgba(124, 58, 237, 0.3);
    }
    
    .chat-input-field::placeholder {
        color: #94a3b8;
    }
    
    .bot-bubble {
        background: rgba(40, 40, 56, 0.85);
        color: #e2e8f0;
    }
    
    .typing-indicator {
        background: rgba(40, 40, 56, 0.85);
    }
    
    .chat-messages {
        background: rgba(30, 30, 40, 0.3);
    }
}
    }
}
</style>
""", unsafe_allow_html=True)

# Create the chat button component
chat_button_html = """
<div class="chatbot-container" style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
    <button onclick="toggleChatWindow()" class="chat-button" 
           style="width: 60px; height: 60px; border-radius: 30px; background-color: #4CAF50; 
                  border: none; box-shadow: 0 2px 10px rgba(0,0,0,0.2); cursor: pointer;
                  display: flex; justify-content: center; align-items: center; transition: transform 0.2s;">
        <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="white">
            <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
            <path d="M7 9h2v2H7zm4 0h2v2h-2zm4 0h2v2h-2z"/>
        </svg>
    </button>
</div>

<style>
    .chat-button:hover {
        transform: scale(1.05);
    }
    .chat-button:active {
        transform: scale(0.95);
    }
</style>

<script>
    function toggleChatWindow() {
        // Force reload with chat=true parameter
        const chatParam = 'chat=true';
        const currentUrl = window.location.href;
        const hasParams = currentUrl.includes('?');
        const hasChat = currentUrl.includes(chatParam);
        
        if (hasChat) {
            // If chat is already enabled, just refresh
            window.location.reload();
        } else if (hasParams) {
            // Has other params, append chat param
            window.location.href = currentUrl + '&' + chatParam;
        } else {
            // No params, add chat param
            window.location.href = currentUrl + '?' + chatParam;
        }
    }
</script>
"""

# Create a simple floating chat button that works reliably
chat_button_html = """
<div style="position: fixed; bottom: 20px; right: 20px; z-index: 9999;">
    <button onclick="toggleChat()" style="width: 60px; height: 60px; border-radius: 30px; background-color: #4CAF50; border: none; 
            cursor: pointer; box-shadow: 0 4px 8px rgba(0,0,0,0.2); display: flex; justify-content: center; align-items: center;">
        <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="white">
            <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
            <path d="M7 9h2v2H7zm4 0h2v2h-2zm4 0h2v2h-2z"/>
        </svg>
    </button>
</div>

<script>
    function toggleChat() {
        // This will simply redirect to the same page with a chat parameter to trigger the chat display
        window.location.href = window.location.pathname + "?show_chat=true";
    }
</script>
"""

# Display the chat button using a custom component
if not st.session_state.chat_visible:
    # Use a more compact chat button
    if st.button("💬 Chat", key="chat_button", on_click=toggle_chat,
                use_container_width=False, 
                help="Get help from Farmcare AI assistant",
                type="primary"):
        pass
    
    # Add CSS to position the button at bottom right
    st.markdown("""
        <style>
        /* Position the Chat button at the bottom right */
        [data-testid="baseButton-primary"]:has(div:contains("💬 Chat")) {
            position: fixed !important;
            right: 20px !important;
            bottom: 20px !important;
            width: auto !important;
            height: 60px !important;
            border-radius: 30px !important;
            padding: 0px 20px !important;
            box-shadow: 0px 3px 10px rgba(0, 0, 0, 0.3) !important;
            z-index: 999 !important;
            font-size: 16px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    
    # Check if a normal button is clicked as a fallback
    if st.button("💬 Chat", key="chat_button_fallback"):
        toggle_chat()

# Add custom CSS for the chat container
st.markdown("""
<style>
.chat-container {
    position: fixed !important;
    bottom: 80px !important;
    right: 20px !important;
    width: 350px !important;
    height: 500px !important;
    background-color: white !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2) !important;
    z-index: 9999 !important;
    overflow: hidden !important;
    border: 1px solid #ddd !important;
    display: flex !important;
    flex-direction: column !important;
}

.chat-header {
    background-color: #4CAF50;
    color: white;
    padding: 10px 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.chat-messages {
    flex-grow: 1;
    overflow-y: auto;
    padding: 10px;
    background-color: #f5f5f5;
}

.chat-input {
    padding: 10px;
    border-top: 1px solid #ddd;
    background-color: white;
}

.user-message {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 10px;
}

.bot-message {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 10px;
}

.message-bubble {
    padding: 8px 12px;
    border-radius: 15px;
    max-width: 80%;
    word-wrap: break-word;
}

.user-bubble {
    background-color: #4CAF50;
    color: white;
    border-radius: 15px 15px 0 15px;
}

.bot-bubble {
    background-color: white;
    color: black;
    border-radius: 15px 15px 15px 0;
    border: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

# Display chat interface when active
if st.session_state.chat_visible:
    # Add CSS for chat container with modern glassmorphism and animations
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    /* Animation keyframes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(124, 58, 237, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(124, 58, 237, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(124, 58, 237, 0); }
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes typing {
        0% { width: 0; }
        100% { width: 100%; }
    }
    
    .chat-overlay {
        position: fixed;
        right: 20px;
        bottom: 80px;
        width: 350px;
        height: 480px;
        background-color: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
        z-index: 9999;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        animation: fadeIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
        transform-origin: bottom right;
    }
    
    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 18px;
        background: linear-gradient(135deg, #4F46E5, #7C3AED, #A78BFA);
        background-size: 300% 300%;
        animation: gradientFlow 6s ease infinite;
        color: white;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        position: relative;
    }
    
    .chat-header::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(to right, transparent, rgba(255, 255, 255, 0.5), transparent);
    }
    
    .close-button {
        cursor: pointer;
        font-size: 20px;
        background: rgba(255, 255, 255, 0.15);
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .close-button:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: scale(1.1) rotate(90deg);
    }
    
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 16px;
        background: rgba(248, 250, 252, 0.4);
        background-image: 
            radial-gradient(circle at 25% 25%, rgba(124, 58, 237, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 75% 75%, rgba(79, 70, 229, 0.03) 0%, transparent 50%);
        scrollbar-width: thin;
        scrollbar-color: rgba(124, 58, 237, 0.3) transparent;
    }
    
    .chat-messages::-webkit-scrollbar {
        width: 5px;
    }
    
    .chat-messages::-webkit-scrollbar-track {
        background: transparent;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background-color: rgba(124, 58, 237, 0.3);
        border-radius: 20px;
        transition: background-color 0.3s ease;
    }
    
    .chat-messages::-webkit-scrollbar-thumb:hover {
        background-color: rgba(124, 58, 237, 0.5);
    }
    
    /* Make all chat elements have animations */
    .stButton, .stTextInput, .css-15zagb {
        animation: fadeIn 0.5s ease;
    }
    
    .stMarkdown p {
        margin-bottom: 8px;
    }
    
    /* Add a subtle glow effect to buttons on hover */
    .stButton > button:hover {
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.4);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a more compact header
    chat_col1, chat_col2 = st.columns([5, 1])
    
    with chat_col2:
        # Add small close button
        if st.button("❌", on_click=toggle_chat, key="close_chat_btn", help="Close chat"):
            pass
    
    with chat_col1:
        st.markdown("💬 **Farmcare AI**")
    
    # Create a container for the chat content
    chat_container = st.container()
    
    # Add a minimal header
    chat_container.markdown("""
    <div style="background-color: #f9f9f9; border-bottom: 1px solid #eee; padding: 8px 10px; margin-bottom: 5px;">
    <h5 style="margin: 0; font-size: 15px;">Chat with Farmcare AI</h5>
    </div>
    """, unsafe_allow_html=True)
    
    # Debug section removed
    
    # Display chat history in a scrollable container
    chat_history_container = chat_container.container()
    with chat_history_container:
        # Add some scrollable area with fixed height
        st.markdown("""
        <div style="max-height: 280px; overflow-y: auto; padding: 8px; background-color: #fafafa; border-radius: 8px; margin-bottom: 5px; font-size: 14px;">
        """, unsafe_allow_html=True)
        
        if not st.session_state.chat_history:
            st.info("No messages yet. Ask about plant diseases or gardening tips!")
        
        # Display messages with improved compact styling
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"""<div style="display: flex; justify-content: flex-end; margin-bottom: 8px;">
                        <div style="background-color: #4CAF50; color: white; padding: 8px 12px; border-radius: 16px 16px 0px 16px; max-width: 80%; box-shadow: 0 1px 1px rgba(0,0,0,0.1); font-size: 14px;">
                            {msg["content"]}
                        </div>
                    </div>""", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div style="display: flex; justify-content: flex-start; margin-bottom: 8px;">
                        <div style="background-color: #F0F0F0; padding: 8px 12px; border-radius: 16px 16px 16px 0px; max-width: 80%; box-shadow: 0 1px 1px rgba(0,0,0,0.1); font-size: 14px;">
                            {msg["content"]}
                        </div>
                    </div>""",
                    unsafe_allow_html=True
                )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Add chat input area with a compact form
    with chat_container:
        with st.form(key="chat_form", clear_on_submit=True, border=False):
            # More compact input area
            st.markdown("<div style='font-size: 13px; margin-bottom: 3px;'>Your message:</div>", unsafe_allow_html=True)
            user_input = st.text_input("", placeholder="Type your question about plants...", label_visibility="collapsed")
            cols = st.columns([0.8, 0.2])
            with cols[1]:
                submit_button = st.form_submit_button("Send 📤", use_container_width=True, type="primary")
            
            if submit_button and user_input:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Show a loading message while generating the response
                with st.spinner("AI is thinking..."):
                    # Generate AI response
                    ai_response = generate_gemini_response(user_input)
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                
                # Rerun to update UI with new messages
                st.rerun()
                
    # End the chat container div
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add dummy HTML to skip the original implementation
    chat_window_html = f"""
    <!-- Dummy HTML -->
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {{
                box-sizing: border-box;
            }}
            
            body {{
                margin: 0;
                padding: 0;
                font-family: Arial, Helvetica, sans-serif;
                overflow: hidden;
                background: transparent;
            }}
            
            #chat-container {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                display: flex;
                flex-direction: column;
                background-color: white;
                border-radius: 12px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            
            .chat-header {{
                background-color: #4CAF50;
                color: white;
                padding: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                height: 60px;
            }}
            
            .chat-messages {{
                flex-grow: 1;
                padding: 15px;
                overflow-y: auto;
                background-color: #f5f5f5;
                display: flex;
                flex-direction: column;
                gap: 10px;
                height: calc(100% - 120px);
            }}
            
            .chat-input-area {{
                display: flex;
                padding: 10px;
                border-top: 1px solid #e0e0e0;
                background-color: white;
                height: 60px;
            }}
            
            .chat-input-field {{
                flex-grow: 1;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 20px;
                margin-right: 8px;
                outline: none;
            }}
            
            .chat-send-button {{
                background-color: #4CAF50;
                border: none;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
            }}
            
            .user-bubble {{
                align-self: flex-end;
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                border-radius: 18px;
                max-width: 80%;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                word-wrap: break-word;
            }}
            
            .bot-bubble {{
                align-self: flex-start;
                background-color: white;
                color: #333;
                padding: 12px;
                border-radius: 18px;
                max-width: 80%;
                box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                word-wrap: break-word;
            }}
        </style>
    </head>
    <body>
    <div id="chat-container">
        <div class="chat-header">
            <div>
                <img src="https://i.ibb.co/0GG6Gx5/plant-icon.png" width="24" height="24" style="margin-right: 8px; vertical-align: middle;">
                Farmcare AI
            </div>
            <div class="chat-close" onclick="closeChat()" style="cursor: pointer; font-size: 24px; font-weight: bold;">×</div>
        </div>
        <div class="chat-messages" id="chat-messages" style="flex-grow: 1; overflow-y: auto; padding: 15px; display: flex; flex-direction: column; gap: 10px; background-color: #f5f5f5;">
            <div class="bot-bubble" style="align-self: flex-start; background-color: white; color: #333; padding: 12px; border-radius: 18px; max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                👋 Hi there! How can we help you today?
            </div>
            {"".join([
                f'<div class="{"user-bubble" if msg["role"] == "user" else "bot-bubble"}" style="align-self: {("flex-end" if msg["role"] == "user" else "flex-start")}; background-color: {("#4CAF50" if msg["role"] == "user" else "white")}; color: {("white" if msg["role"] == "user" else "#333")}; padding: 12px; border-radius: 18px; max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">{msg["content"]}</div>'
                for msg in st.session_state.chat_history
            ])}
        </div>
        <form id="chat-form" onsubmit="sendMessage(); return false;" style="margin-bottom: 0;">
            <div class="chat-input-area" style="display: flex; padding: 10px; border-top: 1px solid #e0e0e0; background-color: white;">
                <input type="text" id="user-input" class="chat-input-field" placeholder="Type here and press enter..." style="flex-grow: 1; padding: 10px; border: 1px solid #ddd; border-radius: 20px; margin-right: 8px; outline: none;">
                <button type="submit" class="chat-send-button" style="background-color: #4CAF50; border: none; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; cursor: pointer;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="white">
                        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path>
                    </svg>
                </button>
            </div>
        </form>
    </div>
    
    <script>
        // Function to close the chat window
        function closeChat() {{
            try {{
                // Try both approaches for maximum compatibility
                window.parent.location.href = window.parent.location.pathname;
                window.top.location.href = window.top.location.pathname;
            }} catch(e) {{
                // Fallback if above methods fail
                window.location.href = window.location.pathname;
            }}
        
        // Function to send message to Streamlit
        function sendMessage() {{
            const input = document.getElementById('user-input');
            const message = input.value.trim();
            
            if (message) {{
                // Add user message to chat UI for immediate feedback
                const messagesContainer = document.getElementById('chat-messages');
                messagesContainer.innerHTML += `<div class="user-bubble" style="align-self: flex-end; background-color: #4CAF50; color: white; padding: 12px; border-radius: 18px; max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">${{message}}</div>`;
                
                // Show typing indicator
                messagesContainer.innerHTML += `<div id="typing-indicator" class="bot-bubble" style="align-self: flex-start; background-color: white; color: #333; padding: 12px; border-radius: 18px; max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">Typing<span class="dot-typing">.</span><span class="dot-typing">.</span><span class="dot-typing">.</span></div>`;
                
                // Add CSS animation for the typing indicator
                const styleEl = document.createElement('style');
                styleEl.innerHTML = `
                    @keyframes dotTyping {{
                      0% {{ opacity: 0; }}
                      50% {{ opacity: 1; }}
                      100% {{ opacity: 0; }}
                    }}
                    .dot-typing:nth-child(1) {{ animation: dotTyping 1s infinite 0s; }}
                    .dot-typing:nth-child(2) {{ animation: dotTyping 1s infinite 0.25s; }}
                    .dot-typing:nth-child(3) {{ animation: dotTyping 1s infinite 0.5s; }}
                `;
                document.head.appendChild(styleEl);
                
                // Clear input
                input.value = '';
                
                // Scroll to bottom
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                
                // Redirect with the message as a parameter
                const searchParams = new URLSearchParams(window.location.search);
                searchParams.set('chat', 'true');
                searchParams.set('message', message);
                
                // Short delay to show the typing indicator before redirecting
                setTimeout(function() {{
                    window.location.href = window.location.pathname + '?' + searchParams.toString();
                }}, 300);
            }}
            
            // Prevent form submission
            return false;
        }}
        
        // Auto-scroll to the bottom of messages
        document.addEventListener('DOMContentLoaded', function() {{
            const messagesContainer = document.getElementById('chat-messages');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }});
        
        // Execute scroll on page load
        window.onload = function() {{
            const messagesContainer = document.getElementById('chat-messages');
            if (messagesContainer) {{
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }}
        }};
    </script>
    </body>
    </html>
    """
    
    # Display the chat window with fixed positioning to ensure it's visible
    st.markdown("""
        <style>
        /* Apply to all iframes but target specifically the chat window */
        .stComponent iframe {
            position: fixed !important;
            bottom: 80px !important;
            right: 20px !important;
            height: 500px !important;
            width: 350px !important;
            border: none !important;
            border-radius: 12px !important;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important;
            z-index: 999999 !important;
            background-color: white !important;
            overflow: hidden !important;
        }
        
        /* Make sure chat elements stay visible */
        .chat-container, 
        .chat-header, 
        .message-bubble, 
        .user-bubble, 
        .bot-bubble {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
        
        /* Chat form and inputs need to be visible */
        form[key="chat_form"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    

