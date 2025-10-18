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
import os
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
import json
import pathlib
import base64

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

# Configuration for the application

# Get API keys from environment variables
# Chatbot functionality removed

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
        'intensity_analysis': '🔍 Attention Intensity Analysis',
        'intensity_description': 'This analysis shows how intensely the AI model is focusing on different parts of the image.',
        'intensity_stats': '📊 Intensity Statistics',
        'attention_distribution': '🎯 Attention Distribution',
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
# API Configuration
# -----------------------------
# Gemini API configuration removed

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
    model_path = "plant_disease_model_final.pth"
    # Check if the model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please make sure the model file is in the same directory as the script.")
        st.info("Looking for model files in the current directory...")
        
        # List .pth files in the current directory
        pth_files = [f for f in os.listdir(".") if f.endswith(".pth")]
        if pth_files:
            st.info(f"Found these model files: {', '.join(pth_files)}")
            # Try to use the first available .pth file
            model_path = pth_files[0]
            st.info(f"Attempting to use '{model_path}' instead")
        else:
            st.error("No .pth model files found in the current directory.")
            # Return a dummy model to avoid crashing
            dummy_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASS_NAMES))
            dummy_model.eval()
            return dummy_model
    
    try:
        # First load the state_dict to check its shape
        state_dict = torch.load(model_path, map_location="cpu")
        
        # Check the shape of the classifier layer to determine number of classes in the saved model
        if 'head.weight' in state_dict:
            num_classes_in_checkpoint = state_dict['head.weight'].shape[0]
            print(f"Detected {num_classes_in_checkpoint} classes in the model checkpoint")
        else:
            # If we can't determine the number of classes, use the length of CLASS_NAMES
            num_classes_in_checkpoint = len(CLASS_NAMES)
        
        # Create the model with the correct number of classes
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes_in_checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Return a dummy model to avoid crashing
        dummy_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASS_NAMES))
        dummy_model.eval()
        return dummy_model

# Load the model once when the app starts
model = load_model()

# Create a mapping from the model's classes to our CLASS_NAMES if needed
# This is a placeholder - you'll need to update this with the actual mapping
# between the 10 classes in your saved model and the 38 classes in CLASS_NAMES

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
    """Generate GradCAM heatmap for model interpretation with intensity analysis"""
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
                
                # Calculate intensity statistics
                intensity_stats = {
                    'max_intensity': float(np.max(grayscale_cam)),
                    'min_intensity': float(np.min(grayscale_cam)),
                    'mean_intensity': float(np.mean(grayscale_cam)),
                    'std_intensity': float(np.std(grayscale_cam)),
                    'high_attention_pixels': float(np.sum(grayscale_cam > 0.5) / grayscale_cam.size * 100),
                    'medium_attention_pixels': float(np.sum((grayscale_cam > 0.3) & (grayscale_cam <= 0.5)) / grayscale_cam.size * 100),
                    'low_attention_pixels': float(np.sum(grayscale_cam <= 0.3) / grayscale_cam.size * 100)
                }
                
                return cam_image, grayscale_cam, intensity_stats
                
            except Exception as layer_error:
                continue
        
        st.info("🔄 Using alternative gradient-based visualization...")
        return generate_alternative_heatmap(model, image_tensor, target_class_idx, image_rgb)
        
    except Exception as e:
        st.warning(f"Heatmap generation failed: {str(e)}")
        return generate_alternative_heatmap(model, image_tensor, target_class_idx, image_rgb)

def generate_alternative_heatmap(model, image_tensor, target_class_idx, image_rgb):
    """Generate alternative heatmap using gradient-based method with intensity analysis"""
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
        
        # Calculate intensity statistics for alternative method
        intensity_stats = {
            'max_intensity': float(np.max(heatmap_resized)),
            'min_intensity': float(np.min(heatmap_resized)),
            'mean_intensity': float(np.mean(heatmap_resized)),
            'std_intensity': float(np.std(heatmap_resized)),
            'high_attention_pixels': float(np.sum(heatmap_resized > 0.5) / heatmap_resized.size * 100),
            'medium_attention_pixels': float(np.sum((heatmap_resized > 0.3) & (heatmap_resized <= 0.5)) / heatmap_resized.size * 100),
            'low_attention_pixels': float(np.sum(heatmap_resized <= 0.3) / heatmap_resized.size * 100)
        }
        
        return cam_image, heatmap_resized, intensity_stats
        
    except Exception as e:
        st.error(f"Alternative heatmap generation failed: {str(e)}")
        return None, None, None

def create_intensity_analysis_chart(grayscale_cam, intensity_stats):
    """Create intensity analysis visualization for GradCAM"""
    fig = go.Figure()
    
    # Flatten the heatmap for histogram
    flat_intensities = grayscale_cam.flatten()
    
    # Create histogram
    fig.add_trace(go.Histogram(
        x=flat_intensities,
        nbinsx=50,
        name='Intensity Distribution',
        marker_color='rgba(55, 83, 109, 0.7)',
        marker_line=dict(color='rgba(55, 83, 109, 1.0)', width=1)
    ))
    
    # Add vertical lines for statistics
    fig.add_vline(x=intensity_stats['mean_intensity'], line_dash="dash", 
                  line_color="red", annotation_text=f"Mean: {intensity_stats['mean_intensity']:.3f}")
    fig.add_vline(x=intensity_stats['max_intensity'], line_dash="dot", 
                  line_color="green", annotation_text=f"Max: {intensity_stats['max_intensity']:.3f}")
    
    fig.update_layout(
        title="GradCAM Intensity Distribution",
        xaxis_title="Attention Intensity",
        yaxis_title="Pixel Count",
        showlegend=False,
        height=300,
        margin=dict(t=50, b=40, l=40, r=40)
    )
    
    return fig

def create_attention_pie_chart(intensity_stats):
    """Create pie chart showing attention level distribution"""
    labels = ['High Attention (>50%)', 'Medium Attention (30-50%)', 'Low Attention (<30%)']
    values = [
        intensity_stats['high_attention_pixels'],
        intensity_stats['medium_attention_pixels'],
        intensity_stats['low_attention_pixels']
    ]
    colors = ['#FF6B6B', '#FFE66D', '#95E1D3']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        hovertemplate='<b>%{label}</b><br>Percentage: %{value:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title="Model Attention Distribution",
        height=300,
        margin=dict(t=50, b=40, l=40, r=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def create_confidence_radar_chart(probs, class_names, top_k=8):
    """Create radar chart for top predictions"""
    # Make sure we're working with the latest data
    probs_detached = probs.detach().clone()
    
    # Get top predictions
    top_indices = torch.topk(probs_detached, min(top_k, len(class_names))).indices
    top_probs = probs_detached[top_indices].numpy() * 100
    top_classes = [class_names[i] for i in top_indices]
    
    # Clean up class names for display
    clean_names = []
    for name in top_classes:
        if '___' in name:
            plant, disease = name.split('___', 1)
            clean_name = f"{plant.replace('_', ' ')}\n{disease.replace('_', ' ')}"
        else:
            clean_name = name.replace('_', ' ')
        clean_names.append(clean_name)
    
    # Create the radar chart
    fig = go.Figure()
    
    # Use different colors based on confidence levels
    colors = []
    for prob in top_probs:
        if prob > 50:
            colors.append('rgb(220, 20, 60)')  # High confidence - red
        elif prob > 20:
            colors.append('rgb(255, 165, 0)')  # Medium confidence - orange
        else:
            colors.append('rgb(34, 139, 34)')  # Low confidence - green
    
    fig.add_trace(go.Scatterpolar(
        r=top_probs,
        theta=clean_names,
        fill='toself',
        name='Confidence %',
        line=dict(color='rgb(34, 139, 34)', width=2),
        fillcolor='rgba(34, 139, 34, 0.3)',
        marker=dict(
            size=8,
            color=colors,
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>%{theta}</b><br>Confidence: %{r:.2f}%<extra></extra>'
    ))
    
    # Calculate a dynamic range for better visualization
    max_prob = max(top_probs) if len(top_probs) > 0 else 100
    range_max = max(max_prob + 10, 50)  # Ensure minimum range of 50%
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, range_max],
                ticksuffix='%',
                showticklabels=True,
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                tickfont_size=10,
                rotation=90
            )
        ),
        showlegend=True,
        title=dict(
            text="Top Predictions - Confidence Radar",
            x=0.5,
            font=dict(size=16)
        ),
        font=dict(size=12),
        height=500,
        margin=dict(t=80, b=40, l=40, r=40)
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

    # Use only the fallback treatment plan (chatbot functionality removed)
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
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/mrvishalg2004/chatbot_farmcare_ai',
        'Report a bug': 'https://github.com/mrvishalg2004/chatbot_farmcare_ai/issues',
        'About': '# FarmCare AI\nPlant Disease Detection Tool using Vision Transformer'
    }
)

# -----------------------------
# Floating Chatbot Button Component
# -----------------------------

def add_chatbot_button():
    """Add the floating chatbot button to the app using multiple fallback methods"""
    
    # Method 1: Direct HTML/CSS method
    chatbot_html = """
    <div style="position: fixed; bottom: 20px; right: 20px; z-index: 9999;">
        <a href="https://farmcareaichatbot.vercel.app/" target="_blank"
           style="background-color: #4CAF50; 
                  color: white; 
                  border-radius: 30px;
                  padding: 12px 20px;
                  text-align: center;
                  text-decoration: none;
                  display: inline-block;
                  font-size: 16px;
                  font-weight: bold;
                  font-family: Arial, sans-serif;
                  cursor: pointer;
                  box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                  transition: all 0.3s ease;">
            Chat with AI 🤖
        </a>
    </div>
    """
    st.markdown(chatbot_html, unsafe_allow_html=True)
    
    # Method 2: External JS for more advanced behavior (as fallback)
    try:
        js_path = pathlib.Path("chatbot_button.js").absolute()
        with open(js_path, "r", encoding="utf-8") as js_file:
            chatbot_js = js_file.read()
        components.html(f"<script>{chatbot_js}</script>", height=0)
    except Exception:
        pass

# Initialize chat widget state in session
if 'enable_tawk' not in st.session_state:
    st.session_state.enable_tawk = True

# -----------------------------
# Language Selection Sidebar
# -----------------------------
# Add the chatbot button
add_chatbot_button()

st.sidebar.markdown("### 🌐 Language / भाषा / मराठी / 语言")
selected_language = st.sidebar.selectbox(
    "Choose your language / भाषा निवडा / भाषा निवडा:",
    options=list(LANGUAGES.keys()),
    index=0,
    key='language_selector'
)
current_lang = LANGUAGES[selected_language]

# Chat support removed
    
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

        # Reset seed to ensure randomness doesn't affect predictions
        torch.manual_seed(int(time.time()))
        
        # Preprocess the image and prepare for model input
        img_tensor = transform(image).unsqueeze(0)
        
        # Perform model inference with fresh processing
        with torch.no_grad():
            # Clear any cached computations
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Run the model
            outputs = model(img_tensor)
            
            # Process the model outputs
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
            result = generate_gradcam_heatmap(model, img_tensor, pred_class.item(), image)
            
            if len(result) == 3:
                cam_image, grayscale_cam, intensity_stats = result
            else:
                cam_image, grayscale_cam = result
                intensity_stats = None
        
        if cam_image is not None and isinstance(grayscale_cam, np.ndarray):
            # Display the three main images
            col_grad1, col_grad2, col_grad3 = st.columns(3)

            with col_grad1:
                st.image(image, caption=get_text("original_image", current_lang), use_column_width=True)
            
            with col_grad2:
                heatmap_colored = plt.cm.jet(grayscale_cam)[:,:,:3]
                st.image(heatmap_colored, caption=get_text("attention_heatmap", current_lang), use_column_width=True)
            
            with col_grad3:
                st.image(cam_image, caption=get_text("ai_focus_overlay", current_lang), use_column_width=True)
            
            # Add intensity analysis section
            if intensity_stats is not None:
                st.markdown("---")
                st.subheader(get_text('intensity_analysis', current_lang))
                st.write(get_text('intensity_description', current_lang))
                
                # Display intensity statistics
                col_stats1, col_stats2 = st.columns(2)
                
                with col_stats1:
                    st.markdown(f"### {get_text('intensity_stats', current_lang)}")
                    st.metric("Maximum Intensity", f"{intensity_stats['max_intensity']:.3f}", 
                             help="Highest attention value in the heatmap (0-1 scale)")
                    st.metric("Average Intensity", f"{intensity_stats['mean_intensity']:.3f}", 
                             help="Average attention across the entire image")
                    st.metric("Intensity Variation", f"{intensity_stats['std_intensity']:.3f}", 
                             help="Standard deviation - higher values indicate more focused attention")
                
                with col_stats2:
                    st.markdown(f"### {get_text('attention_distribution', current_lang)}")
                    st.metric("High Attention Areas", f"{intensity_stats['high_attention_pixels']:.1f}%", 
                             help="Percentage of pixels with >50% attention intensity")
                    st.metric("Medium Attention Areas", f"{intensity_stats['medium_attention_pixels']:.1f}%", 
                             help="Percentage of pixels with 30-50% attention intensity")
                    st.metric("Low Attention Areas", f"{intensity_stats['low_attention_pixels']:.1f}%", 
                             help="Percentage of pixels with <30% attention intensity")
                
                # Display intensity charts
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    intensity_hist = create_intensity_analysis_chart(grayscale_cam, intensity_stats)
                    st.plotly_chart(intensity_hist, use_container_width=True)
                
                with col_chart2:
                    attention_pie = create_attention_pie_chart(intensity_stats)
                    st.plotly_chart(attention_pie, use_container_width=True)

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
        
        # Show debug information
        with st.expander("Debug Information (Click to expand)", expanded=False):
            st.write(f"Total number of classes: {len(CLASS_NAMES)}")
            st.write(f"Model output shape: {probs.shape}")
            st.write(f"Top 5 probabilities: {torch.topk(probs, 5).values.tolist()}")
            st.write(f"Top 5 class indices: {torch.topk(probs, 5).indices.tolist()}")
            top_5_names = [CLASS_NAMES[i] for i in torch.topk(probs, 5).indices.tolist()]
            st.write(f"Top 5 class names: {top_5_names}")
        
        # Create the radar chart with fresh data
        with st.spinner('Generating confidence radar chart...'):
            # Force regeneration of the visualization
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
# Chatbot Implementation Removed
# -----------------------------

# Check URL parameters to show chat if requested via URL
params = st.query_params
if "show_chat" in params and params["show_chat"] == "true":
    st.session_state.chat_visible = True
    # Remove parameter after processing
    new_params = dict(params)
    del new_params["show_chat"]
    st.query_params.update(**new_params)
    
# Language detection function removed

def get_fallback_response(user_input):
    """Provide pre-defined fallback responses for common plant-related queries"""
    user_input_lower = user_input.lower()
    
    # Dictionary of common plant questions and their responses
    fallback_responses = {
        # Common vegetables
        "tomato": "Tomato plants are commonly affected by diseases like Early Blight, Late Blight, and Septoria Leaf Spot. 🍅\n\n"
                  "• Early Blight: Look for dark spots with concentric rings on lower leaves\n"
                  "• Late Blight: Watch for dark greasy spots that quickly enlarge\n"
                  "• Treatment: Ensure good air circulation, avoid wetting foliage when watering, and consider copper-based fungicides for severe cases.",
        
        "potato": "Potato plants can suffer from Late Blight, Colorado Potato Beetle, and Scab. 🥔\n\n"
                  "• Late Blight causes dark lesions on leaves and can affect tubers\n"
                  "• Colorado Potato Beetle: Orange-yellow beetles with black stripes eat leaves\n"
                  "• Scab creates corky spots on tuber surface\n"
                  "• Treatment involves proper crop rotation, removing infected plants, and using disease-free seed potatoes.",
        
        "cucumber": "Cucumbers commonly face Powdery Mildew, Bacterial Wilt, and pest issues. 🥒\n\n"
                    "• Powdery Mildew appears as white powder on leaves\n"
                    "• Bacterial Wilt causes sudden wilting (spread by cucumber beetles)\n"
                    "• Treatment includes providing consistent watering, using trellises for air circulation, and applying neem oil for mild issues.",
        
        "pepper": "Pepper plants can experience Blossom End Rot, Bacterial Spot, and Aphid infestations. 🌶️\n\n"
                  "• Blossom End Rot shows as dark sunken spots on fruit bottoms (calcium deficiency)\n"
                  "• Bacterial Spot causes small dark lesions on leaves and fruits\n"
                  "• Treatment includes consistent watering, calcium supplementation, and crop rotation.",

        # Ornamental flowers
        "rose": "Roses commonly face Black Spot, Powdery Mildew, and Aphid infestations. 🌹\n\n"
                "• Black Spot shows as circular black spots with yellow halos\n"
                "• Powdery Mildew appears as white powdery coating on leaves\n"
                "• Aphids cluster on new growth and buds\n"
                "• Treatment: Improve air circulation, avoid overhead watering, and consider neem oil as a natural treatment.",
        
        "orchid": "Orchid care requires specific attention to thrive: 🌸\n\n"
                  "• Water only when potting medium is nearly dry (usually weekly)\n"
                  "• Provide bright, indirect light - never direct sunlight\n"
                  "• Use specialized orchid fertilizer at half strength\n"
                  "• Common issues include root rot (from overwatering) and insufficient light.",
        
        "sunflower": "Sunflowers are generally hardy but can face Downy Mildew and bird damage. 🌻\n\n"
                     "• Downy Mildew appears as yellow patches on upper leaf surfaces\n"
                     "• Birds often attack developing seed heads\n"
                     "• Plant in full sun with well-draining soil\n"
                     "• Protect developing seed heads with mesh bags if bird damage is a concern.",
        
        # Fruit trees
        "apple": "Apple trees commonly face Fire Blight, Apple Scab, and Codling Moth. 🍎\n\n"
                 "• Fire Blight causes branches to appear scorched/burned\n"
                 "• Apple Scab creates olive-green or black spots on leaves and fruits\n"
                 "• Codling Moth larvae tunnel into fruits\n"
                 "• Treatment includes proper pruning for air circulation, removing fallen fruit, and appropriate spray programs.",
        
        "citrus": "Citrus trees can experience Citrus Greening, Citrus Canker, and nutrient deficiencies. 🍊\n\n"
                  "• Citrus Greening causes mottled leaves and misshapen, bitter fruit\n"
                  "• Citrus Canker shows as raised lesions on leaves, stems, and fruit\n"
                  "• Yellow leaves often indicate nutrient deficiencies\n"
                  "• Treatment includes proper fertilization, removing infected parts, and ensuring good drainage.",
        
        # Care practices
        "water": "Proper watering is essential for plant health: 💧\n\n"
                "• Water deeply but infrequently to encourage deep root growth\n"
                "• Most plants prefer soil to dry slightly between waterings\n"
                "• Water at the base to keep foliage dry and reduce disease risk\n"
                "• Morning is generally the best time to water plants.",
        
        "fertiliz": "Guidelines for fertilizing plants: 🌱\n\n"
                   "• Use balanced fertilizers (NPK) for general growth\n"
                   "• Apply during the growing season, following package directions\n"
                   "• Organic options include compost, worm castings, and fish emulsion\n"
                   "• Over-fertilizing can harm plants more than under-fertilizing.",
        
        "prune": "Proper pruning helps plants thrive: ✂️\n\n"
                 "• Remove dead, damaged, or diseased branches first\n"
                 "• Prune flowering shrubs after blooming to avoid cutting flower buds\n"
                 "• Make clean cuts just outside the branch collar (swollen area)\n"
                 "• Sterilize tools between plants to prevent disease spread.",
        
        "transplant": "Successful transplanting tips: 🌱\n\n"
                      "• Water plants thoroughly before and after transplanting\n"
                      "• Transplant on cloudy days or in evening to reduce shock\n"
                      "• Handle plants by their root ball, not stems\n"
                      "• Keep as much soil around roots as possible\n"
                      "• Water well and provide shade for a few days after transplanting.",
        
        # Problems
        "pest": "Common garden pest management: 🐛\n\n"
                "• Identify the specific pest before treating\n"
                "• Natural predators like ladybugs and praying mantises help control many pests\n"
                "• Neem oil, insecticidal soap, and diatomaceous earth are effective organic options\n"
                "• Regularly inspect plants to catch infestations early.",
        
        "aphid": "Dealing with aphid infestations: 🐜\n\n"
                 "• These small, soft-bodied insects cluster on new growth and undersides of leaves\n"
                 "• They cause curled leaves and sticky honeydew that attracts ants\n"
                 "• Control with strong water spray, insecticidal soap, or by introducing ladybugs\n"
                 "• Prevent with healthy plants and by avoiding excessive nitrogen fertilizer.",
        
        "fungus": "Managing fungal plant diseases: 🍄\n\n"
                  "• Most fungal issues thrive in humid, wet conditions\n"
                  "• Improve air circulation around plants\n"
                  "• Water at plant base, not on foliage\n"
                  "• Remove and dispose of infected plant material (don't compost)\n"
                  "• Apply appropriate fungicides early in disease development.",
        
        "soil": "Healthy soil is the foundation of plant health: 🌱\n\n"
                "• Good soil contains organic matter, proper drainage, and beneficial microorganisms\n"
                "• Test your soil to understand pH and nutrient levels\n"
                "• Amend poor soil with compost, aged manure, or specific amendments based on test results\n"
                "• Mulching helps retain moisture, suppress weeds, and add organic matter over time.",
                
        "plant disease": "Plant disease diagnosis steps: 🔍\n\n"
                         "• Note symptoms: leaf spots, wilting, discoloration, or unusual growth\n"
                         "• Check pattern: isolated or spreading across multiple plants\n"
                         "• Consider environmental factors: recent weather, watering practices\n"
                         "• For accurate diagnosis, take clear photos of symptoms and consider consulting with local extension services.",
        
        "wilt": "Understanding plant wilting: 🥀\n\n"
                "• Temporary wilting during heat is normal if plants recover in evening\n"
                "• Persistent wilting suggests root problems, disease, or severe drought\n"
                "• Bacterial/fungal wilts often affect one side of plant first\n"
                "• Check soil moisture, root health, and stem damage to diagnose the cause.",
        
        "yellow": "Yellow leaves on plants can indicate several issues: 🟡\n\n"
                  "• Overwatering: Yellowing and drooping lower leaves\n"
                  "• Nutrient deficiencies: Yellow patterns vary by missing nutrient\n"
                  "• Pest damage: Look for insects or eggs under leaves\n"
                  "• Normal aging: Lower leaves naturally yellow and drop on many plants."
    }
    
    # Check if any keywords from our dictionary are in the user's query
    for keyword, response in fallback_responses.items():
        if keyword in user_input_lower:
            return response
    
    # General response if no keywords match
    return "I'm currently operating in offline mode due to API rate limits. 🌱\n\n" \
           "I can still help with many plant care topics! Try asking about:\n\n" \
           "🥬 Vegetables: tomatoes, potatoes, cucumbers, peppers\n" \
           "🌸 Flowers: roses, orchids, sunflowers\n" \
           "🍎 Fruit Trees: apple trees, citrus trees\n" \
           "💧 Care Practices: watering, fertilizing, pruning, transplanting\n" \
           "🐛 Common Problems: pests, aphids, fungus, wilting, yellow leaves\n" \
           "🌱 Garden Basics: soil health, plant diseases\n\n" \
           "Just include these keywords in your question!"

