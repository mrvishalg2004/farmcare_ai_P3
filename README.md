# 🌱 FarmCare AI - Plant Disease Detection

FarmCare AI is an advanced plant disease detection tool that uses a Vision Transformer (ViT) model to identify plant diseases from leaf images and provides AI-powered treatment recommendations.

## ✨ Features

- 🔬 **Plant Disease Detection**: Advanced Vision Transformer (ViT) model for accurate disease identification
- 💡 **AI Treatment Recommendations**: Personalized treatment plans using Gemini AI
- 🎯 **GradCAM Visualization**: See which parts of the leaf the AI focuses on
- 📊 **Detailed Analytics**: Confidence scores, probability distributions, and model insights
- 🌐 **Multi-language Support**: Available in 13+ languages including English, Hindi, Marathi, Spanish, and more
- 📈 **Advanced Visualizations**: Interactive charts and heatmaps for better understanding

## 🚀 Quick Start

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/chatbot_farmcare_ai.git
cd chatbot_farmcare_ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and go to `http://localhost:8501`

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Add your API keys in the secrets section (if needed)
6. Deploy!

For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 📋 How to Use

1. **Upload Image**: Drag and drop or select a plant leaf image (JPG, PNG)
2. **Wait for Analysis**: The AI will process the image and detect diseases
3. **View Results**: See the predicted disease with confidence scores
4. **Explore Visualizations**: Check GradCAM heatmaps and detailed analytics
5. **Get Treatment Plan**: Read AI-generated treatment recommendations

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Deep Learning**: PyTorch, TorchVision, Timm (Vision Transformer)
- **AI Integration**: Google Gemini AI
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Image Processing**: OpenCV, PIL
- **Explainability**: GradCAM, PyTorch Grad-CAM

## 📁 Project Structure

```
pipline3/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── packages.txt                        # System dependencies
├── best_plant_disease_model.pth       # Trained model file
├── plant_disease_model_final.pth      # Alternative model
├── .streamlit/
│   ├── config.toml                    # Streamlit configuration
│   └── secrets.toml                   # API keys (not in git)
├── DEPLOYMENT_GUIDE.md                # Deployment instructions
└── README.md                          # This file
```

## 🔑 Environment Variables

For API integration (optional), create a `.env` file or add to Streamlit secrets:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## 🌍 Supported Languages

- English
- हिंदी (Hindi)
- मराठी (Marathi)
- Español (Spanish)
- Français (French)
- Deutsch (German)
- Italiano (Italian)
- Português (Portuguese)
- 中文 (Chinese)
- 日本語 (Japanese)
- 한국어 (Korean)
- العربية (Arabic)
- Русский (Russian)

## 📊 Model Information

- **Architecture**: Vision Transformer (ViT)
- **Framework**: PyTorch + Timm
- **Input Size**: 224x224 pixels
- **Features**: Multi-class classification with confidence scores

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for common problems

## 🎉 Deployment

This app can be deployed on:
- ✅ Streamlit Cloud (Recommended)
- ✅ Hugging Face Spaces
- ✅ Heroku
- ✅ AWS/GCP/Azure

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

---

Made with ❤️ for farmers and agriculture enthusiasts