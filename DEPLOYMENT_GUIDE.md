# 🚀 Streamlit Deployment Guide for FarmCare AI

This guide will help you deploy your Plant Disease Detection application on Streamlit Cloud.

## 📋 Prerequisites

1. A GitHub account
2. Your project pushed to a GitHub repository
3. Model files (`.pth` files) in the repository or accessible via URL

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

#### Step 1: Prepare Your Repository

1. **Push your code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - FarmCare AI Plant Disease Detection"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

2. **Ensure these files are in your repository:**
   - ✅ `app.py` (main application)
   - ✅ `requirements.txt`
   - ✅ `packages.txt` (for system dependencies)
   - ✅ `.streamlit/config.toml`
   - ✅ Model files (`.pth`)
   - ✅ `.gitignore`

#### Step 2: Deploy on Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**

2. **Sign in with GitHub**

3. **Click "New app"**

4. **Fill in the deployment form:**
   - **Repository:** `YOUR_USERNAME/YOUR_REPO_NAME`
   - **Branch:** `main`
   - **Main file path:** `app.py`

5. **Add Secrets (if using API keys):**
   - Click on "Advanced settings"
   - Add your secrets in TOML format:
     ```toml
     GEMINI_API_KEY = "your_api_key_here"
     ```

6. **Click "Deploy"**

7. **Wait for deployment** (usually 2-5 minutes)

#### Step 3: Access Your App

Your app will be available at:
```
https://YOUR_APP_NAME.streamlit.app
```

### Option 2: Run Locally

To test your app locally before deploying:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🔧 Configuration

### requirements.txt
Your `requirements.txt` is already configured with the necessary packages:
- `streamlit>=1.32.0`
- `torch==2.8.0`
- `torchvision==0.23.0`
- And other dependencies

### packages.txt
System-level dependencies for image processing:
- `libgl1-mesa-glx` (for OpenCV)
- `libglib2.0-0`

### Environment Variables

If you're using API keys (like Gemini AI), you have two options:

**Option A: For local development**
Create a `.env` file:
```env
GEMINI_API_KEY=your_api_key_here
```

**Option B: For Streamlit Cloud**
Add secrets in Streamlit Cloud dashboard under "Advanced settings"

## 📊 Model Files

Your model files (`best_plant_disease_model.pth` and `plant_disease_model_final.pth`) should be:

1. **Included in the repository** (if < 100MB)
2. **Or hosted externally** (if > 100MB) and downloaded in the app:
   ```python
   import gdown
   
   # Download model from Google Drive
   url = 'https://drive.google.com/uc?id=YOUR_FILE_ID'
   output = 'best_plant_disease_model.pth'
   gdown.download(url, output, quiet=False)
   ```

## 🐛 Troubleshooting

### Common Issues:

1. **Module not found error:**
   - Ensure all dependencies are in `requirements.txt`
   - Check Python version compatibility

2. **Model file not found:**
   - Verify model files are in the repository
   - Check file paths in `app.py`

3. **Memory issues:**
   - Streamlit Cloud free tier has 1GB RAM limit
   - Consider using a smaller model or external hosting

4. **API Key errors:**
   - Verify secrets are properly set in Streamlit Cloud
   - Check that your code reads from `st.secrets`:
     ```python
     api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
     ```

## 🔐 Security Best Practices

1. **Never commit API keys** to GitHub
2. **Add `.env` to `.gitignore`**
3. **Use Streamlit secrets** for production
4. **Keep model files secure** if they contain proprietary data

## 📈 Monitoring & Updates

- **Monitor app performance** in Streamlit Cloud dashboard
- **View logs** for debugging
- **Update app** by pushing to GitHub (auto-redeploys)

## 🎨 Customization

You can customize your app's appearance by editing `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#4CAF50"  # Green for agriculture
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## 📞 Support

- **Streamlit Documentation:** https://docs.streamlit.io
- **Streamlit Community:** https://discuss.streamlit.io
- **GitHub Issues:** Report issues in your repository

## ✅ Deployment Checklist

Before deploying, ensure:
- [ ] Code is pushed to GitHub
- [ ] `requirements.txt` is complete
- [ ] Model files are accessible
- [ ] API keys are in secrets (not code)
- [ ] `.gitignore` includes sensitive files
- [ ] App runs locally without errors
- [ ] All image paths are correct

## 🚀 Quick Deploy Command

After setting up your GitHub repository:

```bash
# One-time setup
git clone YOUR_REPO_URL
cd YOUR_REPO_NAME

# Test locally
pip install -r requirements.txt
streamlit run app.py

# Deploy: Push to GitHub and use Streamlit Cloud UI
git add .
git commit -m "Ready for deployment"
git push
```

Then go to [share.streamlit.io](https://share.streamlit.io) and deploy!

---

**Happy Deploying! 🎉**
