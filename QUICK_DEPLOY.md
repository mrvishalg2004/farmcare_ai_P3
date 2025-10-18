# 🚀 Quick Deploy Commands

## Local Testing

```bash
# First time setup
./setup.sh

# Run the app
./run_local.sh

# Or manually
source venv/bin/activate
streamlit run app.py
```

## Deploy to Streamlit Cloud

```bash
# 1. Initialize Git (if not already done)
git init
git add .
git commit -m "Ready for Streamlit deployment"

# 2. Create GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main

# 3. Go to https://share.streamlit.io
# 4. Click "New app"
# 5. Select your repo and app.py
# 6. Deploy! 🎉
```

## Update Deployed App

```bash
# Make changes, then:
git add .
git commit -m "Your update message"
git push

# Streamlit auto-redeploys! ✨
```

## Environment Setup

```bash
# Create .env for local testing
echo "GEMINI_API_KEY=your_key_here" > .env

# For Streamlit Cloud:
# Add secrets in app settings dashboard
```

## Troubleshooting

```bash
# Check Python version
python3 --version

# Verify Streamlit installed
source venv/bin/activate
streamlit --version

# Test import
python3 -c "import torch; print(torch.__version__)"

# Check model files
ls -lh *.pth
```

## Useful Links

- 🌐 Streamlit Cloud: https://share.streamlit.io
- 📚 Docs: https://docs.streamlit.io
- 💬 Community: https://discuss.streamlit.io
- 📖 Full Guide: See `DEPLOYMENT_GUIDE.md`

---

**Your app URL after deployment:**
```
https://YOUR_APP_NAME.streamlit.app
```
