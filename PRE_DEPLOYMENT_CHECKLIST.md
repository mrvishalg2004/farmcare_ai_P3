# 📋 Pre-Deployment Checklist

Use this checklist before deploying your FarmCare AI application.

## ✅ Files Ready

- [ ] `app.py` - Main application file
- [ ] `requirements.txt` - Python dependencies
- [ ] `packages.txt` - System dependencies  
- [ ] `.streamlit/config.toml` - Streamlit configuration
- [ ] `.gitignore` - Git ignore file
- [ ] `README.md` - Documentation
- [ ] `DEPLOYMENT_GUIDE.md` - Deployment instructions
- [ ] Model files (`.pth`) present or downloadable

## ✅ Code Preparation

- [ ] App runs locally without errors: `streamlit run app.py`
- [ ] All imports work correctly
- [ ] Model files load successfully
- [ ] Image upload and processing works
- [ ] All visualizations render properly
- [ ] No hardcoded file paths (use relative paths)
- [ ] API keys moved to environment variables/secrets

## ✅ Dependencies

- [ ] All packages listed in `requirements.txt`
- [ ] Package versions specified (e.g., `torch==2.8.0`)
- [ ] System dependencies listed in `packages.txt`
- [ ] No local-only packages included

## ✅ Security

- [ ] No API keys in code
- [ ] `.env` file in `.gitignore`
- [ ] `.streamlit/secrets.toml` in `.gitignore`
- [ ] Sensitive data not committed to Git

## ✅ GitHub Repository

- [ ] Repository created on GitHub
- [ ] All files pushed to main branch
- [ ] `.gitignore` working correctly
- [ ] README.md updated with project info
- [ ] Repository is public (or you have Streamlit Team/Enterprise)

## ✅ Streamlit Cloud Setup

- [ ] Account created at [share.streamlit.io](https://share.streamlit.io)
- [ ] GitHub account connected
- [ ] Repository access granted to Streamlit

## ✅ App Configuration

- [ ] App title and page config set in code
- [ ] Theme configured in `.streamlit/config.toml`
- [ ] Max upload size configured (if needed)
- [ ] Memory usage optimized

## ✅ Testing

- [ ] Test with different image formats (JPG, PNG)
- [ ] Test with various image sizes
- [ ] Test language switching
- [ ] Test all visualization tabs
- [ ] Check error handling for invalid inputs
- [ ] Verify model predictions are reasonable

## ✅ Model Files

If model files > 100MB:
- [ ] Use Git LFS, or
- [ ] Host externally (Google Drive, Hugging Face, etc.)
- [ ] Add download code in app startup

## ✅ Optional Enhancements

- [ ] Add loading spinners for better UX
- [ ] Implement caching for model loading
- [ ] Add error boundaries
- [ ] Set up analytics (optional)
- [ ] Add feedback mechanism

## 🚀 Deployment Steps

1. **Test Locally**
   ```bash
   ./run_local.sh
   # OR
   streamlit run app.py
   ```

2. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for Streamlit deployment"
   git push origin main
   ```

3. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Choose `app.py` as main file
   - Add secrets if needed
   - Click "Deploy"

4. **Monitor Deployment**
   - Watch logs for errors
   - Test deployed app functionality
   - Share app URL with users

## 📊 Post-Deployment

- [ ] App is accessible via public URL
- [ ] All features work as expected
- [ ] No errors in Streamlit logs
- [ ] Performance is acceptable
- [ ] Custom domain configured (optional)

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Module not found | Add to `requirements.txt` |
| Model not loading | Check file path and size |
| Memory error | Optimize model or use smaller batch size |
| API key error | Add to Streamlit secrets |
| Image upload fails | Check max upload size in config |

## 📝 Notes

- Streamlit Cloud free tier: 1GB RAM, 1 CPU
- Max deployment time: ~5 minutes
- Auto-redeploys on GitHub push
- Logs available in Streamlit dashboard

---

**Ready to deploy? Follow the [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)!** 🚀
