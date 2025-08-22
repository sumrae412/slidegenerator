# 🚀 Quick Heroku Deployment

## Option 1: Automated Script

```bash
# Run the deployment script
./deploy.sh
```

## Option 2: Manual Commands

```bash
# 1. Login to Heroku (opens browser)
heroku login

# 2. Create app with your chosen name
heroku create your-app-name

# 3. Deploy
git push heroku main

# 4. Open your app
heroku open
```

## Your App Structure ✅

```
✅ Procfile          # Heroku config
✅ requirements.txt  # Dependencies  
✅ runtime.txt       # Python version
✅ wsgi.py          # Production server
✅ Git repository   # Ready to deploy
```

## After Deployment

### View Your App
```bash
heroku open
```

### Monitor Logs
```bash
heroku logs --tail
```

### Check Status
```bash
heroku ps
```

### Add OpenAI API Key (Optional)
```bash
heroku config:set OPENAI_API_KEY=your-key-here
```

## Expected Result

Your app will be live at: `https://your-app-name.herokuapp.com`

Features available:
- ✅ Document upload (up to 16MB)
- ✅ PowerPoint generation
- ✅ Visual prompt creation
- ✅ AI bullet points (with API key)

## Troubleshooting

### Build Issues
```bash
heroku logs --tail
```

### App Crashes
```bash
heroku restart
heroku logs --tail
```

### Domain Issues
- App names must be globally unique
- Try: `slides-gen-yourname` or `doc2ppt-yourname`

---
**Ready to deploy!** 🎉