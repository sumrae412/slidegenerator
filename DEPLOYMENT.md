# Heroku Deployment Guide

## Prerequisites

1. **Heroku Account**: Sign up at [heroku.com](https://heroku.com)
2. **Heroku CLI**: Install from [devcenter.heroku.com/articles/heroku-cli](https://devcenter.heroku.com/articles/heroku-cli)
3. **Git**: Ensure git is installed and you're in the project directory

## Step-by-Step Deployment

### 1. Initialize Git Repository

```bash
# Navigate to your project directory
cd /Users/summerrae/claude_code/slide_generator

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit for Heroku deployment"
```

### 2. Login to Heroku

```bash
heroku login
```

This will open your browser for authentication.

### 3. Create Heroku Application

```bash
# Create a new Heroku app (replace 'your-app-name' with your desired name)
heroku create your-slide-generator-app

# Or let Heroku generate a random name
heroku create
```

**Note**: App names must be unique across all of Heroku. If your chosen name is taken, try:
- `slides-generator-yourname`
- `doc-to-ppt-converter`
- `script2slides-app`

### 4. Configure Environment Variables (Optional)

If you want to enable enhanced AI features:

```bash
# Set OpenAI API key (optional - app works without it)
heroku config:set OPENAI_API_KEY=your-openai-api-key-here

# Verify config
heroku config
```

### 5. Deploy to Heroku

```bash
# Deploy
git push heroku main
```

If you're on a different branch:
```bash
git push heroku yourbranch:main
```

### 6. Open Your Application

```bash
# Open the app in your browser
heroku open

# Or get the URL
heroku info
```

## Files Required for Deployment

Your project now includes all necessary files:

- ✅ `Procfile` - Tells Heroku how to run your app
- ✅ `requirements.txt` - Python dependencies
- ✅ `runtime.txt` - Python version
- ✅ `wsgi.py` - Production WSGI entry point
- ✅ `.gitignore` - Files to exclude from git

## Deployment Configuration

### Procfile
```
web: gunicorn wsgi:app
```

### Runtime
```
python-3.11.5
```

### Key Dependencies
- Flask 2.3.3
- python-docx 0.8.11
- python-pptx 0.6.21
- gunicorn 21.2.0

## Monitoring Your App

### View Logs
```bash
heroku logs --tail
```

### Check App Status
```bash
heroku ps
```

### Restart App
```bash
heroku restart
```

## Troubleshooting

### Common Issues

1. **Build Failed**
   ```bash
   # Check build logs
   heroku logs --tail
   
   # Ensure all dependencies in requirements.txt
   pip freeze > requirements.txt
   git add requirements.txt
   git commit -m "Update requirements"
   git push heroku main
   ```

2. **App Crashed**
   ```bash
   # Check logs for errors
   heroku logs --tail
   
   # Restart the app
   heroku restart
   ```

3. **Python Version Issues**
   ```bash
   # Update runtime.txt with supported version
   echo "python-3.11.5" > runtime.txt
   git add runtime.txt
   git commit -m "Update Python version"
   git push heroku main
   ```

### Performance Optimization

For better performance on Heroku:

```bash
# Scale to hobby tier for better performance (costs $7/month)
heroku ps:scale web=1:hobby

# Or use free tier (limited hours per month)
heroku ps:scale web=1:eco
```

### Custom Domain (Optional)

```bash
# Add custom domain
heroku domains:add www.yourdomain.com

# Get DNS target
heroku domains
```

## Environment Variables

The app works without any environment variables, but you can enhance it:

```bash
# Optional: OpenAI API Key for enhanced bullet generation
heroku config:set OPENAI_API_KEY=sk-your-key-here

# Optional: Custom file size limits
heroku config:set MAX_FILE_SIZE_MB=20
```

## Updates and Maintenance

### Deploying Updates

```bash
# Make your changes
git add .
git commit -m "Description of changes"
git push heroku main
```

### Database Backups
*Not applicable - this app doesn't use a database*

### Log Retention
```bash
# View recent logs
heroku logs -n 1500

# Continuous log streaming
heroku logs --tail
```

## Cost Estimation

- **Free Tier**: 550-1000 free dyno hours per month
- **Eco Tier**: $5/month for unlimited hours
- **Basic Tier**: $7/month with better performance
- **No database costs** (app doesn't use persistent storage)

## Support and Next Steps

After successful deployment:

1. Test the app thoroughly with various document sizes
2. Monitor logs for any issues
3. Consider upgrading to paid tier for production use
4. Set up custom domain if needed
5. Configure monitoring/alerting

Your app will be accessible at: `https://your-app-name.herokuapp.com`