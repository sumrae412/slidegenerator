# Google API Credentials Setup Guide

## Step 1: Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter a project name (e.g., "docs-to-slides-app")
4. Click "Create"

## Step 2: Enable Required APIs

1. In the Google Cloud Console, go to "APIs & Services" → "Library"
2. Search for and enable these APIs:
   - **Google Docs API**
   - **Google Drive API** 
   - **Google Slides API** (optional)

## Step 3: Create OAuth 2.0 Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth 2.0 Client IDs"
3. Configure the OAuth consent screen first if prompted:
   - Choose "External" user type
   - Fill in App name: "Google Docs to Slides"
   - Add your email as support email
   - Add authorized domains if needed (can skip for testing)
4. Choose "Web application" as application type
5. Add these Authorized redirect URIs:
   ```
   http://localhost:5000/callback
   http://127.0.0.1:5000/callback
   ```
6. Click "Create"

## Step 4: Download Credentials

1. Click the download button (⬇️) next to your newly created OAuth client
2. This downloads a JSON file (usually named like `client_secret_xxx.json`)
3. **Rename this file to `credentials.json`**
4. **Move it to your project directory:**
   ```bash
   # If the file is in Downloads:
   mv ~/Downloads/client_secret_*.json /Users/summerrae/claude_code/multi_llm/credentials.json
   ```

## Step 5: Verify Setup

Your project directory should now have:
```
multi_llm/
├── docs_to_slides.py
├── credentials.json  ← This file is required!
├── templates/
│   └── docs_to_slides.html
└── requirements_docs_to_slides.txt
```

## Step 6: Test Authentication

1. Restart your Flask app if it's running
2. Go to http://127.0.0.1:5000
3. Click "Click here to sign in"
4. You should be redirected to Google's OAuth page

## Important Notes

- **Keep `credentials.json` secure** - don't commit it to version control
- **For production**, use environment variables instead of a file
- **The redirect URI must match exactly** what you configured in Google Cloud Console
- **Your app will be in "testing" mode** initially - only add test users in the OAuth consent screen

## Troubleshooting

**"Credentials file not found"**
- Make sure `credentials.json` is in the same directory as `docs_to_slides.py`
- Check the filename is exactly `credentials.json` (not `.txt` or other extension)

**"Redirect URI mismatch"**
- Ensure you added both `http://localhost:5000/callback` and `http://127.0.0.1:5000/callback`
- Check for typos in the redirect URI

**"App not verified"**
- This is normal for testing - click "Advanced" → "Go to [your app] (unsafe)"
- For production, submit your app for verification