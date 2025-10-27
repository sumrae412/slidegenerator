# Quick Google OAuth Setup for Heroku

## Step-by-Step Instructions

### 1. Create Google Cloud Project & Get Credentials

**Go to Google Cloud Console:**
https://console.cloud.google.com/

**Follow these steps:**

1. **Create New Project**
   - Click "Select a project" → "NEW PROJECT"
   - Name: "Slide Generator"
   - Click "CREATE"

2. **Enable Google Slides API**
   - Navigate to: APIs & Services → Library
   - Search: "Google Slides API"
   - Click it → Click "ENABLE"

3. **Configure OAuth Consent Screen**
   - Go to: APIs & Services → OAuth consent screen
   - User Type: **External** → CREATE
   - App information:
     - App name: `Slide Generator`
     - User support email: Your email
     - Developer contact: Your email
   - Click "SAVE AND CONTINUE"

   - Scopes: Click "ADD OR REMOVE SCOPES"
     - Search for: `presentations`
     - Check: `https://www.googleapis.com/auth/presentations`
     - Click "UPDATE" → "SAVE AND CONTINUE"

   - Test users: Add your email
   - Click "SAVE AND CONTINUE" → "BACK TO DASHBOARD"

4. **Create OAuth Credentials**
   - Go to: APIs & Services → Credentials
   - Click "+ CREATE CREDENTIALS" → "OAuth client ID"
   - Application type: **Web application**
   - Name: `Slide Generator Web`
   - Authorized redirect URIs - Click "+ ADD URI":
     ```
     https://slidegen-bc9420216e1c.herokuapp.com/oauth2callback
     ```
   - Click "CREATE"
   - **DOWNLOAD JSON** (download button appears in popup)
   - Save file as `credentials.json`

### 2. Set Environment Variables on Heroku

**You need to set 2 environment variables:**

#### A. Set Redirect URI:
```bash
heroku config:set GOOGLE_REDIRECT_URI="https://slidegen-bc9420216e1c.herokuapp.com/oauth2callback" --app slidegen
```

#### B. Set Credentials JSON:

**Option 1: Using the downloaded credentials.json file**
```bash
# Navigate to where you saved credentials.json
cd /path/to/download/folder

# Copy the entire JSON content as one line and set it
# On Mac/Linux:
heroku config:set GOOGLE_CREDENTIALS_JSON="$(cat credentials.json | tr -d '\n')" --app slidegen

# On Windows PowerShell:
$json = Get-Content credentials.json -Raw | ConvertTo-Json -Compress
heroku config:set GOOGLE_CREDENTIALS_JSON="$json" --app slidegen
```

**Option 2: Manual copy-paste**
```bash
# 1. Open credentials.json in a text editor
# 2. Copy ALL the content (should look like {"web": {"client_id": ...}})
# 3. Run this command and paste when prompted:
heroku config:set GOOGLE_CREDENTIALS_JSON='PASTE_JSON_HERE' --app slidegen
```

**Important:** Make sure the JSON is properly formatted as a single line with no line breaks!

### 3. Verify Configuration

```bash
# Check that variables are set
heroku config --app slidegen

# You should see:
# GOOGLE_CREDENTIALS_JSON: {"web":{"client_id":"..."}}
# GOOGLE_REDIRECT_URI: https://slidegen-bc9420216e1c.herokuapp.com/oauth2callback
```

### 4. Test It Out!

1. **Visit:** https://slidegen-bc9420216e1c.herokuapp.com/
2. **Select** "Google Slides" format
3. **Click** "Authorize Now"
4. **Sign in** with your Google account
5. **Grant** permissions
6. **Upload** a document and convert!

## Troubleshooting

### "Google OAuth not configured" error
- Make sure `GOOGLE_CREDENTIALS_JSON` is set correctly
- Verify the JSON is valid (no syntax errors)
- Check that it's all on one line

### "Redirect URI mismatch" error
- Verify the redirect URI in Google Console matches exactly:
  `https://slidegen-bc9420216e1c.herokuapp.com/oauth2callback`
- Make sure `GOOGLE_REDIRECT_URI` environment variable matches

### "Access blocked: This app has not been verified"
- This is normal for testing
- Click "Advanced" → "Go to Slide Generator (unsafe)"
- Or add your email as a test user in OAuth consent screen

### Still having issues?
```bash
# Check Heroku logs for details
heroku logs --tail --app slidegen
```

## Quick Reference

**Your app URL:** https://slidegen-bc9420216e1c.herokuapp.com/

**OAuth callback URL:** https://slidegen-bc9420216e1c.herokuapp.com/oauth2callback

**Google Cloud Console:** https://console.cloud.google.com/

## Security Notes

- Keep your `credentials.json` file private
- Never commit it to Git (already in .gitignore)
- The environment variable approach keeps it secure on Heroku
- Only authorized test users can access the app while in development
