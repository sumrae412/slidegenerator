# Google Cloud Setup Guide for @deeplearning.ai

## Quick Setup Steps for Your Organization Email

### Step 1: Create Google Cloud Project
1. **Sign out** of Google Cloud Console if signed in with a different account
2. **Sign in** with your **@deeplearning.ai** email
3. Go to: https://console.cloud.google.com/projectcreate
4. Create a new project:
   - **Project name**: `google-docs-to-slides`
   - Click **"Create"**

### Step 2: Enable Required APIs
1. Go to: https://console.cloud.google.com/apis/library
2. **Search and enable** these APIs (one by one):
   - **Google Docs API** → Click "Enable"
   - **Google Drive API** → Click "Enable"
   - **Google Slides API** → Click "Enable" (optional)

### Step 3: Configure OAuth Consent Screen
1. Go to: https://console.cloud.google.com/apis/credentials/consent
2. **Choose User Type**: **"Internal"** (since you're using @deeplearning.ai)
3. **Fill in App Information**:
   - **App name**: `Google Docs to Slides`
   - **User support email**: Your @deeplearning.ai email
   - **Developer contact email**: Your @deeplearning.ai email
4. **Save and Continue** through all steps

### Step 4: Create OAuth 2.0 Credentials
1. Go to: https://console.cloud.google.com/apis/credentials
2. Click **"+ Create Credentials"** → **"OAuth 2.0 Client IDs"**
3. **Configure the OAuth client**:
   - **Application type**: `Web application`
   - **Name**: `Docs to Slides Web Client`
   - **Authorized redirect URIs** (add both):
     ```
     http://localhost:5000/callback
     http://127.0.0.1:5000/callback
     ```
4. Click **"Create"**
5. **Download** the JSON file

### Step 5: Save Credentials File
1. **Rename** the downloaded file to: `credentials.json`
2. **Move** it to your project directory:
   ```
   /Users/summerrae/claude_code/multi_llm/credentials.json
   ```

### Step 6: Test the Setup
1. **Restart** your Flask app
2. Go to: http://127.0.0.1:5000
3. Click **"Sign in with Google"**
4. **Sign in** with your @deeplearning.ai email
5. Should work without 403 error!

## Why This Should Fix the 403 Error

✅ **Same Organization**: Project and email are both in deeplearning.ai  
✅ **Internal User Type**: No external restrictions  
✅ **Correct Redirect URIs**: Matches the Flask app exactly  
✅ **Proper APIs**: All required APIs enabled  

## Quick Validation

After downloading credentials.json, you can check if it's valid:
```bash
python -c "import json; print('✅ Valid' if 'web' in json.load(open('credentials.json')) else '❌ Invalid')"
```

## If You Still Get 403

Try these additional steps:
1. **Clear browser cache/cookies**
2. **Use incognito/private browsing**
3. **Wait 5-10 minutes** for Google's systems to sync
4. **Double-check redirect URIs** have no typos

---

**Follow these steps manually and the 403 error should be resolved!**