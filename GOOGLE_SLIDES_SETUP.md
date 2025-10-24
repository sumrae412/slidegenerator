# Google Slides Integration Setup Guide

This guide explains how to set up Google Slides integration for the slide generator app.

## Overview

The app now supports creating presentations in both **PowerPoint (.pptx)** and **Google Slides** formats. Google Slides integration requires OAuth2 authentication.

## Prerequisites

- Google Account
- Google Cloud Project with Google Slides API enabled
- OAuth 2.0 credentials

## Setup Steps

### 1. Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Note your project ID for later

### 2. Enable the Google Slides API

1. In your Google Cloud project, go to **APIs & Services > Library**
2. Search for "Google Slides API"
3. Click on it and click **Enable**

### 3. Create OAuth 2.0 Credentials

1. Go to **APIs & Services > Credentials**
2. Click **Create Credentials** > **OAuth client ID**
3. If prompted, configure the OAuth consent screen:
   - Choose **External** user type
   - Fill in required fields (app name, user support email, developer email)
   - Add scopes: `https://www.googleapis.com/auth/presentations`
   - Add test users if needed
4. Back in Create OAuth client ID:
   - Application type: **Web application**
   - Name: "Slide Generator App"
   - Authorized redirect URIs:
     - For local development: `http://localhost:5000/oauth2callback`
     - For Heroku: `https://your-app-name.herokuapp.com/oauth2callback`
5. Click **Create**
6. Download the JSON credentials file

### 4. Configure the Application

#### For Local Development:

1. Save the downloaded JSON file as `credentials.json` in the project root
2. The app will automatically use this file

#### For Heroku Deployment:

You have two options:

**Option A: Upload credentials.json to Heroku**
```bash
# Add credentials.json to your repo (NOT recommended for public repos)
git add credentials.json
git commit -m "Add Google OAuth credentials"
git push heroku main
```

**Option B: Use environment variable (More secure)**
```bash
# Convert credentials.json to base64
base64 -i credentials.json | pbcopy  # macOS
# or
base64 credentials.json              # Linux

# Set as Heroku config var
heroku config:set GOOGLE_CLIENT_SECRETS_BASE64="<paste-base64-here>"
```

Then update the app to decode from environment variable (code modification needed).

### 5. Set Environment Variables

```bash
# For local development (.env file)
GOOGLE_REDIRECT_URI=http://localhost:5000/oauth2callback

# For Heroku
heroku config:set GOOGLE_REDIRECT_URI=https://your-app-name.herokuapp.com/oauth2callback
```

## Usage

### For End Users:

1. **Choose Output Format**: Select "Google Slides" instead of "PowerPoint (.pptx)"
2. **Authorize**: Click "Authorize Now" when prompted
3. **Grant Permissions**: Allow the app to create presentations in your Google Drive
4. **Upload & Convert**: Upload your document and convert
5. **Access**: Click the "Open in Google Slides" link to view your presentation

### Authentication Flow:

1. User selects Google Slides format
2. App shows authentication notice
3. User clicks "Authorize Now"
4. Redirected to Google OAuth consent screen
5. User grants permissions
6. Redirected back to app
7. Credentials stored in session
8. User can now create Google Slides presentations

## Security Considerations

- **Credentials Storage**: Never commit `credentials.json` to public repositories
- **Session Security**: Use strong `FLASK_SECRET_KEY` in production
- **HTTPS Required**: OAuth callbacks require HTTPS in production (Heroku provides this)
- **Scope Limitations**: App only requests `presentations` scope (create/edit slides)
- **Token Refresh**: Credentials include refresh tokens for long-term access

## Troubleshooting

### "Google OAuth not configured" Error
- Ensure `credentials.json` exists in the project root
- Check file permissions
- Verify the file contains valid JSON

### "Authentication Required" Error
- User needs to authorize first
- Session may have expired - re-authorize
- Check that `FLASK_SECRET_KEY` is consistent

### "Redirect URI Mismatch" Error
- Ensure redirect URI in Google Console matches exactly:
  - `http://localhost:5000/oauth2callback` (dev)
  - `https://your-app.herokuapp.com/oauth2callback` (prod)
- Include trailing slash if your URIs have them

### API Quota Exceeded
- Google Slides API has usage limits
- Check [quota limits](https://console.cloud.google.com/apis/api/slides.googleapis.com/quotas)
- Request quota increase if needed

## API Limits

- **Requests per day**: 50,000 (default)
- **Requests per 100 seconds per user**: 300
- **Batch requests**: Up to 500 requests per batch

For most use cases, these limits are sufficient.

## Development vs Production

| Feature | Development | Production |
|---------|-------------|------------|
| Redirect URI | `http://localhost:5000/oauth2callback` | `https://app.herokuapp.com/oauth2callback` |
| Credentials | `credentials.json` file | Environment variable or secure file |
| HTTPS | Not required | Required |
| OAuth Consent | Test users only | Can be made public |

## Additional Resources

- [Google Slides API Documentation](https://developers.google.com/slides/api)
- [OAuth 2.0 for Web Server Applications](https://developers.google.com/identity/protocols/oauth2/web-server)
- [Google Cloud Console](https://console.cloud.google.com/)

## Support

If you encounter issues:
1. Check the app logs: `heroku logs --tail --app your-app-name`
2. Verify credentials are properly configured
3. Ensure Google Slides API is enabled
4. Check OAuth consent screen configuration
