#!/bin/bash
# Fix Heroku deployment after account verification

echo "🔧 Fixing Heroku Deployment..."
echo ""

# Check authentication
echo "🔐 Checking Heroku authentication..."
if ! heroku auth:whoami > /dev/null 2>&1; then
    echo "❌ Not logged in to Heroku. Please run:"
    echo "   heroku login"
    exit 1
fi

EMAIL=$(heroku auth:whoami)
echo "✅ Authenticated as: $EMAIL"
echo ""

# Check if account is verified
echo "🎫 Checking account verification..."
echo "If the next command fails with 'verification_required', you need to:"
echo "1. Visit: https://heroku.com/verify" 
echo "2. Add a payment method (won't be charged)"
echo "3. Run this script again"
echo ""

# Try to create app
echo "🏗️  Creating Heroku app..."
echo "Enter your desired app name (or press Enter for auto-generated):"
read -r app_name

if [ -z "$app_name" ]; then
    echo "Creating app with auto-generated name..."
    heroku create
else
    echo "Creating app: $app_name"
    heroku create "$app_name"
fi

# Check if app creation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Heroku app created successfully!"
    
    # Get app info
    APP_URL=$(heroku info -s | grep web_url | cut -d= -f2)
    APP_NAME=$(heroku info -s | grep name | cut -d= -f2)
    
    echo "📱 App Name: $APP_NAME"
    echo "🌐 App URL: $APP_URL"
    echo ""
    
    # Check git remotes
    echo "🔗 Git remotes:"
    git remote -v
    echo ""
    
    # Deploy
    echo "🚀 Deploying to Heroku..."
    git push heroku main
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Deployment successful!"
        echo ""
        echo "Your app is live at: $APP_URL"
        echo ""
        echo "Useful commands:"
        echo "  heroku open          # Open app in browser"
        echo "  heroku logs --tail   # View logs"
        echo "  heroku ps            # Check app status"
        echo ""
    else
        echo ""
        echo "❌ Deployment failed. Check logs:"
        echo "  heroku logs --tail"
    fi
    
else
    echo ""
    echo "❌ Failed to create Heroku app."
    echo ""
    echo "Common reasons:"
    echo "1. Account not verified - visit: https://heroku.com/verify"
    echo "2. App name already taken - try a different name"
    echo "3. Network issues - try again"
    echo ""
fi