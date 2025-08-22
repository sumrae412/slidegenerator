#!/bin/bash
# Heroku Deployment Script for Script to Slides Generator

echo "🚀 Deploying Script to Slides Generator to Heroku..."
echo ""

# Check if logged in to Heroku
echo "🔐 Checking Heroku authentication..."
if ! heroku auth:whoami > /dev/null 2>&1; then
    echo "❌ Not logged in to Heroku. Please run:"
    echo "   heroku login"
    echo ""
    exit 1
fi

echo "✅ Heroku authentication confirmed"
echo ""

# Check if git repo is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: Script to Slides Generator"
fi

# Check if Heroku remote exists
if ! git remote get-url heroku > /dev/null 2>&1; then
    echo "🏗️  Creating Heroku application..."
    echo "Enter your desired app name (or press Enter for auto-generated):"
    read app_name
    
    if [ -z "$app_name" ]; then
        heroku create
    else
        heroku create "$app_name"
    fi
    echo ""
fi

# Get app info
APP_NAME=$(heroku info | grep "=== " | cut -d' ' -f2)
echo "📱 Deploying to: $APP_NAME"
echo ""

# Deploy
echo "🚀 Deploying to Heroku..."
git push heroku main

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Deployment successful!"
    echo ""
    echo "Your app is available at:"
    heroku info -s | grep web_url | cut -d= -f2
    echo ""
    echo "To open your app:"
    echo "  heroku open"
    echo ""
    echo "To view logs:"
    echo "  heroku logs --tail"
    echo ""
else
    echo ""
    echo "❌ Deployment failed. Check the logs:"
    echo "  heroku logs --tail"
    echo ""
fi