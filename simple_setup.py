#!/usr/bin/env python3
"""
Simple Interactive Setup for Google Docs to Slides App

No CLI tools required - just opens the right pages and guides you through.
"""

import webbrowser
import json
import os
import time

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_step(step_num, title):
    print(f"\n{step_num}️⃣  {title}")
    print("-" * 40)

def wait_for_user():
    input("\n✅ Press Enter when done...")

def open_url_and_wait(url, description):
    print(f"🌐 Opening: {description}")
    print(f"   URL: {url}")
    webbrowser.open(url)
    wait_for_user()

def create_credentials_template():
    """Create a template file to help users"""
    template = {
        "web": {
            "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
            "project_id": "your-project-id", 
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": "YOUR_CLIENT_SECRET",
            "redirect_uris": [
                "http://localhost:5000/callback",
                "http://127.0.0.1:5000/callback"
            ]
        }
    }
    
    with open("credentials_template.json", "w") as f:
        json.dump(template, f, indent=2)
    
    print("📄 Created credentials_template.json for reference")

def validate_credentials():
    """Check if credentials.json is valid"""
    if not os.path.exists("credentials.json"):
        return False, "credentials.json not found"
    
    try:
        with open("credentials.json", "r") as f:
            creds = json.load(f)
        
        if "web" not in creds:
            return False, "Missing 'web' section in credentials"
        
        web = creds["web"]
        required_fields = ["client_id", "client_secret", "auth_uri", "token_uri"]
        
        for field in required_fields:
            if field not in web:
                return False, f"Missing required field: {field}"
        
        return True, "Valid credentials file"
    
    except json.JSONDecodeError:
        return False, "Invalid JSON format"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def main():
    print_header("🚀 Google Docs to Slides - Simple Setup")
    
    print("This script will guide you through setting up Google Cloud credentials.")
    print("No command-line tools required - just follow the steps!")
    
    # Get user info
    email = input("\n📧 Enter your email (e.g., user@deeplearning.ai): ").strip()
    if not email:
        print("❌ Email is required")
        return
    
    user_type = "Internal" if "@" in email and not email.endswith("@gmail.com") else "External"
    
    print(f"\n📋 Setup Summary:")
    print(f"   Email: {email}")
    print(f"   Recommended User Type: {user_type}")
    
    # Step 1: Create Project
    print_step("1", "Create Google Cloud Project")
    print("• Go to Google Cloud Console")
    print("• Click 'Select a project' → 'New Project'")
    print("• Project name: 'Google Docs to Slides'")
    print(f"• Make sure you're signed in with: {email}")
    
    open_url_and_wait(
        "https://console.cloud.google.com/projectcreate", 
        "Google Cloud Project Creation"
    )
    
    # Step 2: Enable APIs
    print_step("2", "Enable Required APIs")
    print("You need to enable these APIs in your project:")
    print("• Google Docs API")
    print("• Google Drive API")
    print("• Google Slides API (optional)")
    
    print("\nFor each API:")
    print("1. Search for the API name")
    print("2. Click on it")
    print("3. Click 'Enable'")
    
    open_url_and_wait(
        "https://console.cloud.google.com/apis/library",
        "API Library"
    )
    
    # Step 3: OAuth Consent Screen
    print_step("3", "Configure OAuth Consent Screen")
    print("Configure these settings:")
    print(f"• User Type: {user_type}")
    print("• App name: Google Docs to Slides")
    print(f"• User support email: {email}")
    print(f"• Developer contact email: {email}")
    
    if user_type == "External":
        print(f"\n⚠️  IMPORTANT: Add {email} as a test user!")
        print("• Scroll down to 'Test users' section")
        print("• Click 'Add Users'")
        print(f"• Enter: {email}")
    
    open_url_and_wait(
        "https://console.cloud.google.com/apis/credentials/consent",
        "OAuth Consent Screen"
    )
    
    # Step 4: Create Credentials
    print_step("4", "Create OAuth 2.0 Credentials")
    print("Create your OAuth credentials:")
    print("• Click 'Create Credentials' → 'OAuth 2.0 Client IDs'")
    print("• Application type: Web application")
    print("• Name: Docs to Slides Web Client")
    print("• Authorized redirect URIs (add both):")
    print("  - http://localhost:5000/callback")
    print("  - http://127.0.0.1:5000/callback")
    print("• Click 'Create'")
    print("• Download the JSON file")
    
    open_url_and_wait(
        "https://console.cloud.google.com/apis/credentials",
        "Credentials Page"
    )
    
    # Step 5: Save Credentials
    print_step("5", "Save Credentials File")
    print("• Rename the downloaded file to: credentials.json")
    print("• Move it to this directory:")
    print(f"  {os.getcwd()}")
    
    create_credentials_template()
    
    # Wait for credentials file
    print("\n📄 Waiting for credentials.json...")
    while True:
        valid, message = validate_credentials()
        if valid:
            print("✅ Valid credentials.json found!")
            break
        else:
            print(f"❌ {message}")
            response = input("Press Enter to check again, or 'q' to quit: ").strip()
            if response.lower() == 'q':
                return
    
    # Final steps
    print_step("6", "Test Your Setup")
    print("✅ Setup complete! Now test it:")
    print("1. Restart your Flask app: python docs_to_slides.py")
    print("2. Go to: http://localhost:5000")
    print("3. Click 'Sign in with Google'")
    print(f"4. Sign in with: {email}")
    
    print("\n🎉 You're all set!")
    
    # Offer to start the app
    start_app = input("\nStart the Flask app now? (y/n): ").strip().lower()
    if start_app == 'y':
        print("🚀 Starting Flask app...")
        os.system("python docs_to_slides.py")

if __name__ == "__main__":
    main()