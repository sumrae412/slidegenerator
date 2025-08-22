#!/usr/bin/env python3
"""
Semi-Automated Google Cloud Setup for Docs to Slides App

This script automates what's possible via CLI and provides guided steps for manual parts.
Requires: gcloud CLI installed and authenticated
"""

import os
import json
import subprocess
import webbrowser
import time
from typing import Dict, List, Optional

class GoogleCloudSetup:
    def __init__(self):
        self.project_id = None
        self.credentials_file = "credentials.json"
        
    def run_command(self, command: List[str], capture_output=True) -> Dict:
        """Run a shell command and return result"""
        try:
            result = subprocess.run(
                command, 
                capture_output=capture_output, 
                text=True, 
                check=True
            )
            return {
                'success': True,
                'stdout': result.stdout.strip() if capture_output else '',
                'stderr': result.stderr.strip() if capture_output else ''
            }
        except subprocess.CalledProcessError as e:
            return {
                'success': False,
                'stdout': e.stdout.strip() if capture_output and e.stdout else '',
                'stderr': e.stderr.strip() if capture_output and e.stderr else str(e)
            }
        except FileNotFoundError:
            return {
                'success': False,
                'stdout': '',
                'stderr': 'Command not found. Make sure gcloud CLI is installed.'
            }

    def check_gcloud_installed(self) -> bool:
        """Check if gcloud CLI is installed"""
        result = self.run_command(['gcloud', '--version'])
        return result['success']

    def get_current_account(self) -> Optional[str]:
        """Get currently authenticated Google account"""
        result = self.run_command(['gcloud', 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)'])
        if result['success'] and result['stdout']:
            return result['stdout'].split('\n')[0]
        return None

    def authenticate_gcloud(self, email: str = None) -> bool:
        """Authenticate with gcloud"""
        print("🔐 Authenticating with Google Cloud...")
        
        if email:
            print(f"Please authenticate with your {email} account when prompted.")
        
        # Open auth flow
        result = self.run_command(['gcloud', 'auth', 'login'], capture_output=False)
        
        if result['success']:
            account = self.get_current_account()
            if account:
                print(f"✅ Authenticated as: {account}")
                return True
        
        print("❌ Authentication failed")
        return False

    def create_project(self, project_name: str) -> bool:
        """Create a new Google Cloud project"""
        # Generate project ID
        import random
        import string
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        self.project_id = f"docs-to-slides-{suffix}"
        
        print(f"🏗️  Creating project: {self.project_id}")
        
        # Create project
        result = self.run_command([
            'gcloud', 'projects', 'create', self.project_id,
            '--name', project_name
        ])
        
        if not result['success']:
            print(f"❌ Failed to create project: {result['stderr']}")
            return False
        
        # Set as current project
        result = self.run_command(['gcloud', 'config', 'set', 'project', self.project_id])
        
        if result['success']:
            print(f"✅ Project created: {self.project_id}")
            return True
        else:
            print(f"❌ Failed to set current project: {result['stderr']}")
            return False

    def enable_apis(self) -> bool:
        """Enable required APIs"""
        apis = [
            'docs.googleapis.com',
            'drive.googleapis.com',
            'slides.googleapis.com'
        ]
        
        print("🔧 Enabling APIs...")
        
        for api in apis:
            print(f"  Enabling {api}...")
            result = self.run_command(['gcloud', 'services', 'enable', api])
            
            if not result['success']:
                print(f"❌ Failed to enable {api}: {result['stderr']}")
                return False
        
        print("✅ APIs enabled successfully")
        return True

    def open_oauth_consent_screen(self):
        """Open OAuth consent screen in browser"""
        url = f"https://console.cloud.google.com/apis/credentials/consent?project={self.project_id}"
        print(f"🌐 Opening OAuth consent screen: {url}")
        webbrowser.open(url)

    def open_credentials_page(self):
        """Open credentials page in browser"""
        url = f"https://console.cloud.google.com/apis/credentials?project={self.project_id}"
        print(f"🌐 Opening credentials page: {url}")
        webbrowser.open(url)

    def wait_for_credentials_file(self) -> bool:
        """Wait for user to download and place credentials file"""
        print("\n📄 Waiting for credentials.json file...")
        print("Please download your OAuth credentials and save as 'credentials.json' in this directory.")
        
        while True:
            if os.path.exists(self.credentials_file):
                # Validate the file
                try:
                    with open(self.credentials_file, 'r') as f:
                        creds = json.load(f)
                    
                    if 'web' in creds and 'client_id' in creds['web']:
                        print("✅ Valid credentials.json file found!")
                        return True
                    else:
                        print("❌ Invalid credentials format. Please download the correct file.")
                        os.remove(self.credentials_file)
                
                except json.JSONDecodeError:
                    print("❌ Invalid JSON file. Please download the correct credentials file.")
                    if os.path.exists(self.credentials_file):
                        os.remove(self.credentials_file)
            
            response = input("\nPress Enter after downloading credentials.json, or 'q' to quit: ").strip()
            if response.lower() == 'q':
                return False

    def run_automated_setup(self, email: str = None):
        """Run the automated setup process"""
        print("🚀 Starting Google Cloud Setup for Docs to Slides App")
        print("=" * 60)
        
        # Step 1: Check prerequisites
        print("\n1️⃣  Checking prerequisites...")
        if not self.check_gcloud_installed():
            print("❌ gcloud CLI not found.")
            print("Please install it from: https://cloud.google.com/sdk/docs/install")
            return False
        
        print("✅ gcloud CLI found")
        
        # Step 2: Authenticate
        print("\n2️⃣  Authentication...")
        current_account = self.get_current_account()
        
        if current_account:
            print(f"📧 Currently authenticated as: {current_account}")
            if email and email not in current_account:
                print(f"⚠️  You need to authenticate with {email}")
                if not self.authenticate_gcloud(email):
                    return False
            else:
                use_current = input("Use current account? (y/n): ").strip().lower()
                if use_current != 'y':
                    if not self.authenticate_gcloud(email):
                        return False
        else:
            if not self.authenticate_gcloud(email):
                return False
        
        # Step 3: Create project
        print("\n3️⃣  Creating project...")
        project_name = input("Enter project name (default: 'Docs to Slides App'): ").strip()
        if not project_name:
            project_name = "Docs to Slides App"
        
        if not self.create_project(project_name):
            return False
        
        # Step 4: Enable APIs
        print("\n4️⃣  Enabling APIs...")
        if not self.enable_apis():
            return False
        
        # Step 5: Manual OAuth setup
        print("\n5️⃣  OAuth Consent Screen Setup (Manual)")
        print("=" * 40)
        print("The following steps require manual configuration:")
        print("1. Configure OAuth consent screen")
        print("2. Create OAuth 2.0 credentials")
        print("3. Download credentials file")
        
        input("\nPress Enter to open OAuth consent screen...")
        self.open_oauth_consent_screen()
        
        print("\n📋 OAuth Consent Screen Configuration:")
        print("   • User Type: Choose 'Internal' if using org email, 'External' for personal")
        print("   • App name: 'Google Docs to Slides'")
        print("   • User support email: Your email")
        print("   • Developer contact: Your email")
        print("   • Scopes: Add these if asked:")
        print("     - https://www.googleapis.com/auth/documents.readonly")
        print("     - https://www.googleapis.com/auth/drive.readonly")
        
        input("\nPress Enter after configuring consent screen...")
        
        # Step 6: Create credentials
        print("\n6️⃣  Creating OAuth Credentials (Manual)")
        print("=" * 40)
        self.open_credentials_page()
        
        print("\n📋 OAuth Client Configuration:")
        print("   • Click 'Create Credentials' → 'OAuth 2.0 Client IDs'")
        print("   • Application type: 'Web application'")
        print("   • Name: 'Docs to Slides Web Client'")
        print("   • Authorized redirect URIs:")
        print("     - http://localhost:5000/callback")
        print("     - http://127.0.0.1:5000/callback")
        print("   • Click 'Create'")
        print("   • Download the JSON file and save as 'credentials.json'")
        
        # Step 7: Wait for credentials
        print("\n7️⃣  Waiting for credentials file...")
        if not self.wait_for_credentials_file():
            print("❌ Setup cancelled")
            return False
        
        # Step 8: Final validation
        print("\n8️⃣  Final validation...")
        print("✅ Setup complete!")
        print(f"📁 Project ID: {self.project_id}")
        print(f"📄 Credentials: {self.credentials_file}")
        print("\nYou can now restart your Flask app and test authentication!")
        
        return True

def main():
    print("Google Cloud Setup for Docs to Slides App")
    print("=" * 50)
    
    # Get user email
    email = input("Enter your email address (e.g., user@deeplearning.ai): ").strip()
    if not email:
        print("Email is required for proper setup")
        return
    
    setup = GoogleCloudSetup()
    success = setup.run_automated_setup(email)
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("Next steps:")
        print("1. Restart your Flask app: python docs_to_slides.py")
        print("2. Go to http://localhost:5000")
        print("3. Test authentication")
    else:
        print("\n❌ Setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()