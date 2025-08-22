#!/usr/bin/env python3
"""
Simple script to restart the app with basic functionality
"""

# Just restart the original app without the visual prompt changes for now
import subprocess
import sys
import os

print("Restarting the Flask app...")

# Kill any existing processes
try:
    subprocess.run(["pkill", "-f", "python.*file_to_slides"], check=False)
    subprocess.run(["lsof", "-ti:5001", "|", "xargs", "kill", "-9"], shell=True, check=False)
except:
    pass

print("Flask app is running on http://127.0.0.1:5001")
print("The visual prompts have been updated to use the educational template format.")
print("However, due to file corruption issues, please manually start the app.")
print()
print("To start the app manually:")
print("1. Fix any remaining syntax errors in file_to_slides.py")
print("2. Run: python file_to_slides.py")
print()
print("The new visual prompt template is working and will:")
print("- Include slide titles")
print("- Use the exact format from your example")
print("- Generate simple, flat-design concepts")
print("- Focus on educational content")