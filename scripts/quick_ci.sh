#!/bin/bash

# Quick CI Script for Slide Generator
# Run this before merging PRs to validate quality and functionality

set -e  # Exit on first error

echo "=========================================="
echo "🚀 Quick CI - Slide Generator"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track overall status
FAILED=0

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ PASSED${NC}: $2"
    else
        echo -e "${RED}❌ FAILED${NC}: $2"
        FAILED=1
    fi
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_info "Python version: $PYTHON_VERSION"
echo ""

# Step 2: Check required dependencies
echo "Step 2: Checking dependencies..."
python3 -c "import flask" 2>/dev/null
print_status $? "Flask installed"

python3 -c "import anthropic" 2>/dev/null
print_status $? "Anthropic SDK installed"

python3 -c "import openai" 2>/dev/null
OPENAI_STATUS=$?
if [ $OPENAI_STATUS -eq 0 ]; then
    print_status 0 "OpenAI SDK installed"
else
    print_warning "OpenAI SDK not installed (optional)"
fi

python3 -c "from pptx import Presentation" 2>/dev/null
print_status $? "python-pptx installed"

python3 -c "from googleapiclient.discovery import build" 2>/dev/null
print_status $? "Google API client installed"

echo ""

# Step 3: Check for syntax errors in main files
echo "Step 3: Checking for syntax errors..."
python3 -m py_compile file_to_slides.py 2>/dev/null
print_status $? "file_to_slides.py syntax"

python3 -m py_compile wsgi.py 2>/dev/null
print_status $? "wsgi.py syntax"

if [ -f "slide_generator_pkg/document_parser.py" ]; then
    python3 -m py_compile slide_generator_pkg/document_parser.py 2>/dev/null
    print_status $? "document_parser.py syntax"
fi

echo ""

# Step 4: Run smoke tests if available
echo "Step 4: Running smoke tests..."
if [ -f "tests/smoke_test.py" ]; then
    python3 tests/smoke_test.py
    SMOKE_STATUS=$?
    print_status $SMOKE_STATUS "Smoke tests"
else
    print_warning "No smoke tests found at tests/smoke_test.py (skipping)"
fi

echo ""

# Step 4b: Run feature-specific tests
echo "Step 4b: Running feature-specific tests..."
if [ -f "tests/test_bullet_quality.py" ]; then
    print_info "Running bullet quality tests..."
    python3 -m pytest tests/test_bullet_quality.py -v 2>/dev/null
    BULLET_QUALITY_STATUS=$?
    print_status $BULLET_QUALITY_STATUS "Bullet quality tests"
else
    print_warning "Bullet quality tests not found (skipping)"
fi

if [ -f "tests/test_topic_separation.py" ]; then
    print_info "Running topic separation tests..."
    python3 -m pytest tests/test_topic_separation.py -v 2>/dev/null
    TOPIC_SEP_STATUS=$?
    print_status $TOPIC_SEP_STATUS "Topic separation tests"
else
    print_warning "Topic separation tests not found (skipping)"
fi

echo ""

# Step 5: Check for common security issues
echo "Step 5: Security checks..."

# Check for hardcoded secrets
if grep -r "sk-ant-api" . --exclude-dir=".git" --exclude-dir="__pycache__" --exclude="*.md" 2>/dev/null | grep -v "example\|test\|dummy" > /dev/null; then
    print_status 1 "No hardcoded Anthropic API keys"
else
    print_status 0 "No hardcoded Anthropic API keys"
fi

if grep -r "sk-[a-zA-Z0-9]\{48\}" . --exclude-dir=".git" --exclude-dir="__pycache__" --exclude="*.md" 2>/dev/null | grep -v "example\|test\|dummy\|xxxxx" > /dev/null; then
    print_status 1 "No hardcoded OpenAI API keys"
else
    print_status 0 "No hardcoded OpenAI API keys"
fi

echo ""

# Step 6: Check file structure
echo "Step 6: Checking project structure..."
[ -f "requirements.txt" ]
print_status $? "requirements.txt exists"

[ -f "Procfile" ]
print_status $? "Procfile exists"

if [ -f "runtime.txt" ]; then
    print_status 0 "runtime.txt exists"
else
    print_warning "runtime.txt not found (optional for local dev)"
fi

[ -d "templates" ]
print_status $? "templates/ directory exists"

[ -d "slide_generator_pkg" ]
print_status $? "slide_generator_pkg/ directory exists"

echo ""

# Step 7: Check documentation
echo "Step 7: Checking documentation..."
[ -f "README.md" ]
print_status $? "README.md exists"

[ -f "OPENAI_INTEGRATION.md" ]
OPENAI_DOC_STATUS=$?
if [ $OPENAI_DOC_STATUS -eq 0 ]; then
    print_status 0 "OPENAI_INTEGRATION.md exists"
else
    print_warning "OPENAI_INTEGRATION.md not found (optional)"
fi

echo ""

# Step 8: Run quality metrics if available
echo "Step 8: Running quality metrics..."
if [ -f "tests/quality_metrics.py" ]; then
    print_info "Running quality metrics..."
    python3 tests/quality_metrics.py 2>/dev/null
    QUALITY_STATUS=$?
    print_status $QUALITY_STATUS "Quality metrics"
else
    print_warning "No quality metrics found at tests/quality_metrics.py (skipping)"
fi

echo ""

# Final summary
echo "=========================================="
echo "📊 CI Summary"
echo "=========================================="

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo ""
    echo "Safe to merge PR ✓"
    exit 0
else
    echo -e "${RED}❌ Some checks failed${NC}"
    echo ""
    echo "Please fix the issues above before merging."
    exit 1
fi
