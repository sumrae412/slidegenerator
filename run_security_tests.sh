#!/bin/bash

#
# Master security testing script
# Runs all security tests and validations
#

set -e  # Exit on error

echo "=========================================="
echo "Security Testing Suite"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Flask app is running
echo -e "${BLUE}Checking if Flask app is running...${NC}"
if curl -s http://localhost:5000 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Flask app is running${NC}"
else
    echo -e "${RED}✗ Flask app is not running${NC}"
    echo ""
    echo "Please start the Flask app first:"
    echo "  python wsgi.py"
    echo ""
    exit 1
fi

echo ""

# Step 1: Quick Validation
echo "=========================================="
echo "Step 1: Quick Validation (30 seconds)"
echo "=========================================="
echo ""

if ./validate_security.sh http://localhost:5000; then
    echo -e "${GREEN}✓ Quick validation passed${NC}"
else
    echo -e "${RED}✗ Quick validation failed${NC}"
    echo "Fix issues before continuing"
    exit 1
fi

echo ""

# Step 2: Security Audit
echo "=========================================="
echo "Step 2: Security Audit (1 minute)"
echo "=========================================="
echo ""

if python tests/test_api_key_security.py; then
    echo -e "${GREEN}✓ Security audit passed${NC}"
else
    echo -e "${RED}✗ Security audit failed${NC}"
    exit 1
fi

echo ""

# Step 3: Full Test Suite
echo "=========================================="
echo "Step 3: Full Test Suite (2 minutes)"
echo "=========================================="
echo ""

if pytest tests/test_api_key_security.py -v; then
    echo -e "${GREEN}✓ Full test suite passed${NC}"
else
    echo -e "${RED}✗ Full test suite failed${NC}"
    exit 1
fi

echo ""

# Summary
echo "=========================================="
echo "Security Testing Summary"
echo "=========================================="
echo -e "${GREEN}✓ All automated tests passed${NC}"
echo ""
echo "Next steps:"
echo "1. Complete manual checklist (SECURITY_TESTING_CHECKLIST.md)"
echo "2. Review deployment guide (DEPLOYMENT_SECURITY_GUIDE.md)"
echo "3. Deploy when ready (git push heroku main)"
echo ""
echo "=========================================="

exit 0
