#!/bin/bash

echo "=========================================="
echo "Security Validation Script"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Test URL
URL=${1:-http://localhost:5000}

echo "Testing: $URL"
echo ""

PASSED=0
FAILED=0

# Test 1: Encryption key endpoint
echo -n "Test 1: Encryption key endpoint... "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$URL/api/encryption-key")
if [ "$STATUS" -eq 200 ]; then
    echo -e "${GREEN}PASS${NC}"
    ((PASSED++))
else
    echo -e "${RED}FAIL${NC} (HTTP $STATUS)"
    ((FAILED++))
fi

# Test 2: Content-Security-Policy
echo -n "Test 2: Content-Security-Policy... "
CSP=$(curl -s -I "$URL" | grep -i "Content-Security-Policy")
if [ -n "$CSP" ]; then
    echo -e "${GREEN}PASS${NC}"
    ((PASSED++))
else
    echo -e "${RED}FAIL${NC}"
    ((FAILED++))
fi

# Test 3: Cache control
echo -n "Test 3: Cache-Control headers... "
CACHE=$(curl -s -I "$URL" | grep -i "Cache-Control" | grep "no-store")
if [ -n "$CACHE" ]; then
    echo -e "${GREEN}PASS${NC}"
    ((PASSED++))
else
    echo -e "${RED}FAIL${NC}"
    ((FAILED++))
fi

# Test 4: X-Frame-Options
echo -n "Test 4: Clickjacking protection... "
FRAME=$(curl -s -I "$URL" | grep -i "X-Frame-Options")
if [ -n "$FRAME" ]; then
    echo -e "${GREEN}PASS${NC}"
    ((PASSED++))
else
    echo -e "${RED}FAIL${NC}"
    ((FAILED++))
fi

# Test 5: XSS Protection
echo -n "Test 5: XSS protection... "
XSS=$(curl -s -I "$URL" | grep -i "X-XSS-Protection")
if [ -n "$XSS" ]; then
    echo -e "${GREEN}PASS${NC}"
    ((PASSED++))
else
    echo -e "${RED}FAIL${NC}"
    ((FAILED++))
fi

# Test 6: X-Content-Type-Options
echo -n "Test 6: Content-Type Options... "
CONTENT_TYPE=$(curl -s -I "$URL" | grep -i "X-Content-Type-Options")
if [ -n "$CONTENT_TYPE" ]; then
    echo -e "${GREEN}PASS${NC}"
    ((PASSED++))
else
    echo -e "${RED}FAIL${NC}"
    ((FAILED++))
fi

# Test 7: Referrer-Policy
echo -n "Test 7: Referrer-Policy... "
REFERRER=$(curl -s -I "$URL" | grep -i "Referrer-Policy")
if [ -n "$REFERRER" ]; then
    echo -e "${GREEN}PASS${NC}"
    ((PASSED++))
else
    echo -e "${RED}FAIL${NC}"
    ((FAILED++))
fi

# Test 8: HTTPS redirect (if testing production)
if [[ $URL == https://* ]]; then
    echo -n "Test 8: HTTP to HTTPS redirect... "
    HTTP_URL=${URL/https:/http:}
    REDIRECT=$(curl -s -I "$HTTP_URL" | grep -i "Location.*https")
    if [ -n "$REDIRECT" ]; then
        echo -e "${GREEN}PASS${NC}"
        ((PASSED++))
    else
        echo -e "${YELLOW}WARN${NC} (Should redirect to HTTPS)"
    fi
fi

# Test 9: Key validation endpoint
echo -n "Test 9: Key validation endpoint... "
VALIDATION=$(curl -s -X POST "$URL/api/validate-key" \
    -H "Content-Type: application/json" \
    -d '{"key_type":"claude","encrypted_key":"test"}' \
    -w "%{http_code}" -o /dev/null)
if [ "$VALIDATION" -eq 200 ]; then
    echo -e "${GREEN}PASS${NC}"
    ((PASSED++))
else
    echo -e "${RED}FAIL${NC} (HTTP $VALIDATION)"
    ((FAILED++))
fi

echo ""
echo "=========================================="
TOTAL=$((PASSED + FAILED))
SCORE=$((PASSED * 100 / TOTAL))

echo "Results: $PASSED passed, $FAILED failed"
echo "Security Score: $SCORE%"

if [ $SCORE -eq 100 ]; then
    echo -e "${GREEN}🎉 Perfect security score!${NC}"
elif [ $SCORE -ge 80 ]; then
    echo -e "${GREEN}✅ Good security posture${NC}"
elif [ $SCORE -ge 60 ]; then
    echo -e "${YELLOW}⚠️  Security improvements recommended${NC}"
else
    echo -e "${RED}❌ Critical security issues detected${NC}"
fi

echo "=========================================="
echo ""
echo "For detailed testing:"
echo "  pytest tests/test_api_key_security.py -v"
echo ""
echo "For manual testing:"
echo "  See SECURITY_TESTING_CHECKLIST.md"
echo "=========================================="

# Exit with failure if score < 80%
if [ $SCORE -lt 80 ]; then
    exit 1
fi

exit 0
