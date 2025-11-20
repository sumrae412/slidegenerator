# Security Testing Guide

## Overview

This guide provides instructions for testing the API key encryption security implementation. The testing framework includes automated tests, manual checklists, and validation scripts.

## Quick Start

### 1. Run Quick Validation (30 seconds)

```bash
# Start the Flask app in one terminal
python wsgi.py

# In another terminal, run the validation script
./validate_security.sh http://localhost:5000
```

Expected output:
```
==========================================
Security Validation Script
==========================================

Testing: http://localhost:5000

Test 1: Encryption key endpoint... PASS
Test 2: Content-Security-Policy... PASS
Test 3: Cache-Control headers... PASS
Test 4: Clickjacking protection... PASS
Test 5: XSS protection... PASS
Test 6: Content-Type Options... PASS
Test 7: Referrer-Policy... PASS
Test 9: Key validation endpoint... PASS

==========================================
Results: 8 passed, 0 failed
Security Score: 100%
🎉 Perfect security score!
==========================================
```

### 2. Run Automated Security Audit (1 minute)

```bash
# Make sure Flask app is running
python wsgi.py

# In another terminal
python tests/test_api_key_security.py
```

Expected output:
```
============================================================
SECURITY AUDIT REPORT
============================================================

Results:
✅ PASS - Encryption key endpoint
✅ PASS - Key validation endpoint
✅ PASS - Security headers present
✅ PASS - CSP configured
✅ PASS - Cache control set
✅ PASS - XSS protection enabled
✅ PASS - Clickjacking protection

Security Score: 100.0% (7/7 checks passed)

🎉 Perfect security score!

============================================================
To run full test suite:
  pytest tests/test_api_key_security.py -v
============================================================
```

### 3. Run Full Test Suite (2 minutes)

```bash
# Install pytest if not already installed
pip install pytest requests cryptography

# Run all security tests
pytest tests/test_api_key_security.py -v
```

Expected output:
```
tests/test_api_key_security.py::TestEncryptionEndpoints::test_encryption_key_endpoint_exists PASSED
tests/test_api_key_security.py::TestEncryptionEndpoints::test_encryption_key_is_unique_per_session PASSED
tests/test_api_key_security.py::TestEncryptionEndpoints::test_encryption_key_persists_in_session PASSED
tests/test_api_key_security.py::TestKeyValidation::test_validate_claude_key_format PASSED
tests/test_api_key_security.py::TestKeyValidation::test_validate_openai_key_format PASSED
tests/test_api_key_security.py::TestSecurityHeaders::test_cache_control_headers PASSED
tests/test_api_key_security.py::TestSecurityHeaders::test_content_security_policy PASSED
tests/test_api_key_security.py::TestSecurityHeaders::test_xss_protection_headers PASSED
tests/test_api_key_security.py::TestSecurityHeaders::test_frame_options PASSED
tests/test_api_key_security.py::TestSecurityHeaders::test_referrer_policy PASSED
tests/test_api_key_security.py::TestCSPViolationReporting::test_csp_report_endpoint_exists PASSED

======================== 11 passed in 2.34s ========================
```

## Testing Framework Components

### 1. Automated Tests (`tests/test_api_key_security.py`)

**What it tests:**
- Encryption key generation and session management
- API key validation endpoints
- Security headers (CSP, XSS protection, etc.)
- CSP violation reporting
- Session isolation

**Usage:**
```bash
# Run all tests
pytest tests/test_api_key_security.py -v

# Run specific test class
pytest tests/test_api_key_security.py::TestEncryptionEndpoints -v

# Run specific test
pytest tests/test_api_key_security.py::TestSecurityHeaders::test_cache_control_headers -v

# Run with coverage
pytest tests/test_api_key_security.py --cov=. --cov-report=html
```

### 2. Quick Validation Script (`validate_security.sh`)

**What it tests:**
- Endpoint availability
- Security headers presence
- HTTPS redirect (production)
- Basic functionality

**Usage:**
```bash
# Local testing
./validate_security.sh http://localhost:5000

# Production testing
./validate_security.sh https://your-app.herokuapp.com

# Get exit code (for CI/CD)
./validate_security.sh http://localhost:5000 && echo "PASSED" || echo "FAILED"
```

### 3. Manual Testing Checklist (`SECURITY_TESTING_CHECKLIST.md`)

**What to test:**
- Browser storage security (localStorage)
- Network traffic inspection
- Key expiration behavior
- XSS protection
- Audit logging
- End-to-end flows

**When to use:**
- Before major releases
- After security-related code changes
- During security audits
- When automated tests fail

### 4. Deployment Guide (`DEPLOYMENT_SECURITY_GUIDE.md`)

**Covers:**
- Pre-deployment security checklist
- Environment configuration
- Heroku deployment steps
- Post-deployment validation
- Incident response procedures

**When to use:**
- Before deploying to production
- Setting up new environments
- During security incidents
- For onboarding new team members

## Test Categories

### Category 1: Encryption & Session Security

**Tests:**
- `test_encryption_key_endpoint_exists`
- `test_encryption_key_is_unique_per_session`
- `test_encryption_key_persists_in_session`

**What's verified:**
- Encryption keys are generated properly
- Each session gets a unique key
- Keys persist within a session
- No key leakage between sessions

### Category 2: API Key Validation

**Tests:**
- `test_validate_claude_key_format`
- `test_validate_openai_key_format`

**What's verified:**
- Validation endpoint is accessible
- Claude API key format is validated
- OpenAI API key format is validated
- Invalid keys are rejected

### Category 3: Security Headers

**Tests:**
- `test_cache_control_headers`
- `test_content_security_policy`
- `test_xss_protection_headers`
- `test_frame_options`
- `test_referrer_policy`

**What's verified:**
- CSP prevents unauthorized scripts
- Cache headers prevent sensitive data caching
- XSS protection is enabled
- Clickjacking protection is active
- Referrer policy is set

### Category 4: Compliance & Reporting

**Tests:**
- `test_csp_report_endpoint_exists`
- `test_no_plaintext_keys_in_logs`

**What's verified:**
- CSP violations are logged
- No plaintext keys in logs
- Audit trail is maintained

## Testing Workflows

### Pre-Commit Testing

```bash
# Quick validation before committing code
./validate_security.sh http://localhost:5000
```

### Pre-Deployment Testing

```bash
# 1. Start local server
python wsgi.py &

# 2. Run automated tests
pytest tests/test_api_key_security.py -v

# 3. Run security audit
python tests/test_api_key_security.py

# 4. Quick validation
./validate_security.sh http://localhost:5000

# 5. Manual checklist (SECURITY_TESTING_CHECKLIST.md)
# Complete all items before deployment
```

### Post-Deployment Validation

```bash
# 1. Quick validation
./validate_security.sh https://your-app.herokuapp.com

# 2. Check security headers
curl -I https://your-app.herokuapp.com

# 3. Test encryption endpoint
curl https://your-app.herokuapp.com/api/encryption-key

# 4. Monitor logs
heroku logs --tail

# 5. Complete manual checklist (Section 10: Post-Deployment)
```

### Continuous Monitoring

```bash
# Weekly security check
pytest tests/test_api_key_security.py -v
./validate_security.sh https://your-app.herokuapp.com

# Monthly full audit
python tests/test_api_key_security.py
# Complete full manual checklist
# Review dependency vulnerabilities: pip install safety && safety check
```

## Interpreting Test Results

### Security Score Interpretation

- **100%**: Perfect! All security measures in place
- **80-99%**: Good, but some improvements needed
- **60-79%**: Acceptable, but address failures
- **<60%**: Critical issues, do NOT deploy

### Common Failures and Solutions

#### 1. Encryption Key Endpoint Failed
```
❌ FAIL - Encryption key endpoint
```

**Cause:** Endpoint not implemented or Flask app not running

**Solution:**
```bash
# Check if app is running
curl http://localhost:5000/api/encryption-key

# Restart Flask app
python wsgi.py
```

#### 2. Security Headers Missing
```
❌ FAIL - CSP configured
```

**Cause:** Security headers middleware not applied

**Solution:**
```python
# Check if security_headers() is called in file_to_slides.py
@app.after_request
def security_headers(response):
    # ... security header code
    return response
```

#### 3. Session Isolation Failed
```
FAILED test_encryption_key_is_unique_per_session
```

**Cause:** Encryption keys not properly isolated by session

**Solution:**
```python
# Ensure encryption key is stored in session, not globally
if 'encryption_key' not in session:
    session['encryption_key'] = Fernet.generate_key().decode()
```

## Browser Testing

### Manual Browser Tests

1. **Storage Security**
   ```javascript
   // Open DevTools Console
   localStorage.getItem('encrypted_claude_key')
   // Should show encrypted gibberish, not plaintext
   ```

2. **Network Security**
   ```javascript
   // Open DevTools Network tab
   // Submit form and inspect request
   // Verify encrypted_claude_key (not claude_api_key)
   ```

3. **XSS Protection**
   ```javascript
   // Try injecting script in input field
   <script>alert('XSS')</script>
   // Should NOT execute
   ```

### Browser Compatibility

Test in:
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

Check:
- Encryption/decryption works
- localStorage works
- No console errors
- API key validation works

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Security Tests

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest requests cryptography

    - name: Start Flask app
      run: |
        python wsgi.py &
        sleep 5

    - name: Run security tests
      run: |
        pytest tests/test_api_key_security.py -v

    - name: Run security validation
      run: |
        chmod +x validate_security.sh
        ./validate_security.sh http://localhost:5000
```

### Heroku CI Example

```json
{
  "environments": {
    "test": {
      "scripts": {
        "test": "pytest tests/test_api_key_security.py -v && ./validate_security.sh http://localhost:5000"
      }
    }
  }
}
```

## Troubleshooting

### Tests Won't Run

**Problem:** `ModuleNotFoundError: No module named 'requests'`

**Solution:**
```bash
pip install pytest requests cryptography
```

### App Not Responding

**Problem:** `Connection refused` or `Failed to establish connection`

**Solution:**
```bash
# Check if app is running
curl http://localhost:5000

# Restart app
pkill -f "python wsgi.py"
python wsgi.py
```

### Security Headers Missing

**Problem:** Headers not appearing in responses

**Solution:**
```python
# Verify @app.after_request decorator is applied
@app.after_request
def security_headers(response):
    response.headers['Content-Security-Policy'] = "..."
    # ... other headers
    return response
```

### Session Tests Failing

**Problem:** Same key returned for different sessions

**Solution:**
```python
# Ensure SECRET_KEY is set
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(64))")

# Restart Flask app
python wsgi.py
```

## Security Testing Best Practices

### Do's ✅
- Run tests before every commit
- Complete manual checklist before deployment
- Monitor security scores over time
- Test in multiple browsers
- Review audit logs regularly
- Keep dependencies updated

### Don'ts ❌
- Don't skip tests to save time
- Don't deploy if score < 80%
- Don't ignore test failures
- Don't test with real API keys
- Don't disable security features for testing
- Don't commit test credentials

## Additional Resources

### Documentation
- OWASP Testing Guide: https://owasp.org/www-project-web-security-testing-guide/
- Flask Security: https://flask.palletsprojects.com/en/latest/security/
- Heroku Security: https://devcenter.heroku.com/categories/security

### Tools
- **OWASP ZAP**: Web app security scanner
- **Burp Suite**: Security testing platform
- **Mozilla Observatory**: Security analysis tool
- **Security Headers**: Header checker (securityheaders.com)

### Support
- Project issues: GitHub Issues
- Security concerns: [security@example.com]
- Documentation: This repository

## Summary

The security testing framework provides:

1. **Automated Tests**: Fast, reliable, repeatable tests
2. **Manual Checklists**: Comprehensive coverage of scenarios
3. **Validation Scripts**: Quick smoke tests
4. **Deployment Guides**: Step-by-step security procedures

**Minimum Testing Requirements:**
- ✅ Automated tests pass (pytest)
- ✅ Security score ≥ 80% (audit script)
- ✅ Quick validation passes (validation script)
- ✅ Manual checklist complete (pre-deployment items)

**Only deploy when all requirements are met!**
