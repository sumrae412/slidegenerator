# Deployment Security Guide

## Pre-Deployment Checklist

### Environment Variables

#### Local Development
```bash
# Required
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(64))")
export FLASK_ENV=development

# Optional (if using server-side key)
export ANTHROPIC_API_KEY=your-key-here
export OPENAI_API_KEY=your-key-here

# Google OAuth
export GOOGLE_CREDENTIALS_JSON='{"web": {...}}'
```

#### Production (Heroku)
```bash
# Set secure session secret (REQUIRED)
heroku config:set SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(64))")

# Enable production mode
heroku config:set FLASK_ENV=production

# Google OAuth (from credentials.json)
heroku config:set GOOGLE_CREDENTIALS_JSON='{"web": {...}}'

# Optional: Server-side API keys
heroku config:set ANTHROPIC_API_KEY=sk-ant-xxx
heroku config:set OPENAI_API_KEY=sk-xxx

# Verify configuration
heroku config
```

### Code Security Audit

Before deploying, review and remove:

#### 1. Debug Logging
```python
# ❌ REMOVE these before production
console.log("API Key:", apiKey)
logger.debug(f"Decrypted key: {decrypted_key}")
print(f"User key: {user_key}")

# ✅ KEEP these (safe logging)
logger.info("API key validation successful")
logger.info(f"Processing with {key_type} API")
```

#### 2. Hardcoded Secrets
```python
# ❌ REMOVE - never hardcode
ANTHROPIC_API_KEY = "sk-ant-xxx"
SECRET_KEY = "my-secret"

# ✅ USE environment variables
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
```

#### 3. Test/Debug Code
```python
# ❌ REMOVE test code
if True:  # DEBUG
    return jsonify({'decrypted_key': key})

# ❌ REMOVE bypass code
# Temporary bypass for testing
if not encrypted_key:
    encrypted_key = plaintext_key
```

### Dependencies Security

#### Check for Vulnerabilities
```bash
# Install safety checker
pip install safety

# Check for known vulnerabilities
safety check --json

# Update vulnerable packages
pip install --upgrade package-name

# Update requirements.txt
pip freeze > requirements.txt
```

#### Verify Security Dependencies
```bash
# Must be present and up-to-date
pip show cryptography  # Should be >= 41.0.7
pip show pycryptodome  # Should be >= 3.19.0 (if used)
pip show flask         # Should be >= 2.3.3

# Check all package versions
pip list --outdated
```

### Git Security

#### Remove Sensitive Data from History
```bash
# Check for exposed secrets
git log --all --full-history --source --all -- "*.py" | grep -i "sk-ant\|sk-"

# If found, use git-filter-repo to remove
# pip install git-filter-repo
# git filter-repo --path credentials.json --invert-paths
```

#### Update .gitignore
```bash
# Ensure these are ignored
echo "credentials.json" >> .gitignore
echo ".env" >> .gitignore
echo "*.key" >> .gitignore
echo "secrets/" >> .gitignore

# Verify
git status
```

## Deployment Process

### Step 1: Run Security Tests
```bash
# Automated test suite
pytest tests/test_api_key_security.py -v

# Quick validation script
bash validate_security.sh http://localhost:5000

# Manual audit
python tests/test_api_key_security.py
```

Expected output:
```
Security Score: 100.0% (7/7 checks passed)
✅ Good security posture
```

### Step 2: Code Review

Check these critical files:
- [ ] `file_to_slides.py` - No plaintext keys logged
- [ ] `templates/file_to_slides.html` - Encryption enabled
- [ ] `static/js/encryption.js` - No debug logging
- [ ] `.gitignore` - All secrets ignored

### Step 3: Deploy to Heroku
```bash
# Commit changes
git add .
git commit -m "Security: Enable API key encryption"

# Push to Heroku
git push heroku main

# Or push specific branch
git push heroku security/backend-encryption:main
```

### Step 4: Verify Deployment
```bash
# Check dyno status
heroku ps

# Check recent logs
heroku logs --tail

# Test encryption endpoint
curl https://your-app.herokuapp.com/api/encryption-key

# Test security headers
curl -I https://your-app.herokuapp.com
```

## Post-Deployment Validation

### 1. Security Headers Scan

#### Automated Check
```bash
# Using curl
curl -I https://your-app.herokuapp.com

# Expected headers:
# Content-Security-Policy: default-src 'self' ...
# Strict-Transport-Security: max-age=31536000; includeSubDomains
# X-Frame-Options: SAMEORIGIN
# X-Content-Type-Options: nosniff
# X-XSS-Protection: 1; mode=block
# Cache-Control: no-store, no-cache, must-revalidate
# Referrer-Policy: strict-origin-when-cross-origin
```

#### Online Tools
1. **Security Headers Check**: https://securityheaders.com
2. **SSL Labs**: https://www.ssllabs.com/ssltest/
3. **Mozilla Observatory**: https://observatory.mozilla.org/

### 2. SSL/TLS Verification

```bash
# Verify HTTPS redirect
curl -I http://your-app.herokuapp.com
# Should return 301 Moved Permanently
# Location: https://your-app.herokuapp.com

# Check SSL certificate
openssl s_client -connect your-app.herokuapp.com:443 -servername your-app.herokuapp.com

# Expected: Valid certificate chain
```

### 3. Functional Testing

Test complete flow:
1. Open production URL
2. Enter API key
3. Verify encryption (Network tab)
4. Select Google Doc
5. Process document
6. Verify success
7. Check logs (no errors)

### 4. Monitoring Setup

#### Heroku Logging
```bash
# Real-time logs
heroku logs --tail

# Filter for errors
heroku logs --tail | grep ERROR

# Filter for security events
heroku logs --tail | grep "CSP Violation\|API Key Usage"

# Save logs for analysis
heroku logs --num 1500 > deployment_logs.txt
```

#### Log Alerts
```bash
# Install log drain addon (optional)
heroku addons:create papertrail

# Or use Heroku native logging
heroku logs --tail | grep "ERROR\|CSP Violation" | mail -s "Security Alert" admin@example.com
```

#### Metrics to Monitor
- CSP violation frequency
- API key validation failures
- Decryption errors
- Authentication errors
- Unusual traffic patterns

## Production Security Configuration

### Heroku-Specific Settings

```bash
# Enable automatic SSL
heroku certs:auto:enable

# Enable automated certificate management
heroku labs:enable http-sni

# Set session affinity (sticky sessions)
heroku features:enable http-session-affinity

# Restrict access to admin features (if applicable)
heroku config:set ADMIN_WHITELIST=123.45.67.89,98.76.54.32
```

### Application Configuration

Update `file_to_slides.py`:
```python
# Production-only security features
if os.getenv('FLASK_ENV') == 'production':
    # Force HTTPS
    @app.before_request
    def force_https():
        if request.headers.get('X-Forwarded-Proto') == 'http':
            return redirect(request.url.replace('http://', 'https://'), code=301)

    # Enable HSTS
    @app.after_request
    def add_hsts(response):
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
        return response
```

## Incident Response Plan

### If API Key is Compromised

1. **Immediate Actions**
   - User rotates key at Anthropic/OpenAI console
   - User clears browser localStorage
   - Review audit logs for unauthorized usage

2. **Investigation**
   ```bash
   # Search logs for compromised key usage
   heroku logs --num 1500 | grep "API Key Usage"

   # Check for unusual activity
   heroku logs --num 1500 | grep "ERROR\|WARN"

   # Export logs for forensics
   heroku logs --num 1500 > incident_logs.txt
   ```

3. **Communication**
   - Notify affected user
   - Document incident
   - Review security measures
   - Implement additional safeguards if needed

### If Security Vulnerability is Discovered

1. **Assessment**
   - Severity: Critical / High / Medium / Low
   - Scope: Who is affected?
   - Exposure: Has it been exploited?

2. **Mitigation**
   ```bash
   # Disable affected feature (if critical)
   heroku config:set FEATURE_ENABLED=false

   # Deploy hotfix
   git commit -m "Security: Fix vulnerability CVE-XXXX"
   git push heroku main

   # Verify fix
   bash validate_security.sh https://your-app.herokuapp.com
   ```

3. **Recovery**
   - Test fix in staging
   - Deploy to production
   - Monitor for issues
   - Update documentation

4. **Post-Incident**
   - Document findings
   - Update security tests
   - Review prevention measures
   - Share learnings with team

## Ongoing Security Maintenance

### Weekly Tasks
```bash
# Check dependency vulnerabilities
pip install safety
safety check

# Review CSP violations
heroku logs --num 500 | grep "CSP Violation"

# Review API key usage patterns
heroku logs --num 500 | grep "API Key Usage"
```

### Monthly Tasks
```bash
# Update dependencies
pip list --outdated
pip install --upgrade package-name
pip freeze > requirements.txt

# Run full security audit
python tests/test_api_key_security.py
pytest tests/test_api_key_security.py -v

# Review access logs
heroku logs --num 5000 > monthly_logs.txt
# Analyze for unusual patterns
```

### Quarterly Tasks
- Full security penetration test
- Dependency audit (pip audit)
- Code security review
- Update security documentation
- Review incident response plan
- Test backup/recovery procedures

## Security Best Practices

### Do's ✅
- Always use HTTPS in production
- Rotate secrets regularly (every 90 days)
- Monitor logs continuously
- Keep dependencies updated
- Use environment variables for secrets
- Implement rate limiting for API endpoints
- Validate all user inputs
- Log security events (without sensitive data)

### Don'ts ❌
- Never commit secrets to git
- Never log plaintext API keys
- Never disable security headers
- Never use weak session secrets
- Never trust user input without validation
- Never expose internal error details to users
- Never skip security testing before deployment

## Support and Resources

### Documentation
- Heroku Security: https://devcenter.heroku.com/categories/security
- Flask Security: https://flask.palletsprojects.com/en/latest/security/
- OWASP Top 10: https://owasp.org/www-project-top-ten/

### Security Tools
- **OWASP ZAP**: Web application security scanner
- **Burp Suite**: Security testing platform
- **Snyk**: Dependency vulnerability scanner
- **Safety**: Python dependency checker

### Emergency Contacts
- Heroku Support: https://help.heroku.com
- Security Incidents: security@heroku.com
- Project Maintainer: [your-email]

## Deployment Checklist Summary

Pre-Deployment:
- [ ] All security tests pass
- [ ] Dependencies updated and scanned
- [ ] No secrets in code/git
- [ ] Environment variables configured
- [ ] Code reviewed for security issues

Deployment:
- [ ] Deployed to Heroku
- [ ] HTTPS enabled
- [ ] Security headers verified
- [ ] SSL certificate valid

Post-Deployment:
- [ ] Functional testing complete
- [ ] Security headers scanned
- [ ] Monitoring configured
- [ ] Incident response plan ready
- [ ] Documentation updated

**Only deploy if ALL items are checked!**
