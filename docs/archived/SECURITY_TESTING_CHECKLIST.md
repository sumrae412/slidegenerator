# Security Testing Checklist

## Pre-Deployment Security Validation

### 1. Browser Storage Security
- [ ] Open browser DevTools → Application → Local Storage
- [ ] Verify API keys are NOT in plaintext
- [ ] Verify keys appear as encrypted gibberish (base64-encoded)
- [ ] Verify timestamp is stored with keys

### 2. Network Traffic Security
- [ ] Open browser DevTools → Network tab
- [ ] Submit a document for processing
- [ ] Check form data in request
- [ ] Verify API keys are sent as `encrypted_claude_key`, NOT `claude_api_key`
- [ ] Verify encrypted values are base64-encoded gibberish
- [ ] Verify NO plaintext keys in request/response

### 3. Security Headers Validation
- [ ] Open browser DevTools → Network → Select any request
- [ ] Check Response Headers
- [ ] Verify `Content-Security-Policy` is present
- [ ] Verify `X-Frame-Options: SAMEORIGIN`
- [ ] Verify `X-Content-Type-Options: nosniff`
- [ ] Verify `Cache-Control: no-store, no-cache`
- [ ] Verify `Strict-Transport-Security` (production only)

### 4. Key Expiration
- [ ] Enter API key and save
- [ ] Check localStorage for timestamp
- [ ] Manually change timestamp to 25 hours ago
- [ ] Refresh page
- [ ] Verify key is cleared automatically
- [ ] Verify no errors in console

### 5. Session Security
- [ ] Get encryption key: `fetch('/api/encryption-key')`
- [ ] Note the key value
- [ ] Open incognito window
- [ ] Get encryption key again
- [ ] Verify keys are different (unique per session)

### 6. XSS Protection
- [ ] Try entering `<script>alert('XSS')</script>` in any input field
- [ ] Verify script does NOT execute
- [ ] Check console for CSP violations

### 7. Audit Logging
- [ ] Check application logs: `heroku logs --tail` (production)
- [ ] Or: `tail -f logs/app.log` (local)
- [ ] Process a document with API key
- [ ] Verify log entry: "API Key Usage: {...}"
- [ ] Verify log does NOT contain actual API key value
- [ ] Verify log contains: timestamp, key_type, action, IP address

### 8. Error Handling
- [ ] Clear session cookies
- [ ] Try to process document
- [ ] Verify graceful error handling
- [ ] Verify user-friendly error message

### 9. Backward Compatibility (Temporary)
- [ ] Send plaintext key (for testing only)
- [ ] Verify fallback still works
- [ ] Plan to remove plaintext support after migration

### 10. End-to-End Flow
- [ ] Enter API key in form
- [ ] Verify green checkmark (validation)
- [ ] Select Google Doc
- [ ] Process document
- [ ] Verify successful processing
- [ ] Verify NO errors in console
- [ ] Verify NO errors in server logs

## Manual Testing Scenarios

### Scenario 1: First-Time User
1. Open app in fresh browser (clear cache)
2. Enter Claude API key
3. Verify encryption happens (check Network tab)
4. Verify key is stored encrypted (check localStorage)
5. Process a document
6. Verify success

### Scenario 2: Returning User
1. Open app with existing encrypted key in localStorage
2. Verify key is automatically loaded
3. Verify key is decrypted on server-side only
4. Process a document
5. Verify success

### Scenario 3: Expired Key
1. Load app with key timestamp > 24 hours old
2. Verify key is cleared automatically
3. Verify user is prompted to re-enter key
4. Enter new key
5. Verify success

### Scenario 4: Multiple Sessions
1. Open app in two different browsers
2. Enter different API keys in each
3. Verify keys don't conflict
4. Process documents in both
5. Verify both succeed independently

### Scenario 5: Security Attack Simulation
1. Try XSS injection in text fields
2. Try SQL injection in inputs
3. Try CSRF attack (form submission from external site)
4. Verify all attempts are blocked
5. Check CSP violation logs

## Browser Compatibility Testing

Test in the following browsers:
- [ ] Chrome (latest)
- [ ] Firefox (latest)
- [ ] Safari (latest)
- [ ] Edge (latest)

For each browser:
- [ ] Encryption/decryption works
- [ ] localStorage works
- [ ] API key validation works
- [ ] Document processing works
- [ ] No console errors

## Performance Testing

### Encryption Performance
- [ ] Measure time to encrypt key (should be < 100ms)
- [ ] Measure time to decrypt key (should be < 100ms)
- [ ] Verify no UI blocking during encryption
- [ ] Verify no lag when entering API key

### Network Performance
- [ ] Check request payload size with encrypted key
- [ ] Verify no significant increase vs. plaintext
- [ ] Monitor network errors
- [ ] Check response times

## Security Regression Testing

Run after any code changes:
- [ ] All automated tests pass (`pytest tests/test_api_key_security.py`)
- [ ] Security audit score ≥ 80% (`python tests/test_api_key_security.py`)
- [ ] No new security warnings in console
- [ ] No new CSP violations
- [ ] All headers still present

## Production Deployment Checklist

Before deploying to production:
- [ ] All tests pass
- [ ] Manual checklist complete
- [ ] Security score ≥ 80%
- [ ] No plaintext keys in code
- [ ] No API keys in git history
- [ ] Environment variables set on Heroku
- [ ] HTTPS enforced
- [ ] Monitoring configured
- [ ] Incident response plan ready

## Post-Deployment Validation

Immediately after deployment:
- [ ] Visit production URL
- [ ] Verify HTTPS is active
- [ ] Check security headers (curl -I https://your-app.herokuapp.com)
- [ ] Test complete flow with real API key
- [ ] Monitor logs for errors
- [ ] Check CSP violation reports

## Continuous Monitoring

Weekly checks:
- [ ] Review CSP violation logs
- [ ] Review API key usage logs
- [ ] Check for dependency vulnerabilities (`pip audit`)
- [ ] Review error logs
- [ ] Check for unusual traffic patterns

## Incident Response

If security issue detected:
1. [ ] Assess severity
2. [ ] Disable affected feature if critical
3. [ ] Review audit logs
4. [ ] Deploy hotfix
5. [ ] Notify affected users if necessary
6. [ ] Document incident and learnings
7. [ ] Update security measures
