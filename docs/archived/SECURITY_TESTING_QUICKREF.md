# Security Testing Quick Reference

## One-Line Commands

```bash
# Quick validation (30 seconds)
./validate_security.sh http://localhost:5000

# Security audit (1 minute)
python tests/test_api_key_security.py

# Full test suite (2 minutes)
pytest tests/test_api_key_security.py -v

# Run all tests (3 minutes)
./run_security_tests.sh

# Production validation
./validate_security.sh https://your-app.herokuapp.com
```

## Before You Commit

```bash
./validate_security.sh http://localhost:5000 && echo "✅ SAFE TO COMMIT" || echo "❌ FIX ISSUES FIRST"
```

## Before You Deploy

```bash
# 1. Run all tests
./run_security_tests.sh

# 2. Check manual items in SECURITY_TESTING_CHECKLIST.md
# 3. Follow DEPLOYMENT_SECURITY_GUIDE.md
```

## After You Deploy

```bash
# Validate production
./validate_security.sh https://your-app.herokuapp.com

# Check headers
curl -I https://your-app.herokuapp.com

# Monitor logs
heroku logs --tail | grep "ERROR\|CSP Violation"
```

## Score Interpretation

| Score | Status | Action |
|-------|--------|--------|
| 100% | 🎉 Perfect | Deploy! |
| 80-99% | ✅ Good | Review failures, then deploy |
| 60-79% | ⚠️ Warning | Fix issues before deploy |
| <60% | ❌ Critical | DO NOT DEPLOY |

## Common Commands

```bash
# Install dependencies
pip install pytest requests cryptography safety

# Start Flask app
python wsgi.py

# Run specific test
pytest tests/test_api_key_security.py::TestSecurityHeaders -v

# Check for vulnerabilities
pip install safety && safety check

# View test coverage
pytest tests/test_api_key_security.py --cov=. --cov-report=html
```

## File Locations

```
tests/test_api_key_security.py          # Automated tests
SECURITY_TESTING_CHECKLIST.md          # Manual checklist
DEPLOYMENT_SECURITY_GUIDE.md           # Deployment guide
SECURITY_TESTING_GUIDE.md              # Complete guide
validate_security.sh                   # Quick validation
run_security_tests.sh                  # Test runner
```

## Critical Checks

**Before Deployment:**
- [ ] All automated tests pass
- [ ] Security score ≥ 80%
- [ ] No plaintext keys in code/logs
- [ ] Environment variables set
- [ ] Manual checklist complete

**After Deployment:**
- [ ] HTTPS working
- [ ] Security headers present
- [ ] No errors in logs
- [ ] End-to-end flow works

## Emergency

If security issue found:
1. Review DEPLOYMENT_SECURITY_GUIDE.md "Incident Response"
2. Disable feature: `heroku config:set FEATURE_ENABLED=false`
3. Check logs: `heroku logs --num 1500 > incident.txt`
4. Deploy fix
5. Validate: `./validate_security.sh https://your-app.herokuapp.com`

## Support

- **Quick help**: SECURITY_TESTING_GUIDE.md → "Troubleshooting"
- **Common failures**: SECURITY_TESTING_GUIDE.md → "Common Failures and Solutions"
- **Deployment**: DEPLOYMENT_SECURITY_GUIDE.md

## Weekly Routine

```bash
# Monday morning security check
./validate_security.sh https://your-app.herokuapp.com
heroku logs --num 500 | grep "CSP Violation"
safety check
```
