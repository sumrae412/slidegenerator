# API Key Security Enhancement Plan

**Priority:** HIGH
**Risk Level:** Current implementation has HIGH security risks
**Recommendation:** Implement immediately before public deployment

---

## 🚨 Current Security Issues

### **Critical Vulnerabilities:**

1. **Plaintext localStorage Storage**
   - Location: `templates/file_to_slides.html:1105-1142`
   - Issue: API keys stored unencrypted in browser localStorage
   - Risk: XSS attacks can steal keys, keys persist indefinitely
   - Severity: **HIGH**

2. **No Encryption in Transit (Client → Server)**
   - Location: `templates/file_to_slides.html:1213, 1219`
   - Issue: Keys sent as plain form data
   - Risk: Vulnerable if HTTPS is disabled or compromised
   - Severity: **MEDIUM**

3. **Visible in Browser DevTools**
   - Location: All form submissions
   - Issue: Keys visible in Network tab, Console logs
   - Risk: Shoulder surfing, screen recording attacks
   - Severity: **MEDIUM**

4. **No Key Expiration/Rotation**
   - Location: N/A (not implemented)
   - Issue: Compromised keys remain valid forever
   - Risk: Long-term unauthorized access
   - Severity: **MEDIUM**

---

## ✅ Recommended Security Architecture

### **Option 1: Server-Side Encrypted Storage** (Recommended for production)

**Best for:** Production apps with user accounts

**Architecture:**
```
User → Server (HTTPS) → Encrypted DB → Server decrypts → Use API key → Destroy
```

**Benefits:**
- ✅ Keys never stored in browser
- ✅ Encrypted at rest (database)
- ✅ User accounts enable key management
- ✅ Can implement key rotation, expiration
- ✅ Audit logging possible

**Drawbacks:**
- ❌ Requires user authentication system
- ❌ Requires database (PostgreSQL, etc.)
- ❌ More complex implementation

---

### **Option 2: Session-Based Encrypted Storage** (Recommended for current MVP)

**Best for:** Apps without user accounts, quick MVP

**Architecture:**
```
User → Encrypted in browser (AES) → Send encrypted to server →
Server decrypts with session key → Use API key → Destroy
```

**Benefits:**
- ✅ No database required
- ✅ No user authentication needed
- ✅ Keys encrypted at rest in browser
- ✅ Keys encrypted in transit
- ✅ Session-based (expires automatically)

**Drawbacks:**
- ⚠️ Encryption key stored in session (server-side)
- ⚠️ User must re-enter keys if session expires

---

### **Option 3: No Client Storage (Session-Only)** (Simplest, most secure)

**Best for:** Maximum security, acceptable UX trade-off

**Architecture:**
```
User enters key → Sent to server (HTTPS) → Stored in Flask session →
Session expires after 1 hour → User must re-enter
```

**Benefits:**
- ✅ No browser storage at all
- ✅ Simplest implementation
- ✅ Keys auto-expire with session
- ✅ Lowest attack surface

**Drawbacks:**
- ❌ User must re-enter keys every hour
- ❌ Keys lost on browser refresh (unless session persists)

---

## 🎯 Implementation Plan (Option 2 - Recommended)

### **Phase 1: Client-Side Encryption** (2-3 hours)

**Objective:** Encrypt API keys before storing in browser

#### Step 1.1: Add Crypto Library

```html
<!-- templates/file_to_slides.html -->
<!-- Add before closing </head> -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/crypto-js/4.1.1/crypto-js.min.js"></script>
```

#### Step 1.2: Implement Client-Side Encryption

```javascript
// templates/file_to_slides.html
// Add encryption utilities

const KeyManager = {
    // Generate encryption key from session ID
    async getEncryptionKey() {
        // Fetch session-specific encryption key from server
        const response = await fetch('/api/encryption-key');
        const data = await response.json();
        return data.key;
    },

    // Encrypt API key before storing
    async encryptKey(apiKey) {
        if (!apiKey) return null;

        const encryptionKey = await this.getEncryptionKey();
        const encrypted = CryptoJS.AES.encrypt(apiKey, encryptionKey).toString();
        return encrypted;
    },

    // Decrypt API key when retrieving
    async decryptKey(encryptedKey) {
        if (!encryptedKey) return null;

        const encryptionKey = await this.getEncryptionKey();
        const decrypted = CryptoJS.AES.decrypt(encryptedKey, encryptionKey);
        return decrypted.toString(CryptoJS.enc.Utf8);
    },

    // Save encrypted key to localStorage
    async saveKey(keyType, apiKey) {
        if (!apiKey) {
            localStorage.removeItem(`encrypted_${keyType}_key`);
            return;
        }

        const encrypted = await this.encryptKey(apiKey);
        localStorage.setItem(`encrypted_${keyType}_key`, encrypted);

        // Store timestamp for expiration
        localStorage.setItem(`${keyType}_key_timestamp`, Date.now());
    },

    // Load and decrypt key from localStorage
    async loadKey(keyType) {
        const encrypted = localStorage.getItem(`encrypted_${keyType}_key`);
        if (!encrypted) return null;

        // Check if key has expired (24 hours)
        const timestamp = localStorage.getItem(`${keyType}_key_timestamp`);
        if (timestamp) {
            const age = Date.now() - parseInt(timestamp);
            const maxAge = 24 * 60 * 60 * 1000; // 24 hours

            if (age > maxAge) {
                console.log(`${keyType} key expired, removing`);
                this.clearKey(keyType);
                return null;
            }
        }

        return await this.decryptKey(encrypted);
    },

    // Clear key from storage
    clearKey(keyType) {
        localStorage.removeItem(`encrypted_${keyType}_key`);
        localStorage.removeItem(`${keyType}_key_timestamp`);
    }
};

// Update existing localStorage usage
async function loadSavedKeys() {
    // Load Claude key
    const claudeKey = await KeyManager.loadKey('claude');
    if (claudeKey) {
        document.getElementById('claude-key').value = claudeKey;
        validateApiKey();
    }

    // Load OpenAI key
    const openaiKey = await KeyManager.loadKey('openai');
    if (openaiKey) {
        document.getElementById('openai-key').value = openaiKey;
    }
}

// Update key save on change
document.getElementById('claude-key').addEventListener('change', async function() {
    const key = this.value.trim();
    await KeyManager.saveKey('claude', key);
});

document.getElementById('openai-key').addEventListener('change', async function() {
    const key = this.value.trim();
    await KeyManager.saveKey('openai', key);
});

// Clear key button
document.getElementById('clear-key-btn').addEventListener('click', async function() {
    KeyManager.clearKey('claude');
    document.getElementById('claude-key').value = '';
    document.getElementById('key-validation').classList.add('hidden');
    this.classList.add('hidden');
});

// Load keys on page load
window.addEventListener('DOMContentLoaded', loadSavedKeys);
```

---

### **Phase 2: Server-Side Encryption Key Management** (1-2 hours)

**Objective:** Generate session-specific encryption keys

#### Step 2.1: Add Encryption Key Endpoint

```python
# file_to_slides.py

from cryptography.fernet import Fernet
import secrets

# Add to app initialization
def get_or_create_session_encryption_key():
    """
    Get or create encryption key for current session.
    This key is used to encrypt API keys in browser localStorage.
    """
    if 'encryption_key' not in flask.session:
        # Generate new key for this session
        flask.session['encryption_key'] = secrets.token_urlsafe(32)

    return flask.session['encryption_key']

@app.route('/api/encryption-key', methods=['GET'])
def get_encryption_key():
    """
    Return session-specific encryption key for client-side encryption.
    This key is used to encrypt API keys before storing in localStorage.
    """
    try:
        encryption_key = get_or_create_session_encryption_key()

        return jsonify({
            'status': 'success',
            'key': encryption_key
        })

    except Exception as e:
        logger.error(f"Error getting encryption key: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get encryption key'
        }), 500
```

#### Step 2.2: Add to requirements.txt

```python
# requirements.txt
# Add if not already present
cryptography>=41.0.7
```

---

### **Phase 3: Secure Transmission** (1 hour)

**Objective:** Add additional layer of encryption for API keys in transit

#### Step 3.1: Encrypt Keys Before Sending to Server

```javascript
// templates/file_to_slides.html
// Modify processDocument function

async function processDocument() {
    // ... existing validation ...

    const formData = new FormData();

    // ... existing form data ...

    // Encrypt API keys before sending
    const claudeKey = document.getElementById('claude-key').value.trim();
    if (claudeKey) {
        const encryptedClaude = await KeyManager.encryptKey(claudeKey);
        formData.append('encrypted_claude_key', encryptedClaude);
    }

    const openaiKey = document.getElementById('openai-key').value.trim();
    if (openaiKey) {
        const encryptedOpenai = await KeyManager.encryptKey(openaiKey);
        formData.append('encrypted_openai_key', encryptedOpenai);
    }

    // ... rest of function ...
}
```

#### Step 3.2: Decrypt Keys on Server

```python
# file_to_slides.py

def decrypt_api_key(encrypted_key: str) -> str:
    """
    Decrypt API key received from client.
    Uses session encryption key.
    """
    if not encrypted_key:
        return None

    try:
        import base64
        from cryptography.fernet import Fernet

        # Get session encryption key
        session_key = flask.session.get('encryption_key')
        if not session_key:
            logger.warning("No session encryption key found")
            return None

        # Decrypt using session key
        # (Note: This is simplified - actual implementation needs proper key derivation)
        # For production, use proper KDF like PBKDF2

        # For now, assume client uses same key for AES encryption
        # and we just extract the plaintext
        # This is a placeholder - actual crypto implementation needed

        return encrypted_key  # TEMPORARY - implement proper decryption

    except Exception as e:
        logger.error(f"Failed to decrypt API key: {e}")
        return None

@app.route('/convert', methods=['POST'])
def convert_document():
    # ... existing code ...

    # Get encrypted keys
    encrypted_claude_key = request.form.get('encrypted_claude_key', '').strip()
    encrypted_openai_key = request.form.get('encrypted_openai_key', '').strip()

    # Decrypt keys
    claude_api_key = decrypt_api_key(encrypted_claude_key) if encrypted_claude_key else None
    openai_api_key = decrypt_api_key(encrypted_openai_key) if encrypted_openai_key else None

    # ... rest of function uses decrypted keys ...
```

---

### **Phase 4: Additional Security Measures** (2-3 hours)

#### Step 4.1: Add Key Validation Endpoint

```python
# file_to_slides.py

@app.route('/api/validate-key', methods=['POST'])
def validate_api_key_endpoint():
    """
    Validate API key without storing it.
    Returns validation status only.
    """
    try:
        data = request.json
        key_type = data.get('key_type')  # 'claude' or 'openai'
        encrypted_key = data.get('encrypted_key')

        # Decrypt key
        api_key = decrypt_api_key(encrypted_key)

        if not api_key:
            return jsonify({
                'valid': False,
                'error': 'Invalid key format'
            })

        # Validate based on type
        if key_type == 'claude':
            valid = _validate_claude_api_key(api_key)
        elif key_type == 'openai':
            # Add OpenAI validation
            valid = api_key.startswith('sk-') and len(api_key) > 20
        else:
            valid = False

        return jsonify({
            'valid': valid,
            'key_type': key_type
        })

    except Exception as e:
        logger.error(f"Key validation error: {e}")
        return jsonify({
            'valid': False,
            'error': str(e)
        }), 500
```

#### Step 4.2: Implement Key Expiration UI

```javascript
// templates/file_to_slides.html
// Add expiration warning

function checkKeyExpiration() {
    const timestamp = localStorage.getItem('claude_key_timestamp');
    if (!timestamp) return;

    const age = Date.now() - parseInt(timestamp);
    const maxAge = 24 * 60 * 60 * 1000; // 24 hours
    const warningThreshold = 22 * 60 * 60 * 1000; // 22 hours (warn 2 hours before)

    if (age > warningThreshold && age < maxAge) {
        const hoursLeft = Math.floor((maxAge - age) / (60 * 60 * 1000));
        showNotification(`Your API key will expire in ${hoursLeft} hour(s). Consider re-entering it.`, 'warning');
    }
}

// Check expiration every 30 minutes
setInterval(checkKeyExpiration, 30 * 60 * 1000);
checkKeyExpiration(); // Check on load
```

#### Step 4.3: Add Security Headers

```python
# file_to_slides.py

@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""

    # Prevent browsers from caching API keys
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'

    # Content Security Policy (prevent XSS)
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdnjs.cloudflare.com https://apis.google.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "connect-src 'self' https://accounts.google.com; "
        "frame-src https://accounts.google.com;"
    )

    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'

    # XSS protection
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-XSS-Protection'] = '1; mode=block'

    # HTTPS enforcement (if in production)
    if os.environ.get('FLASK_ENV') == 'production':
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

    return response
```

#### Step 4.4: Add Audit Logging (Optional)

```python
# file_to_slides.py

def log_api_key_usage(key_type: str, action: str, success: bool):
    """
    Log API key usage for security auditing.
    Does NOT log the actual key.
    """
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'key_type': key_type,
        'action': action,  # 'validate', 'use', 'store', 'delete'
        'success': success,
        'ip_address': request.remote_addr,
        'user_agent': request.user_agent.string[:100]
    }

    # Log to file or monitoring service
    logger.info(f"API Key Usage: {json.dumps(log_entry)}")

    # Optional: Send to monitoring service
    # send_to_monitoring_service(log_entry)

# Use in convert endpoint
@app.route('/convert', methods=['POST'])
def convert_document():
    # ... existing code ...

    if claude_api_key:
        log_api_key_usage('claude', 'use', True)

    if openai_api_key:
        log_api_key_usage('openai', 'use', True)

    # ... rest of function ...
```

---

## 🔒 Security Best Practices Checklist

### **Implementation Checklist:**

- [ ] **Remove plaintext localStorage storage**
  - Replace `localStorage.setItem('claude_api_key', key)` with encrypted version
  - Update all `localStorage.getItem()` calls to decrypt

- [ ] **Implement client-side encryption**
  - Add CryptoJS library
  - Implement KeyManager utility
  - Encrypt keys before storing

- [ ] **Add server-side encryption key endpoint**
  - Generate session-specific keys
  - Return keys via secure API

- [ ] **Encrypt keys in transit**
  - Encrypt before sending to server
  - Decrypt on server before use

- [ ] **Add security headers**
  - CSP to prevent XSS
  - HSTS for HTTPS enforcement
  - Cache-Control to prevent key caching

- [ ] **Implement key expiration**
  - 24-hour expiration
  - Warning 2 hours before expiration
  - Auto-clear on expiration

- [ ] **Add audit logging**
  - Log key usage (not actual keys)
  - Track IP, timestamp, action
  - Monitor for suspicious activity

- [ ] **Update UI messaging**
  - Inform users keys are encrypted
  - Show expiration status
  - Provide clear instructions

---

## 📊 Security Comparison

| Feature | Current (❌ Insecure) | After Implementation (✅ Secure) |
|---------|---------------------|----------------------------------|
| Storage encryption | Plaintext | AES-256 encrypted |
| Transmission encryption | HTTPS only | HTTPS + additional encryption |
| Key expiration | Never | 24 hours |
| XSS protection | Vulnerable | CSP headers + encryption |
| Audit logging | None | Full logging |
| Network visibility | Visible | Encrypted payload |
| Session security | Basic | Enhanced with crypto keys |

---

## 🚀 Deployment Checklist

### **Before Production:**

1. **Environment Variables:**
   ```bash
   # Set strong SECRET_KEY
   export SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(64))")

   # Enable HTTPS
   export FLASK_ENV=production
   ```

2. **HTTPS Configuration:**
   - ✅ Enable HTTPS on Heroku (automatic)
   - ✅ Verify SSL certificate valid
   - ✅ Test HSTS headers

3. **Testing:**
   - ✅ Test key encryption/decryption
   - ✅ Test key expiration
   - ✅ Test XSS protection (CSP)
   - ✅ Test audit logging

4. **Monitoring:**
   - ✅ Set up monitoring for suspicious key usage
   - ✅ Alert on failed key validations
   - ✅ Monitor for XSS attempts

---

## 💡 User Communication

### **Update UI Text:**

```html
<!-- templates/file_to_slides.html -->

<!-- Update API key help text -->
<p class="text-xs text-slate-500 mt-1.5">
    🔒 Your API key is encrypted before storage and expires after 24 hours.
    Never shared or logged.
    <a href="https://console.anthropic.com/settings/keys" target="_blank"
       class="text-slate-700 underline hover:text-slate-800">Get a key</a>
</p>

<!-- Add security notice -->
<div class="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
    <p class="text-xs text-blue-900">
        🔒 <strong>Security:</strong> API keys are encrypted with AES-256 before storage,
        never sent to our servers unencrypted, and automatically expire after 24 hours.
        We never log or store your keys permanently.
    </p>
</div>
```

---

## 🎯 Quick Wins (Start Here)

If you only have 1-2 hours:

### **Quick Win 1: Remove localStorage (30 min)**
- Comment out all `localStorage.setItem/getItem` calls
- Force users to re-enter keys each session
- Immediate security improvement

### **Quick Win 2: Add Security Headers (30 min)**
- Implement `add_security_headers()` function
- Test CSP doesn't break existing functionality
- Significant XSS protection

### **Quick Win 3: Key Expiration Warning (30 min)**
- Add timestamp when key is entered
- Show warning after 1 hour
- Clear keys after 2 hours

---

## 📚 Additional Resources

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [CryptoJS Documentation](https://cryptojs.gitbook.io/docs/)
- [Flask Security Best Practices](https://flask.palletsprojects.com/en/2.3.x/security/)
- [Content Security Policy Guide](https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP)

---

## ⚠️ Important Notes

1. **This is a significant security upgrade** - test thoroughly before deploying
2. **Backup current code** before making changes
3. **Test with real API keys** in staging environment
4. **Monitor for issues** after deployment
5. **Consider hiring security audit** before handling sensitive data at scale

---

**Last Updated:** 2025-11-18
**Status:** Ready for implementation
**Priority:** HIGH - Implement before public launch
