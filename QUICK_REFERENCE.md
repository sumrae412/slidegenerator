# QUICK REFERENCE: OPENAI INTEGRATION POINTS

## 1. HIGH-PRIORITY MODIFICATIONS (Critical for OpenAI Support)

### A. Client Initialization (Line 92-109)
**Current State:**
```python
self.api_key = claude_api_key or os.getenv('ANTHROPIC_API_KEY')
self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
```

**What Needs to Change:**
- Detect API key type (sk- prefix = OpenAI, sk-ant- = Anthropic)
- Initialize appropriate client (OpenAI vs Anthropic)
- Support both API keys simultaneously

**Difficulty:** HIGH | **Scope:** Refactor initialization logic

---

### B. API Retry Logic (Line 180-238)
**Current:** `_call_claude_with_retry(**api_params)`
**Problem:** Anthropic SDK-specific exception handling

**What Needs to Change:**
- Rename to `_call_api_with_retry(provider, **api_params)`
- Add provider-specific error mapping:
  - OpenAI: `openai.RateLimitError`, `openai.APIError`
  - Anthropic: `anthropic.RateLimitError`, `anthropic.APIError`
- Response format is different between SDKs

**Difficulty:** HIGH | **Scope:** Error handling abstraction

---

### C. LLM Bullet Generation (Line 1958-1972)
**Current State:**
```python
if self.api_key and not self.force_basic_mode:
    llm_bullets = self._create_llm_only_bullets(text, context_heading, style)
```

**What Needs to Change:**
- Route to provider-specific method:
  - If OpenAI: `_create_openai_bullets(...)`
  - If Anthropic: `_create_llm_only_bullets(...)` (current)

**Difficulty:** MEDIUM | **Scope:** Strategy routing

---

### D. LLM-Only Bullets Method (Line 2813-2900)
**Current:** Anthropic SDK calls only

**What Needs to Change:**
- Extract provider-agnostic logic (steps 1-4 and 6-7 are identical)
- Create wrapper that handles API calls differently:

**Flow (Same for both providers):**
1. `_detect_content_type()` [no change]
2. `_detect_content_style()` [no change]
3. `_build_structured_prompt()` [no change]
4. **API CALL - DIFFERENT** ← Different here
5. Parse response [different format]
6. `_refine_bullets()` [provider-agnostic but has API call]
7. Return

**Difficulty:** HIGH | **Scope:** Full method refactor

---

### E. Response Parsing (Line 2877-2887)
**Current:** `message.content[0].text` (Anthropic format)
**OpenAI Format:** `message.choices[0].message.content`

**What Needs to Change:**
- Abstract response parsing into provider-specific methods:
  - `_extract_text_from_anthropic_response(message)`
  - `_extract_text_from_openai_response(message)`

**Difficulty:** LOW | **Scope:** Parsing adapter

---

### F. Refinement Pass (Line 2740-2811)
**Current:** Uses `_call_claude_with_retry()` (Anthropic only)

**What Needs to Change:**
- Update to use new `_call_api_with_retry(provider, ...)`
- Pass provider context through refinement flow

**Difficulty:** MEDIUM | **Scope:** Argument passing

---

## 2. MEDIUM-PRIORITY MODIFICATIONS (Optimization)

### G. Model Selection (Line 2869)
**Current:** Hardcoded `"claude-3-5-sonnet-20241022"`

**What Needs to Change:**
```python
# Add config:
self.models = {
    'anthropic': 'claude-3-5-sonnet-20241022',
    'openai': 'gpt-3.5-turbo'  # or gpt-4-turbo
}

# Use in API call:
model = self.models.get(self.provider, 'claude-3-5-sonnet-20241022')
```

**Difficulty:** LOW | **Scope:** Configuration

---

### H. Temperature/Token Adaptation (Line 2845-2863)
**Current:** Works for both (same API params)

**Status:** ✓ NO CHANGES NEEDED

---

## 3. LOW-PRIORITY (No Changes Needed)

✓ Caching system (lines 116-178)
✓ Prompt building (lines 2639-2738)
✓ Content detection (lines 2124-2186)
✓ Style detection (lines 2188-2284)
✓ NLP fallback (line 3140)
✓ Basic fallback (line 1998)
✓ Compression (line 3693)

---

## 4. METHOD CREATION ROADMAP

### Create 4 New Methods:

```python
1. _initialize_client()
   Input: API keys
   Output: Appropriate SDK client
   Complexity: LOW
   
2. _detect_api_key_type(api_key: str) -> str
   Input: API key string
   Output: 'openai' | 'anthropic' | 'unknown'
   Complexity: LOW
   
3. _call_openai_with_retry(**api_params)
   Input: OpenAI API params
   Output: Response object
   Complexity: MEDIUM (copy from _call_claude_with_retry)
   
4. _create_openai_bullets(text, context_heading, style, enable_refinement)
   Input: Content + context
   Output: List[str] bullets
   Complexity: HIGH (copy from _create_llm_only_bullets with API call changes)
```

---

## 5. TESTING POINTS (After Implementation)

### Test with both providers:
```
✓ Parse .txt file → detect content → generate bullets (Anthropic)
✓ Parse .txt file → detect content → generate bullets (OpenAI)
✓ Cache hit scenario (should skip API call for both)
✓ API retry (simulate timeout, test backoff)
✓ Fallback to NLP (if API fails)
✓ Fallback to basic (if NLP fails)
✓ Both keys present - verify correct one is used
```

---

## 6. RISKS & MITIGATION

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Breaking Anthropic flow | CRITICAL | Implement as additive (new methods), keep old methods working |
| Response format differences | HIGH | Test response parsing thoroughly, use adapters |
| Error handling differences | HIGH | Map all error types to standard retry logic |
| Token count differences | MEDIUM | Monitor actual token usage, adjust limits if needed |
| Cost differences | MEDIUM | Document cost differences, add cost tracking |
| Cache key changes | LOW | Cache key is content-based, not provider-based (OK) |

---

## 7. EXISTING BROKEN IMPLEMENTATIONS (REFERENCE ONLY)

⚠️ Do NOT use these as-is:
- Line 4268: `_create_llm_guided_by_nlp()` - Direct HTTP to OpenAI (missing error handling)
- Line 4436: `_create_ai_enhanced_bullets()` - Direct HTTP to OpenAI (duplicate code)
- Line 5572: Unknown method - Uses `self.client.chat.completions` (wrong for Anthropic)

These suggest someone attempted OpenAI integration but:
- Used HTTP requests instead of SDK
- Duplicated logic instead of abstracting
- Not integrated with retry/cache systems
- Not used in main flow

**Action:** Review these for patterns but rewrite cleanly.

