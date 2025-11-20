# IMPLEMENTATION GUIDE: OpenAI Support for Bullet Generation

## FILE: slide_generator_pkg/document_parser.py

---

## STEP 1: Update Imports (Top of file, around line 80)

**ADD:**
```python
# Line 80: existing import
import anthropic

# ADD after line 80:
from openai import OpenAI
import openai  # For exception handling
```

---

## STEP 2: Modify __init__ Method (Lines 92-124)

**CURRENT CODE (lines 92-109):**
```python
def __init__(self, claude_api_key=None):
    self.heading_patterns = [
        r'^#{1,6}\s+(.+)$',  # Markdown headings
        r'^(.+)\n[=-]{3,}$',  # Underlined headings
        r'^\d+\.\s+(.+)$',   # Numbered headings
        r'^([A-Z][A-Z\s]{5,})$',  # ALL CAPS headings
    ]

    # Store API key for Claude
    self.api_key = claude_api_key or os.getenv('ANTHROPIC_API_KEY')
    self.client = None
    if self.api_key:
        try:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            logger.info("✅ Claude API client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Claude client: {e}")
            self.client = None

    self.force_basic_mode = False  # Flag to override AI processing for large files

    # Initialize semantic analyzer
    self.semantic_analyzer = SemanticAnalyzer()

    # Initialize LRU cache for Claude API responses
    self._api_cache = OrderedDict()
    self._cache_max_size = 1000
    self._cache_hits = 0
    self._cache_misses = 0

    if not self.api_key:
        logger.warning("No Claude API key found - bullet generation will use fallback method")
```

**REPLACE WITH:**
```python
def __init__(self, claude_api_key=None, openai_api_key=None):
    self.heading_patterns = [
        r'^#{1,6}\s+(.+)$',  # Markdown headings
        r'^(.+)\n[=-]{3,}$',  # Underlined headings
        r'^\d+\.\s+(.+)$',   # Numbered headings
        r'^([A-Z][A-Z\s]{5,})$',  # ALL CAPS headings
    ]

    # Support for both Anthropic and OpenAI APIs
    self.anthropic_key = claude_api_key or os.getenv('ANTHROPIC_API_KEY')
    self.openai_key = openai_api_key or os.getenv('OPENAI_API_KEY')
    
    # Backward compatibility: Keep self.api_key pointing to active key
    self.api_key = self.anthropic_key or self.openai_key
    self.provider = None  # Will be set in _initialize_client()
    self.client = None
    
    # Initialize appropriate client based on available keys
    self.client = self._initialize_client()

    self.force_basic_mode = False  # Flag to override AI processing for large files

    # Initialize semantic analyzer
    self.semantic_analyzer = SemanticAnalyzer()

    # Initialize LRU cache for API responses
    self._api_cache = OrderedDict()
    self._cache_max_size = 1000
    self._cache_hits = 0
    self._cache_misses = 0

    # Model configuration per provider
    self.models = {
        'anthropic': 'claude-3-5-sonnet-20241022',
        'openai': 'gpt-3.5-turbo'  # Could be gpt-4-turbo for higher quality
    }

    if not self.api_key:
        logger.warning("No API key found (Claude or OpenAI) - bullet generation will use fallback method")
    else:
        logger.info(f"✅ Using {self.provider} for LLM bullet generation")
```

---

## STEP 3: Add New Helper Methods (After __init__, around line 125)

**ADD AFTER line 124:**
```python
def _initialize_client(self):
    """Initialize appropriate LLM client based on available API keys"""
    # Priority: OpenAI if available, then Anthropic
    if self.openai_key:
        try:
            self.provider = 'openai'
            self.api_key = self.openai_key
            logger.info("Initializing OpenAI client")
            return OpenAI(api_key=self.openai_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.provider = None
            return None
    
    elif self.anthropic_key:
        try:
            self.provider = 'anthropic'
            self.api_key = self.anthropic_key
            logger.info("Initializing Anthropic (Claude) client")
            return anthropic.Anthropic(api_key=self.anthropic_key)
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            self.provider = None
            return None
    
    return None

def _detect_api_key_type(self, api_key: str) -> str:
    """Determine which LLM provider an API key belongs to"""
    if not api_key:
        return 'unknown'
    if api_key.startswith('sk-'):
        return 'openai'  # OpenAI keys start with sk-
    elif api_key.startswith('sk-ant-'):
        return 'anthropic'  # Anthropic keys start with sk-ant-
    else:
        return 'unknown'

def _extract_text_from_response(self, message, provider: str = None) -> str:
    """Extract text from API response based on provider format"""
    provider = provider or self.provider
    
    try:
        if provider == 'openai':
            # OpenAI format: message.choices[0].message.content
            return message.choices[0].message.content.strip()
        else:  # anthropic
            # Anthropic format: message.content[0].text
            return message.content[0].text.strip()
    except (AttributeError, IndexError) as e:
        logger.error(f"Failed to extract text from {provider} response: {e}")
        return ""
```

---

## STEP 4: Replace _call_claude_with_retry (Lines 180-238)

**CURRENT SIGNATURE:**
```python
def _call_claude_with_retry(self, **api_params) -> Any:
```

**REPLACE ENTIRE METHOD WITH:**
```python
def _call_claude_with_retry(self, **api_params) -> Any:
    """
    DEPRECATED: Use _call_api_with_retry instead.
    Kept for backward compatibility - routes to appropriate provider.
    """
    provider = api_params.pop('provider', self.provider)
    return self._call_api_with_retry(provider=provider, **api_params)

def _call_api_with_retry(self, provider: str = None, **api_params) -> Any:
    """
    Call LLM API with exponential backoff retry logic.
    Handles both Anthropic and OpenAI providers.

    Args:
        provider: 'anthropic' or 'openai' (defaults to self.provider)
        **api_params: Parameters to pass to appropriate client API

    Returns:
        API response message

    Raises:
        Exception: After all retries exhausted or on non-retryable errors
    """
    provider = provider or self.provider
    max_retries = 3
    base_delay = 1.0  # Start with 1 second

    for attempt in range(max_retries):
        try:
            if provider == 'openai':
                # OpenAI SDK format
                message = self.client.chat.completions.create(**api_params)
            else:  # anthropic
                # Anthropic SDK format (current behavior)
                message = self.client.messages.create(**api_params)

            # Log success on retry
            if attempt > 0:
                logger.info(f"🔄 API call succeeded on retry {attempt + 1}/{max_retries}")

            return message

        except Exception as e:
            error_str = str(e).lower()
            is_last_attempt = (attempt == max_retries - 1)

            # Determine if error is retryable (same for both providers)
            retryable_errors = [
                'rate limit',
                'timeout',
                'connection',
                'network',
                'server error',
                '429',  # Too Many Requests
                '500',  # Internal Server Error
                '502',  # Bad Gateway
                '503',  # Service Unavailable
                '504',  # Gateway Timeout
            ]

            is_retryable = any(err in error_str for err in retryable_errors)

            if not is_retryable or is_last_attempt:
                # Don't retry on client errors (4xx except 429) or if out of retries
                logger.error(f"❌ API call failed ({provider}): {e}")
                raise

            # Calculate exponential backoff delay
            delay = base_delay * (2 ** attempt)
            logger.warning(f"⚠️  API call failed (attempt {attempt + 1}/{max_retries}, {provider}): {e}")
            logger.info(f"🔄 Retrying in {delay:.1f}s...")
            time.sleep(delay)

    # Should never reach here, but just in case
    raise Exception("Max retries exhausted")
```

---

## STEP 5: Update _create_unified_bullets (Line 1907)

**FIND LINE 1958-1972:**
```python
# Try LLM first if API key is available
if self.api_key and not self.force_basic_mode:
    logger.info("Using enhanced LLM approach with structured prompts")
    # Auto-detect style based on content and context
    style = self._detect_content_style(text, context_heading)
    llm_bullets = self._create_llm_only_bullets(
        text,
        context_heading=context_heading,
        style=style,
        enable_refinement=False  # Set to True for extra quality pass (uses more API tokens)
    )
    if llm_bullets and len(llm_bullets) >= 1:
        logger.info(f"✅ LLM SUCCESS: Generated {len(llm_bullets)} LLM bullets")
        unique_bullets = self._deduplicate_bullets(llm_bullets)
        self._cache_response(cache_key, unique_bullets[:4])  # Cache LLM bullets
        return unique_bullets[:4]
    else:
        logger.warning("LLM approach failed - falling back to lightweight NLP")
```

**REPLACE WITH:**
```python
# Try LLM first if API key is available
if self.api_key and not self.force_basic_mode:
    logger.info(f"Using enhanced LLM approach ({self.provider}) with structured prompts")
    # Auto-detect style based on content and context
    style = self._detect_content_style(text, context_heading)
    
    # Route to appropriate provider-specific method
    if self.provider == 'openai':
        llm_bullets = self._create_openai_bullets(
            text,
            context_heading=context_heading,
            style=style,
            enable_refinement=False
        )
    else:  # anthropic
        llm_bullets = self._create_llm_only_bullets(
            text,
            context_heading=context_heading,
            style=style,
            enable_refinement=False
        )
    
    if llm_bullets and len(llm_bullets) >= 1:
        logger.info(f"✅ LLM SUCCESS ({self.provider}): Generated {len(llm_bullets)} LLM bullets")
        unique_bullets = self._deduplicate_bullets(llm_bullets)
        self._cache_response(cache_key, unique_bullets[:4])  # Cache LLM bullets
        return unique_bullets[:4]
    else:
        logger.warning(f"LLM approach ({self.provider}) failed - falling back to lightweight NLP")
```

---

## STEP 6: Update _create_llm_only_bullets (Line 2813)

**FIND LINE 2868-2875 (API CALL SECTION):**
```python
# STEP 3: Generate initial bullets
message = self._call_claude_with_retry(
    model="claude-3-5-sonnet-20241022",
    max_tokens=max_tokens,
    temperature=temperature,
    messages=[
        {"role": "user", "content": prompt}
    ]
)
```

**REPLACE WITH:**
```python
# STEP 3: Generate initial bullets
message = self._call_api_with_retry(
    provider=self.provider,
    model=self.models.get(self.provider, "claude-3-5-sonnet-20241022"),
    max_tokens=max_tokens,
    temperature=temperature,
    messages=[
        {"role": "user", "content": prompt}
    ]
)
```

**FIND LINE 2877-2887 (RESPONSE PARSING):**
```python
content = message.content[0].text.strip()

# STEP 4: Parse bullets from response
bullets = []
for line in content.split('\n'):
    line = line.strip()
    if line and len(line) > 15:
        # Clean up any formatting
        line = line.lstrip('•-*123456789. ')
        if line and not line.startswith('(') and len(line) > 15:
            bullets.append(line)
```

**REPLACE WITH:**
```python
# STEP 3.5: Extract text from provider-specific response format
content = self._extract_text_from_response(message, self.provider)
if not content:
    logger.error(f"Failed to extract content from {self.provider} response")
    return []

# STEP 4: Parse bullets from response
bullets = []
for line in content.split('\n'):
    line = line.strip()
    if line and len(line) > 15:
        # Clean up any formatting
        line = line.lstrip('•-*123456789. ')
        if line and not line.startswith('(') and len(line) > 15:
            bullets.append(line)
```

---

## STEP 7: Update _refine_bullets (Line 2740)

**FIND LINE 2782-2789 (API CALL SECTION):**
```python
message = self._call_claude_with_retry(
    model="claude-3-5-sonnet-20241022",
    max_tokens=400,
    temperature=0.1,  # Lower temperature for refinement
    messages=[
        {"role": "user", "content": refinement_prompt}
    ]
)
```

**REPLACE WITH:**
```python
message = self._call_api_with_retry(
    provider=self.provider,
    model=self.models.get(self.provider, "claude-3-5-sonnet-20241022"),
    max_tokens=400,
    temperature=0.1,  # Lower temperature for refinement
    messages=[
        {"role": "user", "content": refinement_prompt}
    ]
)
```

**FIND LINE 2791 (RESPONSE PARSING):**
```python
refined_text = message.content[0].text.strip()
```

**REPLACE WITH:**
```python
refined_text = self._extract_text_from_response(message, self.provider)
if not refined_text:
    logger.warning(f"Failed to extract refinement from {self.provider}, keeping original")
    return bullets
```

---

## STEP 8: Add New Method _create_openai_bullets

**ADD AFTER _create_llm_only_bullets (after line 2900), around line 2901:**

```python
def _create_openai_bullets(self, text: str, context_heading: str = None,
                          style: str = 'professional', enable_refinement: bool = False) -> List[str]:
    """
    Create bullets using OpenAI with structured, adaptive prompts.

    Args:
        text: Content to summarize
        context_heading: Optional heading for contextual awareness
        style: 'professional', 'educational', 'technical', or 'executive'
        enable_refinement: If True, run second pass for quality improvement

    Returns:
        List of bullet points
    """
    if not self.client or self.provider != 'openai':
        return []

    try:
        # STEP 1: Detect content type for adaptive strategy
        content_info = self._detect_content_type(text)
        logger.info(f"OpenAI bullet generation: {content_info['type']} content, {content_info['word_count']} words")

        # STEP 2: Build structured prompt based on content type and style
        prompt = self._build_structured_prompt(
            text,
            content_info,
            context_heading=context_heading,
            style=style
        )

        # Adaptive temperature based on content type and style
        if content_info['type'] == 'table' or style == 'technical':
            temperature = 0.2  # More deterministic for technical content
        elif style == 'educational' or style == 'executive':
            temperature = 0.4  # Slightly more creative for educational/exec content
        else:
            temperature = 0.3  # Balanced default

        # Dynamic max_tokens based on content length
        char_count = len(text)
        if char_count < 200:
            max_tokens = 400
        elif char_count < 600:
            max_tokens = 600
        else:
            max_tokens = 800

        logger.info(f"OpenAI params: temperature={temperature}, max_tokens={max_tokens}")

        # STEP 3: Generate initial bullets with OpenAI-specific parameter format
        message = self._call_api_with_retry(
            provider='openai',
            model=self.models.get('openai', 'gpt-3.5-turbo'),
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # STEP 3.5: Extract text from OpenAI response
        content = self._extract_text_from_response(message, 'openai')
        if not content:
            logger.error("Failed to extract content from OpenAI response")
            return []

        # STEP 4: Parse bullets from response (same as Claude method)
        bullets = []
        for line in content.split('\n'):
            line = line.strip()
            if line and len(line) > 15:
                # Clean up any formatting
                line = line.lstrip('•-*123456789. ')
                if line and not line.startswith('(') and len(line) > 15:
                    bullets.append(line)

        logger.info(f"OpenAI generated {len(bullets)} bullets (type: {content_info['type']}, style: {style})")

        # STEP 5: Optional refinement pass for quality improvement
        if enable_refinement and bullets:
            logger.info("Running refinement pass (OpenAI)...")
            bullets = self._refine_bullets(bullets, text)

        return bullets

    except Exception as e:
        logger.error(f"Error in OpenAI bullet generation: {e}")
        return []
```

---

## SUMMARY OF CHANGES

| File | Lines | Change | Impact |
|------|-------|--------|--------|
| document_parser.py | 80-81 | Add OpenAI import | Add new SDK |
| document_parser.py | 92-124 | Update __init__ | Support both providers |
| document_parser.py | 125-155 | Add 3 new methods | Helper methods for routing |
| document_parser.py | 180-238 | Refactor retry logic | Provider-agnostic API calls |
| document_parser.py | 1958-1972 | Add routing logic | Route to correct provider |
| document_parser.py | 2868-2887 | Update API call | Use new retry method |
| document_parser.py | 2782-2791 | Update refinement | Use new retry method |
| document_parser.py | 2901-2980 | Add new method | OpenAI-specific bullets |

**Total new/modified lines: ~200 lines**
**Risk level: MEDIUM (well-isolated changes, backward compatible)**
**Testing effort: MEDIUM (need to test both provider paths)**

---

## VALIDATION CHECKLIST

Before submitting PR:
- [ ] Both providers initialize correctly
- [ ] Cache works with both providers
- [ ] Retry logic handles OpenAI errors
- [ ] Response parsing works for both
- [ ] Fallback chain still works if API fails
- [ ] Backward compatible with Anthropic-only code
- [ ] Added unit tests for provider detection
- [ ] Added integration tests for bullet generation
- [ ] Documented API key environment variables
- [ ] Documented cost differences between providers

