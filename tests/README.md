# Slide Generator Testing Suite

Comprehensive quality testing infrastructure for bullet generation.

## 🚀 Quick Start

```bash
# Quick validation (30 seconds)
python tests/smoke_test.py

# Establish baseline
python tests/regression_benchmark.py --version v87_baseline

# After making changes
python tests/smoke_test.py                                    # Quick check
python tests/regression_benchmark.py --version v88_changes    # Full test
python tests/regression_benchmark.py --compare v87_baseline v88_changes  # Compare
```

## 📁 Files

| File | Purpose | Usage |
|------|---------|-------|
| `smoke_test.py` | Quick validation (4 tests) | Run before every deploy |
| `regression_benchmark.py` | Full quality testing (11 tests) | Track quality across versions |
| `golden_test_set.py` | Hand-crafted test cases | Reference data for tests |
| `quality_metrics.py` | Objective scoring system | Evaluates bullet quality 0-100 |

## 🧪 Test Suites

### 1. Smoke Test (`smoke_test.py`)

**Purpose:** Fast sanity check before deployment

**Test Cases:**
- Educational paragraph (ML course)
- Technical content (microservices)
- Table structure (pricing)
- Executive metrics (transformation results)

**Usage:**
```bash
python tests/smoke_test.py

# Output:
# ✅ All smoke tests passed - Safe to deploy
# (exit code 0)
#
# OR
#
# ❌ 2 test(s) failed - DO NOT DEPLOY
# (exit code 1)
```

**When to Run:**
- ✅ Before every `git push heroku main`
- ✅ After any code change
- ✅ As part of CI/CD pipeline

---

### 2. Regression Benchmark (`regression_benchmark.py`)

**Purpose:** Track quality across versions and detect regressions

**Features:**
- 11 comprehensive test cases
- Objective quality scoring (0-100)
- Version comparison reports
- Automatic regression detection

**Usage:**
```bash
# Run benchmark for current version
python tests/regression_benchmark.py --version v88

# Compare two versions
python tests/regression_benchmark.py --compare v87 v88

# List stored results
python tests/regression_benchmark.py --list

# With Claude API key (tests LLM mode)
python tests/regression_benchmark.py --version v88 --api-key sk-ant-...
```

**Example Output:**
```
BENCHMARK SUMMARY: v88
======================================================================
Tests Passed: 10/11 (90.9%)

Average Scores:
  Overall Quality: 84.2/100
  Structure:       82.5/100
  Relevance:       85.8/100
  Style:           83.1/100
  Readability:     86.4/100

❌ Failing Tests (1):
  - edu_ml_basics: 68.3/100 (below threshold: 70.0)
```

---

### 3. Golden Test Set (`golden_test_set.py`)

**Purpose:** Hand-crafted reference data with expected outputs

**11 Test Cases:**

| ID | Category | Description |
|----|----------|-------------|
| `edu_ml_basics` | Educational | ML course description |
| `tech_microservices` | Technical | Architecture overview |
| `exec_digital_transform` | Executive | Transformation metrics |
| `pro_cloud_benefits` | Professional | Cloud advantages |
| `table_feature_comparison` | Table | Pricing plan table |
| `list_consolidation` | List | Bullet list synthesis |
| `heading_expansion` | Heading | Expand short heading |
| `edge_very_short` | Edge case | Minimal input text |
| `edge_very_long` | Edge case | Long complex paragraph |
| `mixed_paragraph_with_metrics` | Mixed | Technical + metrics |
| `pro_ai_ethics` | Professional | AI ethics discussion |

**Adding New Test Cases:**
```python
# tests/golden_test_set.py

GOLDEN_TEST_SET.append({
    "id": "your_test_id",
    "category": "professional",  # educational, technical, executive, professional
    "input_text": """Your test input text...""",
    "context_heading": "Test Heading",
    "expected_style": "professional",
    "expected_bullets": [
        "Expected bullet 1",
        "Expected bullet 2"
    ],
    "quality_criteria": {
        "min_bullets": 3,
        "max_bullets": 5,
        "avg_word_length": (8, 15),
        "must_contain_keywords": ["keyword1", "keyword2"]
    }
})
```

---

### 4. Quality Metrics (`quality_metrics.py`)

**Purpose:** Objective scoring system for bullet points

**Evaluated Dimensions:**

1. **Structure Score (0-100)**
   - Bullet count (3-5 optimal)
   - Word length consistency (8-15 words)
   - Parallel formatting
   - Proper punctuation

2. **Relevance Score (0-100)**
   - Keyword coverage from input
   - Context alignment with heading
   - Avoids generic filler

3. **Style Score (0-100)**
   - Tone appropriateness (educational/technical/executive/professional)
   - Active vs passive voice
   - Consistent verb tense
   - Style-specific keywords

4. **Readability Score (0-100)**
   - Flesch Reading Ease (60-80 ideal)
   - Sentence complexity
   - Word syllable count
   - Clarity

5. **Overall Quality (0-100)**
   - Weighted composite: `(structure * 0.25) + (relevance * 0.35) + (style * 0.20) + (readability * 0.20)`

**Usage:**
```python
from tests.quality_metrics import BulletQualityMetrics, format_metrics_report
from tests.golden_test_set import GOLDEN_TEST_SET

evaluator = BulletQualityMetrics()
test_case = GOLDEN_TEST_SET[0]

bullets = ["Generated bullet 1", "Generated bullet 2", "Generated bullet 3"]
metrics = evaluator.evaluate(bullets, test_case)

print(format_metrics_report(metrics, test_id="edu_ml_basics"))

# Output:
# Test: edu_ml_basics
# ============================================================
# Overall Quality: 84.2/100
#   - Structure:    82.5/100
#   - Relevance:    85.8/100
#   - Style:        83.1/100
#   - Readability:  86.4/100
#
# Bullets: 3
# Avg Length: 9.7 words (58.3 chars)
```

---

## 📊 Quality Thresholds

Tests enforce minimum quality standards:

```python
QUALITY_THRESHOLDS = {
    "overall_quality": 70.0,    # Must be above 70/100
    "structure_score": 65.0,    # Must be above 65/100
    "relevance_score": 70.0,    # Must be above 70/100
    "style_score": 60.0,        # Must be above 60/100
    "readability_score": 65.0   # Must be above 65/100
}
```

**If any score falls below threshold → Test FAILS**

---

## 🔄 Typical Workflow

### Scenario 1: Bug Fix

```bash
# 1. Quick validation
python tests/smoke_test.py

# If passes, deploy
git commit -m "Fix: ..."
git push heroku main
```

### Scenario 2: Minor Enhancement

```bash
# 1. Quick validation
python tests/smoke_test.py

# 2. Run full benchmark
python tests/regression_benchmark.py --version v88_fix

# 3. If quality maintained, deploy
git push heroku main
```

### Scenario 3: Major Feature

```bash
# 1. Establish baseline BEFORE changes
python tests/regression_benchmark.py --version v87_baseline

# 2. Make code changes
# ... edit files ...

# 3. Quick validation
python tests/smoke_test.py

# 4. Full benchmark
python tests/regression_benchmark.py --version v88_new_feature

# 5. Compare versions
python tests/regression_benchmark.py --compare v87_baseline v88_new_feature

# 6. Review comparison report
# - Check for regressions
# - Verify improvements
# - Assess trade-offs

# 7. If metrics improve, deploy
git push heroku main
```

---

## 📈 Interpreting Results

### Smoke Test

```
✅ All smoke tests passed - Safe to deploy
```
→ **Action:** Proceed with deployment

```
❌ 2 test(s) failed - DO NOT DEPLOY
  └─ Too few bullets: 1 < 2
  └─ Missing keyword: 'machine learning'
```
→ **Action:** Fix issues before deploying

### Regression Benchmark

```
Overall Quality: 82.4 → 85.1 (+2.7) ✅
```
→ **Action:** Quality improved, safe to deploy

```
Overall Quality: 82.4 → 79.8 (-2.6) ❌
```
→ **Action:** Regression detected, investigate

```
⚠️ REGRESSIONS DETECTED (3 tests):
  - edu_ml_basics: 79.2 → 73.5 (-5.7)
  - tech_microservices: 82.1 → 78.4 (-3.7)
```
→ **Action:** Critical regression, DO NOT DEPLOY

---

## 🛠️ Troubleshooting

### "All advanced approaches failed"

**Problem:** No API key provided, NLP fallback is failing

**Solution:**
```bash
# Test with API key
export CLAUDE_API_KEY='sk-ant-...'
python tests/regression_benchmark.py --version test --api-key $CLAUDE_API_KEY
```

### "ModuleNotFoundError: tests"

**Problem:** Python can't find tests module

**Solution:**
```bash
# Run from project root
cd /path/to/slide_generator
python tests/smoke_test.py
```

### "No benchmark results found"

**Problem:** Trying to compare versions that haven't been run

**Solution:**
```bash
# List available versions
python tests/regression_benchmark.py --list

# Run missing version
python tests/regression_benchmark.py --version v87
```

---

## 📝 Best Practices

### ✅ DO:
- Run smoke test before every deployment
- Establish baseline before making changes
- Compare versions after major changes
- Add test cases for new features
- Keep quality thresholds consistent

### ❌ DON'T:
- Skip testing for "small changes"
- Deploy with failing smoke tests
- Ignore regressions without investigation
- Lower quality thresholds to pass tests
- Manually test instead of automated tests

---

## 🎯 Success Metrics

Track these over time:

- **Test Pass Rate**: Target 100% (all tests passing)
- **Average Quality Score**: Target >80/100
- **Regression Rate**: Target <5% per version
- **Deployment Confidence**: High (data-driven decisions)
- **Testing Time**: <5 minutes (vs 30+ minutes manual)

---

## 🔮 Future Enhancements

Planned improvements:

1. **CI/CD Integration**
   - GitHub Actions workflow
   - Automatic testing on PR
   - Quality gates for merges

2. **A/B Testing Framework**
   - Route % of users to variants
   - Collect real user feedback
   - Statistical significance testing

3. **User Feedback Collection**
   - In-app rating widget
   - Track satisfaction scores
   - Correlate with quality metrics

4. **Performance Benchmarks**
   - Processing time tracking
   - API token usage monitoring
   - Cost per document analysis

5. **Visual Regression Testing**
   - Screenshot comparisons
   - Slide layout validation
   - Font/formatting checks

---

## 📚 Additional Resources

- **CLAUDE.md** - Testing protocol documented for future sessions
- **LLM_ENHANCEMENTS_2025.md** - LLM system documentation
- **NLP_APPROACH_DECISION.md** - NLP approach rationale

---

**Last Updated:** October 28, 2025
**Version:** 1.0.0
**Status:** ✅ Production Ready
