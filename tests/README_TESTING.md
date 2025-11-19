# OpenAI Integration Testing Guide

Comprehensive testing infrastructure for OpenAI integration and new features.

## Quick Start

### Run All Tests (Mock APIs - Fast)
```bash
./tests/run_all_tests.sh
```

### Run All Tests (Real APIs - Requires API Keys)
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
./tests/run_all_tests.sh --real
```

### Generate Coverage Report
```bash
./tests/run_all_tests.sh --coverage
# Open htmlcov/index.html to view report
```

## Test Suites

### 1. OpenAI Integration Tests (`test_openai_integration.py`)

Tests OpenAI API functionality, JSON mode, function calling, and embeddings.

**Run:**
```bash
pytest tests/test_openai_integration.py -v
```

**What's Tested:**
- ✅ OpenAI library import and client creation
- ✅ JSON mode responses (structured output)
- ✅ Function calling with categorized bullets
- ✅ API retry logic (rate limits, timeouts)
- ✅ Embedding generation for deduplication
- ✅ Cosine similarity calculations
- ✅ Deduplication threshold logic
- ✅ GPT-3.5-turbo integration
- ✅ Chain-of-thought prompting
- ✅ Batch processing
- ✅ Error handling (invalid keys, timeouts, malformed JSON)
- ✅ Cache hit/miss scenarios

**Key Test Classes:**
- `TestOpenAIBasics` - Basic functionality
- `TestOpenAIAPICallsMocked` - Mocked API calls
- `TestEmbeddingDeduplication` - Embedding-based dedup
- `TestGPT35TurboIntegration` - GPT-3.5 support
- `TestChainOfThoughtPrompting` - CoT prompting
- `TestBatchProcessing` - Batch operations
- `TestErrorHandling` - Error scenarios
- `TestCacheIntegration` - Cache behavior

### 2. Intelligent Routing Tests (`test_intelligent_routing.py`)

Tests automatic model selection based on content type and characteristics.

**Run:**
```bash
pytest tests/test_intelligent_routing.py -v
```

**What's Tested:**
- ✅ Content type detection (table, paragraph, mixed)
- ✅ Content length classification (short, medium, long)
- ✅ Style detection (technical, executive, educational)
- ✅ Routing decisions (table→OpenAI, long→Claude)
- ✅ Fallback behavior when model unavailable
- ✅ Routing performance (<10ms decisions)
- ✅ Priority ordering of routing rules
- ✅ Consistency for same content

**Routing Rules Tested:**
| Content Type | Preferred Model | Reason |
|--------------|----------------|---------|
| Tables | OpenAI | Better structured data |
| Short (<100 words) | OpenAI | Faster |
| Long (>500 words) | Claude | Better context |
| Technical | OpenAI | Precise terminology |
| Executive (metrics) | OpenAI | Good with numbers |

**Key Test Classes:**
- `TestContentTypeDetection` - Identify content types
- `TestContentLengthClassification` - Classify by length
- `TestStyleDetection` - Detect writing style
- `TestRoutingDecisions` - Routing logic
- `TestFallbackBehavior` - Model unavailability
- `TestRoutingPriorities` - Rule priority ordering

### 3. Cost Tracking Tests (`test_cost_tracking.py`)

Tests API cost calculations, token estimation, and cost optimization.

**Run:**
```bash
pytest tests/test_cost_tracking.py -v
```

**What's Tested:**
- ✅ Token estimation (1 token ≈ 4 chars)
- ✅ Cost calculation for Claude (Sonnet, Opus)
- ✅ Cost calculation for OpenAI (GPT-4o, GPT-3.5)
- ✅ Embedding cost calculation
- ✅ Document-level cost estimation (10-page, 50-page)
- ✅ Cache savings calculation (40-60% reduction)
- ✅ Cost comparison between models
- ✅ Per-slide cost tracking
- ✅ Cost optimization strategies
- ✅ Cost reporting and summaries

**Pricing (as of 2025):**
- Claude Sonnet: $3/$15 per million tokens (input/output)
- OpenAI GPT-4o: $5/$15 per million tokens
- OpenAI GPT-3.5: $0.50/$1.50 per million tokens
- Embeddings: $0.02 per million tokens

**Key Test Classes:**
- `TestTokenEstimation` - Token counting
- `TestCostCalculation` - Price calculations
- `TestDocumentCostEstimation` - Full doc estimates
- `TestCacheSavings` - Cache ROI
- `TestCostComparison` - Model comparison
- `TestCostOptimization` - Optimization strategies

### 4. Ensemble Mode Tests (`test_ensemble_mode.py`)

Tests combining outputs from multiple models for highest quality.

**Run:**
```bash
pytest tests/test_ensemble_mode.py -v
```

**What's Tested:**
- ✅ Combining outputs from multiple models
- ✅ Semantic deduplication of similar bullets
- ✅ Majority voting for bullet selection
- ✅ Weighted voting based on model quality
- ✅ Ensemble quality improvement
- ✅ Union/intersection strategies
- ✅ Best-of-N selection
- ✅ Performance characteristics (latency, cost)
- ✅ Error handling (partial/complete failures)
- ✅ Configuration options
- ✅ Diversity and consensus metrics

**Ensemble Strategies Tested:**
- **Union**: All unique bullets from all models
- **Intersection**: Only bullets agreed upon by all
- **Majority**: Bullets selected by ≥50% of models
- **Weighted**: Bullets scored by model quality weights
- **Best-of-N**: Top bullet from each model

**Key Test Classes:**
- `TestEnsembleConcepts` - Basic concepts
- `TestEnsembleVoting` - Voting mechanisms
- `TestEnsembleQuality` - Quality improvement
- `TestEnsembleStrategies` - Different strategies
- `TestEnsemblePerformance` - Performance traits

### 5. Performance Tests (`test_performance.py`)

Tests response times, throughput, scalability, and optimization.

**Run:**
```bash
pytest tests/test_performance.py -v
```

**What's Tested:**
- ✅ API response times (Claude, OpenAI)
- ✅ Cache lookup performance (<1ms)
- ✅ Routing decision speed (<1ms)
- ✅ Multi-slide throughput
- ✅ Batch processing efficiency
- ✅ Linear scaling with document size
- ✅ Cache scalability improvements
- ✅ Parallel vs sequential execution
- ✅ Memory usage (bounded cache)
- ✅ Concurrency safety
- ✅ Benchmarks (NLP vs LLM, GPT-3.5 vs GPT-4)

**Performance Targets:**
- Routing decision: <1ms
- Cache lookup: <1ms (sub-millisecond)
- Cache size: ≤1000 entries (bounded)
- Throughput: >50 slides/second (mocked)
- P95 latency: <10ms (excluding API calls)

**Key Test Classes:**
- `TestResponseTimes` - API latency
- `TestThroughput` - Processing speed
- `TestScalability` - Scaling behavior
- `TestOptimization` - Performance optimizations
- `TestMemoryUsage` - Memory efficiency
- `TestBenchmarks` - Comparative benchmarks

### 6. UI Integration Tests (`test_ui_integration.py`)

Tests web interface parameter handling and user interactions.

**Run:**
```bash
pytest tests/test_ui_integration.py -v
```

**What's Tested:**
- ✅ API key format validation (Claude: `sk-ant-`, OpenAI: `sk-`)
- ✅ API key masking for display
- ✅ Form parameter parsing (script_column, skip_visuals, output_format)
- ✅ Google Docs URL extraction
- ✅ Error message formatting
- ✅ Progress indicators and ETA
- ✅ Output format responses (PPTX, Google Slides)
- ✅ Cost estimation display
- ✅ Model selection UI
- ✅ Client-side validation
- ✅ OAuth authentication flow
- ✅ Session management
- ✅ Responsive UI behavior

**Key Test Classes:**
- `TestAPIKeyHandling` - API key management
- `TestFormParameterHandling` - Form inputs
- `TestDocumentURLHandling` - URL parsing
- `TestErrorMessageFormatting` - Error display
- `TestOutputOptions` - Download/Slides responses
- `TestCostEstimationUI` - Cost display
- `TestModelSelectionUI` - Model dropdown

### 7. Full Integration Tests (`integration_test_openai.py`)

End-to-end tests with sample documents and real/mock API calls.

**Run (Mock - Fast):**
```bash
python tests/integration_test_openai.py --mock
```

**Run (Real APIs - Requires Keys):**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
python tests/integration_test_openai.py --real
```

**Run Specific Document:**
```bash
python tests/integration_test_openai.py --mock --doc-type technical
```

**Compare Models:**
```bash
python tests/integration_test_openai.py --mock --compare
```

**Sample Documents Tested:**
- **Technical**: Microservices architecture (tables + paragraphs)
- **Educational**: Machine learning course (bullet lists)
- **Executive**: Q3 results with metrics (percentages, tables)
- **Mixed**: Cloud computing (headings, tables, lists)

**What's Tested:**
- ✅ Full document parsing and processing
- ✅ Slide generation from real content
- ✅ Bullet point quality for different content types
- ✅ Model comparison (Claude vs OpenAI vs Auto)
- ✅ Cost estimation for full documents
- ✅ Cache effectiveness across documents
- ✅ End-to-end processing time
- ✅ Error handling for various document structures

**Output:**
```
Testing TECHNICAL Document with model=auto
✅ Document: technical
   Model: auto
   Slides: 5
   Time: 0.12s
   Cache: 0 hits, 5 misses (0.0%)

   Sample slides:
   1. Microservices Architecture (3 bullets)
   2. Overview (3 bullets)
   3. Key Features (3 bullets)
```

## Test Coverage

### Coverage Goals
- **OpenAI Integration**: >80% code coverage
- **Routing Logic**: >90% code coverage
- **Cost Tracking**: >85% code coverage
- **UI Integration**: >75% code coverage

### Generate Coverage Report
```bash
./tests/run_all_tests.sh --coverage
open htmlcov/index.html  # View in browser
```

### Coverage by Component
```bash
# Specific module coverage
pytest tests/test_openai_integration.py --cov=file_to_slides --cov-report=term-missing
```

## Running Specific Tests

### Run Single Test File
```bash
pytest tests/test_openai_integration.py -v
```

### Run Single Test Class
```bash
pytest tests/test_openai_integration.py::TestOpenAIBasics -v
```

### Run Single Test Method
```bash
pytest tests/test_openai_integration.py::TestOpenAIBasics::test_openai_import -v
```

### Run Tests Matching Pattern
```bash
pytest tests/ -k "embedding" -v
pytest tests/ -k "cost" -v
pytest tests/ -k "routing" -v
```

### Run Tests with Markers
```bash
# Skip slow tests
pytest tests/ -m "not slow"

# Run only integration tests
pytest tests/ -m "integration"
```

## Continuous Integration

### Pre-Deployment Checklist
```bash
# 1. Run smoke tests (30 seconds)
python tests/smoke_test.py

# 2. Run OpenAI integration tests (2-3 minutes)
./tests/run_all_tests.sh

# 3. Run regression benchmark (5-10 minutes)
python tests/regression_benchmark.py --version v_current

# 4. If all pass, deploy
git push heroku main
```

### GitHub Actions (Planned)
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: ./tests/run_all_tests.sh
      - name: Generate coverage
        run: ./tests/run_all_tests.sh --coverage
```

## Test Data

### Mock Responses
All tests use mocked API responses by default (fast, no API costs).

**Example Mock:**
```python
mock_response = Mock()
mock_response.status_code = 200
mock_response.json.return_value = {
    "choices": [{
        "message": {
            "content": "- Bullet 1\n- Bullet 2\n- Bullet 3"
        }
    }]
}
```

### Real API Testing
Use `--real` flag to test with actual API calls (requires API keys).

**Warning**: Real API tests consume API credits!

```bash
# Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# Run with real APIs
./tests/run_all_tests.sh --real
```

## Troubleshooting

### "OpenAI library not installed"
```bash
pip install openai==1.13.0
```

### "pytest not found"
```bash
pip install pytest pytest-cov
```

### "No API keys found"
```bash
# For real API tests
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# Or use mock APIs (default)
./tests/run_all_tests.sh  # No keys needed
```

### Tests failing with "ModuleNotFoundError"
```bash
# Run from project root
cd /home/user/slidegenerator
./tests/run_all_tests.sh
```

### Coverage report not generating
```bash
pip install pytest-cov
./tests/run_all_tests.sh --coverage
```

## Best Practices

### Before Committing
1. Run smoke tests: `python tests/smoke_test.py`
2. Run OpenAI tests: `./tests/run_all_tests.sh`
3. Check coverage: `./tests/run_all_tests.sh --coverage`

### Before Deploying
1. Run full test suite: `./tests/run_all_tests.sh`
2. Run regression benchmark: `python tests/regression_benchmark.py`
3. Review test results and fix any failures

### When Adding New Features
1. Write tests first (TDD approach)
2. Ensure >80% coverage for new code
3. Add integration test if needed
4. Update this documentation

## Test Structure

```
tests/
├── test_openai_integration.py     # OpenAI API tests
├── test_intelligent_routing.py    # Routing logic tests
├── test_cost_tracking.py          # Cost calculation tests
├── test_ensemble_mode.py          # Ensemble feature tests
├── test_performance.py            # Performance benchmarks
├── test_ui_integration.py         # UI parameter tests
├── integration_test_openai.py     # Full integration tests
├── run_all_tests.sh               # Test runner script
├── README_TESTING.md              # This file
├── smoke_test.py                  # Quick validation (existing)
├── regression_benchmark.py        # Regression tests (existing)
├── golden_test_set.py             # Test cases (existing)
└── quality_metrics.py             # Quality scoring (existing)
```

## Contributing

### Adding New Tests
1. Create test file: `tests/test_new_feature.py`
2. Follow existing patterns (see `test_openai_integration.py`)
3. Add to `run_all_tests.sh`
4. Update this README

### Test Naming Conventions
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Mock fixtures: `mock_*`

### Documentation
- Update this README when adding new tests
- Include usage examples
- Document any special requirements

## Resources

- **Pytest Documentation**: https://docs.pytest.org/
- **unittest.mock Guide**: https://docs.python.org/3/library/unittest.mock.html
- **Coverage.py**: https://coverage.readthedocs.io/
- **OpenAI API Docs**: https://platform.openai.com/docs
- **Claude API Docs**: https://docs.anthropic.com

## Support

For issues or questions:
1. Check existing test files for examples
2. Review test output for error details
3. Check `OPENAI_INTEGRATION.md` for API documentation
4. Review `CLAUDE.md` for testing guidelines

---

**Version**: 1.0
**Last Updated**: 2025-11-19
**Maintainer**: Claude Code Assistant
