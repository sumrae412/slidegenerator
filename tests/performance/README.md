# Performance Tests for Slide Generator

Comprehensive performance and load testing suite for the Slide Generator application.

## Overview

The performance tests cover 15 different test scenarios across 6 categories:

1. **Large Document Tests** (3 tests)
2. **Cache Performance Tests** (3 tests)
3. **Concurrent Processing Tests** (2 tests)
4. **Memory Tests** (2 tests)
5. **Response Time Benchmarks** (3 tests)
6. **Stress & Throughput Tests** (2 tests)

## Installation

Before running the tests, install the required dependencies:

```bash
# Install pytest for running tests
pip install pytest

# Install psutil for memory monitoring (optional but recommended)
pip install psutil

# Or install all test dependencies
pip install pytest psutil
```

## Running the Tests

### Run all performance tests
```bash
pytest tests/performance/test_load.py -v
```

### Run only performance-marked tests
```bash
pytest tests/performance/test_load.py -v -m performance
```

### Run only slow tests (>1 second)
```bash
pytest tests/performance/test_load.py -v -m slow
```

### Run only memory tests
```bash
pytest tests/performance/test_load.py -v -m memory
```

### Run only concurrent tests
```bash
pytest tests/performance/test_load.py -v -m concurrent
```

### Run a specific test
```bash
pytest tests/performance/test_load.py::test_cache_hit_performance -v
```

### Run with detailed output
```bash
pytest tests/performance/test_load.py -v -s
```

### Run with performance timing
```bash
pytest tests/performance/test_load.py -v --durations=10
```

## Test Categories

### 1. Large Document Tests

Test the system's ability to handle large-scale content:

- **test_process_large_document()** - Processes a 100+ paragraph document (~12,000 words)
  - Expects: < 30 seconds processing time
  - Validates: Correct bullet generation at scale

- **test_process_very_long_paragraphs()** - Processes 1000+ word paragraphs
  - Expects: < 20 seconds for 3,600-word document
  - Validates: Can handle unusually long paragraphs

- **test_many_bullet_points()** - Generates 50+ bullet points
  - Expects: < 15 seconds, generates 50+ bullets
  - Validates: Consistent quality with high volume

### 2. Cache Performance Tests

Verify caching system effectiveness:

- **test_cache_hit_performance()** - Measures cache hit speed
  - Expects: Cache hits are significantly faster than misses (10x+)
  - Validates: Cache key generation works correctly

- **test_cache_miss_performance()** - Tests first-time processing
  - Expects: < 5 seconds average for cache misses
  - Validates: Acceptable performance for new content

- **test_cache_effectiveness()** - Measures cache hit rate
  - Expects: > 90% hit rate with repeated content
  - Validates: Cache properly stores and retrieves content

### 3. Concurrent Processing Tests

Test behavior under concurrent load:

- **test_concurrent_requests()** - Processes 10 documents in parallel
  - Expects: Completes in < 60 seconds with parallel speedup
  - Validates: Thread-safe document processing
  - Method: Uses ThreadPoolExecutor with 5 workers

- **test_concurrent_bullet_generation()** - Generates bullets for 8 sections in parallel
  - Expects: < 30 seconds, proper parallel execution
  - Validates: Concurrent cache operations work correctly
  - Method: Uses ThreadPoolExecutor with 4 workers

### 4. Memory Tests

Monitor memory usage and detect leaks:

- **test_memory_usage_large_doc()** - Memory consumption with large documents
  - Expects: Memory increase < 500MB for large document
  - Validates: No excessive memory allocation

- **test_no_memory_leaks()** - Detects memory leaks over multiple iterations
  - Expects: Consistent memory usage, avg growth < 50MB/iteration
  - Validates: Objects are properly garbage collected
  - Method: Processes 5 documents sequentially with gc.collect()

### 5. Response Time Benchmarks

Track response times for different document sizes:

- **test_simple_doc_response_time()** - Small document performance
  - Expects: < 5 seconds
  - Document: ~20 words

- **test_medium_doc_response_time()** - Medium document performance
  - Expects: < 15 seconds
  - Document: 50 paragraphs (~5,000 words)

- **test_bullet_generation_time()** - Comprehensive timing analysis
  - Expects: Short < 2s, Medium < 8s, Long < 15s
  - Validates: Performance scales appropriately
  - Runs 3 iterations per size for consistency data

### 6. Stress & Throughput Tests

Test system limits and throughput:

- **test_sequential_processing_throughput()** - Document processing throughput
  - Expects: > 0.1 documents per second
  - Validates: Acceptable sequential processing speed

- **test_cache_size_management()** - Cache growth control
  - Expects: Cache doesn't grow unbounded
  - Validates: Cache implements size management

## Test Markers

Tests are marked with pytest markers for selective execution:

| Marker | Purpose | Count |
|--------|---------|-------|
| `@performance` | General performance benchmark | 13 |
| `@slow` | Takes > 1 second | 7 |
| `@memory` | Memory profiling tests | 2 |
| `@concurrent` | Concurrency tests | 2 |

## Performance Thresholds

Key performance targets:

| Metric | Target | Purpose |
|--------|--------|---------|
| Simple document | < 5 seconds | Basic responsiveness |
| Medium document | < 15 seconds | Acceptable UX wait time |
| Large document | < 30 seconds | Graceful degradation |
| Cache hit | < 100ms | Must be significantly faster |
| Concurrent speedup | > 1.5x | Parallelization benefit |
| Memory growth | < 50MB/iteration | No memory leaks |
| Hit rate | > 90% | Cache effectiveness |

## Output Example

```
tests/performance/test_load.py::test_cache_hit_performance PASSED      [ 0%]
[CACHE HIT TEST] Measuring cache hit performance...
  ✓ Cache miss: 0.1234s
  ✓ Cache hit: 0.0045s
  ✓ Speedup: 27.4x faster
  ✓ Cache stats: {'hits': 1, 'misses': 1, 'size': 2}

tests/performance/test_load.py::test_concurrent_requests PASSED        [ 5%]
[CONCURRENT REQUESTS TEST] Processing 10 concurrent requests...
  ✓ Processed 10 documents concurrently
  ✓ Total time: 12.34s
  ✓ Average per document: 0.85s
  ✓ Speedup: 2.1x (parallel vs sequential)
```

## Integration with CI/CD

For GitHub Actions or similar CI/CD:

```yaml
- name: Run performance tests
  run: |
    pip install pytest psutil
    pytest tests/performance/test_load.py -v --tb=short

- name: Run only fast tests
  run: |
    pytest tests/performance/test_load.py -v -m "not slow"

- name: Check memory tests
  run: |
    pytest tests/performance/test_load.py -v -m memory
```

## Dependencies

Required:
- `pytest` - Test framework
- `psutil` - Memory monitoring (optional but recommended for memory tests)

## Notes

- Memory tests are more accurate on Linux systems where `psutil` is available
- Concurrent tests use ThreadPoolExecutor; timing may vary based on system load
- Cache hit rates depend on DocumentParser implementation
- All tests use NLP fallback (no Claude API key) for consistency
- Tests are isolated and don't interfere with each other

## Troubleshooting

### Tests fail with "ModuleNotFoundError: No module named 'pytest'"
```bash
pip install pytest
```

### Memory tests show 0 memory usage
```bash
pip install psutil
```

### Tests are very slow
- Reduce document sizes in test generators
- Run with `-m "not slow"` to skip slow tests
- Run individual tests with `-k test_name`

### Concurrent tests fail with thread errors
- Ensure DocumentParser is thread-safe
- Check for global state issues
- Increase thread pool workers if needed

## Contributing

When adding new performance tests:

1. Add the test function to appropriate category
2. Include appropriate pytest markers: `@pytest_mark_performance`, `@pytest_mark_slow`, etc.
3. Document expected performance targets
4. Add to this README with description
5. Ensure test is isolated (no dependencies on other tests)
6. Print progress with formatted output: `print(f"  ✓ Message")`

## Related Documentation

- Main testing guide: `../README_TESTING.md`
- Quality metrics: `../quality_metrics.py`
- Regression testing: `../regression_benchmark.py`
- Smoke tests: `../smoke_test.py`
