# Error Handling Tests - Integration Test Suite

## Overview
Comprehensive error handling test suite for the Slide Generator application with **31 tests** organized into **8 test classes**, providing **723 lines** of well-documented test code.

## Quick Start
```bash
# Run all tests
pytest tests/integration/test_error_handling.py -v

# Run specific test category
pytest tests/integration/test_error_handling.py::TestInvalidInputs -v

# Run with coverage report
pytest tests/integration/test_error_handling.py --cov=file_to_slides -v
```

## What's Included

### Main Test File
- **test_error_handling.py** (723 lines, 31 tests)
  - Complete error handling test implementation
  - 8 test classes covering different error scenarios
  - Uses mocking to avoid external dependencies
  - Thread-safe concurrent request testing
  - Edge case and boundary condition testing

### Documentation Files
1. **TEST_COVERAGE_SUMMARY.md** - Detailed test descriptions
2. **RUNNING_TESTS.md** - How to run tests with examples
3. **TEST_INVENTORY.md** - Complete test listing and statistics
4. **README.md** - This file

## Test Categories

### 1. Invalid Input Tests (4 tests)
Tests handling of malformed and invalid inputs:
- Malformed Google Docs URLs
- Empty document content
- Null/None input values
- Invalid file formats

### 2. Network Error Tests (4 tests)
Tests handling of network-related failures:
- Request timeouts
- Connection refused
- DNS resolution failures
- Connection reset

### 3. API Error Tests (5 tests)
Tests handling of API-specific errors:
- Claude API quota exceeded (429)
- Invalid API key (401)
- Rate limiting with retry mechanism
- Server errors (500)
- Generic API errors

### 4. Document Processing Error Tests (4 tests)
Tests handling of document parsing errors:
- Documents too large
- Malformed document structure
- Missing document content
- Corrupted document data

### 5. Concurrent Request Tests (3 tests)
Tests thread safety and parallel processing:
- Multiple concurrent document requests
- Cache thread safety
- Concurrent API calls

### 6. Edge Case Tests (6 tests)
Tests unusual inputs and boundary conditions:
- Extremely long text (50,000 words)
- Special characters and Unicode
- Very short documents (1 character)
- Whitespace-only content
- Extremely deep nesting (100 levels)
- Repeated identical content (1,000 repetitions)

### 7. Cache Error Tests (2 tests)
Tests caching mechanisms:
- Cache key generation with special inputs
- Cache overflow and LRU eviction

### 8. Flask App Error Handling Tests (3 tests)
Tests application-level error handling:
- Null file input handling
- Malformed JSON handling
- Error recovery

## Test Statistics

| Category | Tests | % | Lines |
|----------|-------|---|-------|
| Invalid Input | 4 | 13% | 80 |
| Network Errors | 4 | 13% | 70 |
| API Errors | 5 | 16% | 110 |
| Document Processing | 4 | 13% | 85 |
| Concurrent Requests | 3 | 10% | 95 |
| Edge Cases | 6 | 19% | 145 |
| Cache Errors | 2 | 6% | 45 |
| Flask App | 3 | 10% | 50 |
| **TOTAL** | **31** | **100%** | **723** |

## Key Features

✅ **Comprehensive Coverage**
- 31 tests covering 8 error categories
- Realistic failure scenarios
- Boundary condition testing

✅ **Isolated Testing**
- Uses mocking to avoid external dependencies
- No real API calls
- No real file operations

✅ **Thread Safety**
- Tests concurrent operations
- Validates race condition handling
- Cache thread safety verification

✅ **Well Documented**
- Clear docstrings on every test
- Expected behavior specified
- Mock patterns explained

✅ **CI/CD Ready**
- Easy integration into pipelines
- Clear exit codes
- Detailed error reporting

✅ **Performance Tested**
- Handles large inputs (100MB)
- Tests concurrent requests (5-10 threads)
- Cache performance validation

## Dependencies

**Built-in:**
- `pytest` - Test framework
- `unittest.mock` - Mocking
- `threading` - Concurrency testing
- `json`, `sys`, `os`, `time`

**Project:**
- `file_to_slides` - Main application
- `DocumentParser` - Core parsing class

## Running Tests

### Quick Commands
```bash
# All tests
pytest tests/integration/test_error_handling.py -v

# By category
pytest tests/integration/test_error_handling.py::TestInvalidInputs -v
pytest tests/integration/test_error_handling.py::TestNetworkErrors -v
pytest tests/integration/test_error_handling.py::TestAPIErrors -v

# With options
pytest tests/integration/test_error_handling.py -v --tb=short
pytest tests/integration/test_error_handling.py -v -s
pytest tests/integration/test_error_handling.py --cov=file_to_slides -v
```

### For More Options
See [RUNNING_TESTS.md](RUNNING_TESTS.md) for comprehensive testing commands.

## Test Coverage

### Error Types Tested
- **Input Validation:** Malformed URLs, empty content, null values, invalid formats
- **Network:** Timeouts, connection refused, DNS failures, connection reset
- **API:** Rate limiting, authentication, server errors, retry logic
- **Processing:** Large files, malformed structure, missing content, corrupted data
- **Concurrency:** Thread safety, race conditions, parallel requests
- **Edge Cases:** Long text, unicode, short text, special characters, deep nesting
- **Caching:** Key generation, overflow, LRU eviction
- **HTTP:** Error codes, JSON parsing, error recovery

### Mock Coverage
- `requests.get()` - HTTP requests
- `anthropic.Anthropic` - Claude API
- `time.sleep()` - Retry delays
- Custom error scenarios (429, 401, 500, etc.)

## Usage Examples

### Example 1: Quick Validation
```bash
pytest tests/integration/test_error_handling.py::TestInvalidInputs -v
```

### Example 2: Full Pre-Deploy Check
```bash
pytest tests/integration/test_error_handling.py -v --tb=short
```

### Example 3: Debug Specific Test
```bash
pytest tests/integration/test_error_handling.py::TestAPIErrors::test_claude_api_rate_limit_with_retry -vv -s
```

### Example 4: Performance Analysis
```bash
pytest tests/integration/test_error_handling.py -v --durations=10
```

## Test Design Principles

1. **Isolation** - Tests don't affect each other
2. **Clarity** - Test names describe what they test
3. **Realism** - Scenarios based on actual failure modes
4. **Completeness** - Both success and failure paths tested
5. **Performance** - Reasonable timeouts and resource usage
6. **Maintainability** - Clear structure and documentation

## Maintenance

### Adding New Tests
1. Choose appropriate test class or create new one
2. Follow naming convention: `test_<scenario>`
3. Add docstring explaining test
4. Use existing mock patterns
5. Update documentation

### Updating Tests
1. Modify test code
2. Run tests locally: `pytest -v`
3. Verify mock signatures match actual code
4. Update documentation if needed

## Integration with CI/CD

Tests are designed for easy CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run error handling tests
  run: |
    pytest tests/integration/test_error_handling.py -v \
      --tb=short \
      --junit-xml=test-results.xml
```

## Documentation Files

| File | Purpose |
|------|---------|
| test_error_handling.py | Main test implementation |
| TEST_COVERAGE_SUMMARY.md | Detailed test descriptions |
| RUNNING_TESTS.md | How to run tests |
| TEST_INVENTORY.md | Complete test listing |
| README.md | This file |

## Performance Characteristics

- **Fast:** Individual tests complete in < 1 second
- **Scalable:** 31 tests run in ~10-20 seconds
- **Isolated:** No database or API dependencies
- **Concurrent:** Tests can run in parallel

## Next Steps

1. **Review** - Read TEST_COVERAGE_SUMMARY.md for details
2. **Run** - Execute tests with RUNNING_TESTS.md commands
3. **Integrate** - Add to CI/CD pipeline
4. **Extend** - Add more tests as needed

## Support

For questions or issues:
1. Check TEST_COVERAGE_SUMMARY.md for test details
2. Review test docstrings for expected behavior
3. Use RUNNING_TESTS.md for debugging options
4. Check test code for mock patterns

---

**Last Updated:** 2025-11-19
**Test Count:** 31 tests
**Code Lines:** 723 lines
**Status:** Ready for deployment
