# Running Error Handling Tests

## Quick Start

```bash
# Run all error handling tests
pytest tests/integration/test_error_handling.py -v

# Run specific test class
pytest tests/integration/test_error_handling.py::TestInvalidInputs -v

# Run single test
pytest tests/integration/test_error_handling.py::TestInvalidInputs::test_null_input_values -v
```

## Available Test Commands

### Run All Tests
```bash
pytest tests/integration/test_error_handling.py -v
```

### Run by Category

**Invalid Input Tests (4 tests)**
```bash
pytest tests/integration/test_error_handling.py::TestInvalidInputs -v
```

**Network Error Tests (4 tests)**
```bash
pytest tests/integration/test_error_handling.py::TestNetworkErrors -v
```

**API Error Tests (5 tests)**
```bash
pytest tests/integration/test_error_handling.py::TestAPIErrors -v
```

**Document Processing Error Tests (4 tests)**
```bash
pytest tests/integration/test_error_handling.py::TestDocumentProcessingErrors -v
```

**Concurrent Request Tests (3 tests)**
```bash
pytest tests/integration/test_error_handling.py::TestConcurrentRequests -v
```

**Edge Case Tests (6 tests)**
```bash
pytest tests/integration/test_error_handling.py::TestEdgeCases -v
```

**Cache Error Tests (2 tests)**
```bash
pytest tests/integration/test_error_handling.py::TestCacheErrors -v
```

**Flask App Error Handling Tests (3 tests)**
```bash
pytest tests/integration/test_error_handling.py::TestFlaskAppErrorHandling -v
```

## Output Options

### Verbose Output
```bash
pytest tests/integration/test_error_handling.py -v
```

### Very Verbose (show all assertions)
```bash
pytest tests/integration/test_error_handling.py -vv
```

### Show Print Statements
```bash
pytest tests/integration/test_error_handling.py -v -s
```

### Short Traceback
```bash
pytest tests/integration/test_error_handling.py -v --tb=short
```

### Long Traceback (full details)
```bash
pytest tests/integration/test_error_handling.py -v --tb=long
```

### Line Traceback
```bash
pytest tests/integration/test_error_handling.py -v --tb=line
```

### No Traceback
```bash
pytest tests/integration/test_error_handling.py -v --tb=no
```

## Performance Monitoring

### Show test duration
```bash
pytest tests/integration/test_error_handling.py -v --durations=10
```

### Show slowest tests
```bash
pytest tests/integration/test_error_handling.py -v --durations=0
```

## Filtering Tests

### Run tests matching pattern
```bash
# Run only tests with "timeout" in name
pytest tests/integration/test_error_handling.py -k "timeout" -v

# Run tests NOT matching pattern
pytest tests/integration/test_error_handling.py -k "not concurrent" -v
```

## Test Collection

### List all tests without running
```bash
pytest tests/integration/test_error_handling.py --collect-only
```

### Count tests
```bash
pytest tests/integration/test_error_handling.py --collect-only | grep "<Function" | wc -l
```

## Coverage Reporting

### Generate coverage report
```bash
pytest tests/integration/test_error_handling.py --cov=file_to_slides --cov-report=html
```

### Show coverage in terminal
```bash
pytest tests/integration/test_error_handling.py --cov=file_to_slides --cov-report=term-missing
```

## CI/CD Integration

### Pre-deployment check (fail fast)
```bash
pytest tests/integration/test_error_handling.py -x -v
```

### Full test with report
```bash
pytest tests/integration/test_error_handling.py -v --tb=short --junit-xml=test-results.xml
```

### Continuous testing (watch for changes)
```bash
# Requires pytest-watch
ptw tests/integration/test_error_handling.py
```

## Debugging

### Run with Python debugger
```bash
pytest tests/integration/test_error_handling.py -v --pdb
```

### Drop into debugger on failure
```bash
pytest tests/integration/test_error_handling.py -v --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb
```

### Show local variables on failure
```bash
pytest tests/integration/test_error_handling.py -v -l
```

## Parallel Execution

### Run tests in parallel (requires pytest-xdist)
```bash
pytest tests/integration/test_error_handling.py -v -n auto
```

### Run on 4 workers
```bash
pytest tests/integration/test_error_handling.py -v -n 4
```

## Example Scenarios

### Scenario 1: Quick validation (30 seconds)
```bash
pytest tests/integration/test_error_handling.py::TestInvalidInputs -v
```

### Scenario 2: Full validation before deploy
```bash
pytest tests/integration/test_error_handling.py -v --tb=short
```

### Scenario 3: Debug failing test
```bash
pytest tests/integration/test_error_handling.py::TestAPIErrors::test_claude_api_rate_limit_with_retry -vv -s --tb=long
```

### Scenario 4: Check coverage
```bash
pytest tests/integration/test_error_handling.py --cov=file_to_slides --cov-report=term-missing -v
```

### Scenario 5: Find slow tests
```bash
pytest tests/integration/test_error_handling.py -v --durations=10
```

## Configuration

### Create pytest.ini for default options
```ini
[pytest]
testpaths = tests/integration
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

### Run with custom config
```bash
pytest tests/integration/test_error_handling.py -c pytest.ini
```

## Troubleshooting

### Tests not found
```bash
# Check path and collection
pytest tests/integration/test_error_handling.py --collect-only
```

### Import errors
```bash
# Ensure path is correct
python -c "import tests.integration.test_error_handling"
```

### Fixture errors
```bash
# Check conftest.py exists
ls tests/conftest.py
```

## Requirements

- pytest >= 6.0
- unittest.mock (built-in, Python 3.3+)
- requests (HTTP library)
- anthropic (Claude API client)
- threading (built-in)

Install with:
```bash
pip install pytest requests anthropic
```
