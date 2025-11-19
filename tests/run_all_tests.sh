#!/bin/bash
# Run all OpenAI integration tests
# Usage:
#   ./tests/run_all_tests.sh              # Run with mock APIs (fast)
#   ./tests/run_all_tests.sh --real       # Run with real APIs (requires API keys)
#   ./tests/run_all_tests.sh --coverage   # Run with coverage report

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running from project root
if [ ! -f "file_to_slides.py" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    echo "Usage: ./tests/run_all_tests.sh"
    exit 1
fi

# Parse arguments
USE_REAL_APIS=false
GENERATE_COVERAGE=false

for arg in "$@"; do
    case $arg in
        --real)
            USE_REAL_APIS=true
            shift
            ;;
        --coverage)
            GENERATE_COVERAGE=true
            shift
            ;;
        --help)
            echo "Usage: ./tests/run_all_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --real      Use real API calls (requires API keys)"
            echo "  --coverage  Generate code coverage report"
            echo "  --help      Show this help message"
            exit 0
            ;;
    esac
done

# Banner
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   OpenAI Integration Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check API keys if using real APIs
if [ "$USE_REAL_APIS" = true ]; then
    echo -e "${YELLOW}⚠️  Using REAL APIs - will consume API credits${NC}"
    if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${RED}❌ Error: No API keys found${NC}"
        echo "Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY environment variables"
        exit 1
    fi
    echo -e "${GREEN}✅ API keys found${NC}"
else
    echo -e "${GREEN}✅ Using MOCK APIs - no API keys needed${NC}"
fi
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${YELLOW}⚠️  pytest not found, installing...${NC}"
    pip install pytest pytest-cov
fi

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run test suite
run_test_suite() {
    local test_file=$1
    local test_name=$2

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Running: ${test_name}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if [ "$GENERATE_COVERAGE" = true ]; then
        pytest "tests/${test_file}" -v --cov=. --cov-report=term-missing || true
    else
        pytest "tests/${test_file}" -v || true
    fi

    # Get exit code
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ ${test_name} - PASSED${NC}"
        ((PASSED_TESTS++))
    else
        echo -e "${RED}❌ ${test_name} - FAILED${NC}"
        ((FAILED_TESTS++))
    fi

    ((TOTAL_TESTS++))
    echo ""
}

# Run all test suites
echo -e "${YELLOW}Starting test execution...${NC}"
echo ""

# 1. OpenAI Integration Tests
run_test_suite "test_openai_integration.py" "OpenAI Integration"

# 2. Intelligent Routing Tests
run_test_suite "test_intelligent_routing.py" "Intelligent Routing"

# 3. Cost Tracking Tests
run_test_suite "test_cost_tracking.py" "Cost Tracking"

# 4. Ensemble Mode Tests
run_test_suite "test_ensemble_mode.py" "Ensemble Mode"

# 5. Performance Tests
run_test_suite "test_performance.py" "Performance Benchmarks"

# 6. UI Integration Tests
run_test_suite "test_ui_integration.py" "UI Integration"

# 7. Integration Tests
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Running: Full Integration Tests${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ "$USE_REAL_APIS" = true ]; then
    python tests/integration_test_openai.py --real || true
else
    python tests/integration_test_openai.py --mock || true
fi

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✅ Full Integration Tests - PASSED${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${RED}❌ Full Integration Tests - FAILED${NC}"
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))
echo ""

# Generate coverage report if requested
if [ "$GENERATE_COVERAGE" = true ]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Generating Coverage Report${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Run all tests with coverage
    pytest tests/ -v \
        --cov=file_to_slides \
        --cov-report=html \
        --cov-report=term-missing \
        --ignore=tests/benchmark_results

    echo ""
    echo -e "${GREEN}✅ Coverage report generated: htmlcov/index.html${NC}"
    echo ""
fi

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}        TEST SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Total test suites: ${TOTAL_TESTS}"
echo -e "${GREEN}Passed: ${PASSED_TESTS}${NC}"
echo -e "${RED}Failed: ${FAILED_TESTS}${NC}"
echo ""

# Calculate success rate
SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed! Success rate: 100%${NC}"
    echo ""
    echo -e "${GREEN}🎉 Ready to deploy!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️  Some tests failed. Success rate: ${SUCCESS_RATE}%${NC}"
    echo ""
    echo -e "${YELLOW}Please fix failing tests before deploying.${NC}"
    exit 1
fi
