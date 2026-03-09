#!/bin/bash
# Test runner script for Model Compression Pipeline

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  Model Compression Pipeline Tests${NC}"
echo -e "${YELLOW}========================================${NC}\n"

# Parse arguments
TEST_TYPE=${1:-all}
COVERAGE=${2:-no}

run_unit_tests() {
    echo -e "${GREEN}Running Unit Tests...${NC}"
    if [ "$COVERAGE" = "coverage" ]; then
        pytest tests/unit/ -v --cov=model_compression --cov=config --cov-report=term-missing
    else
        pytest tests/unit/ -v
    fi
}

run_integration_tests() {
    echo -e "${GREEN}Running Integration Tests...${NC}"
    pytest tests/integration/ -v
}

run_all_tests() {
    echo -e "${GREEN}Running All Tests...${NC}"
    if [ "$COVERAGE" = "coverage" ]; then
        pytest tests/ -v --cov=model_compression --cov=config --cov-report=html --cov-report=term-missing
        echo -e "\n${YELLOW}Coverage report generated in htmlcov/index.html${NC}"
    else
        pytest tests/ -v
    fi
}

# Run tests based on argument
case $TEST_TYPE in
    unit)
        run_unit_tests
        ;;
    integration)
        run_integration_tests
        ;;
    all)
        run_all_tests
        ;;
    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo "Usage: ./run_tests.sh [unit|integration|all] [coverage]"
        echo "Examples:"
        echo "  ./run_tests.sh                    # Run all tests"
        echo "  ./run_tests.sh unit              # Run unit tests only"
        echo "  ./run_tests.sh integration       # Run integration tests only"
        echo "  ./run_tests.sh all coverage      # Run all tests with coverage"
        exit 1
        ;;
esac

TEST_RESULT=$?

echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ Tests passed!${NC}"
else
    echo -e "${RED}✗ Tests failed!${NC}"
fi

exit $TEST_RESULT
