#!/bin/bash

# Test suite for title-converter.sh

# Script to test
SCRIPT="scripts/title-converter.sh"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counters
PASSED=0
FAILED=0

# Assert function
assert_equals() {
    local expected="$1"
    local actual="$2"
    local test_name="$3"
    
    if [ "$expected" = "$actual" ]; then
        echo -e "${GREEN}✓ PASS${NC}: $test_name"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAIL${NC}: $test_name"
        echo "  Expected: $expected"
        echo "  Actual:   $actual"
        ((FAILED++))
    fi
}

# Run tests
echo "Running tests for title-converter.sh"
echo "======================================"

# Test 1: Basic title conversion
result=$(echo "Attention Is All You Need" | bash $SCRIPT)
assert_equals "attention-is-all-you-need.md" "$result" "Basic title conversion"

# Test 2: Lowercase title
result=$(echo "hello world" | bash $SCRIPT)
assert_equals "hello-world.md" "$result" "Lowercase title"

# Test 3: Mixed case
result=$(echo "The Quick Brown Fox" | bash $SCRIPT)
assert_equals "the-quick-brown-fox.md" "$result" "Mixed case title"

# Test 4: Single word
result=$(echo "Introduction" | bash $SCRIPT)
assert_equals "introduction.md" "$result" "Single word title"

# Test 5: Multiple spaces
result=$(echo "Machine  Learning  Basics" | bash $SCRIPT)
assert_equals "machine--learning--basics.md" "$result" "Multiple spaces"

# Test 6: All uppercase
result=$(echo "API REFERENCE" | bash $SCRIPT)
assert_equals "api-reference.md" "$result" "All uppercase title"

# Summary
echo "======================================"
echo "Tests passed: $PASSED"
echo "Tests failed: $FAILED"
echo "======================================"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
