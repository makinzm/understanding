.PHONY: test clean

# Run all tests in scripts directory
test:
	@echo "Running all tests..."
	@for test_file in scripts/test-*.sh; do \
		if [ -f "$$test_file" ]; then \
			echo ""; \
			echo "Running $$test_file..."; \
			bash "$$test_file"; \
		fi; \
	done
	@echo ""
	@echo "All tests completed!"

clean:
	@echo "Nothing to clean"
