# Convenience targets. NOTE: these require an environment you set up yourself;
# this scaffold creates none and runs nothing by default.
.RECIPEPREFIX = >
.PHONY: help lint format test
help:
> @echo "Targets: lint, format, test (require your own configured env)"
lint:
> ruff check .
format:
> ruff format .
test:
> pytest -q
