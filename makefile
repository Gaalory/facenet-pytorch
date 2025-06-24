.PHONY: test coverage_report clean format lint typecheck

test:
	python -m pytest --ignore=dependencies
coverage_report:
	python -m coverage run -m pytest --ignore=dependencies
	python -m coverage report --show-missing
clean:
	python -m coverage erase# Format code
format:
	poetry run ruff format .
lint:
	poetry run ruff check ./src --fix
typecheck:
	poetry run mypy ./src