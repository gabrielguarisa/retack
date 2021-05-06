.PHONY: tests precommit

tests:
	poetry run pytest --cov-report term-missing --cov=retack tests/

precommit:
	poetry run pre-commit run --all-files
