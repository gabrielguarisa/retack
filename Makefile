.PHONY: tests precommit

tests:
	pytest --cov-report term-missing --cov=retack tests/

precommit:
	pre-commit run --all-files
