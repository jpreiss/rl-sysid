files = $(filter-out logger.py, $(wildcard *.py))

lint:
	pep8 --max-line-length=100 --ignore=E231 $(files)
	mypy --ignore-missing-imports $(files)
