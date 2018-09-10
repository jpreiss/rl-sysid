lint:
	pep8 --max-line-length=100 libexperiment.py
	mypy --ignore-missing-imports libexperiment.py
