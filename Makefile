install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black .

lint:
	pylint --disable=R,C */*.py

test:
	python -m pytest

run:
	python -m rag_eval.cli

rag-eval:
	pip install -e .