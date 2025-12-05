.PHONY: install folds train experiment inference clean lint test docker-build

install:
	poetry install

folds:
	poetry run python scripts/create_folds.py

train:
	poetry run python -m src.train

experiment:
	poetry run python -m src.train experiment=debug

train-test:
	poetry run python -m src.train experiment=test

train-prod:
	poetry run python -m src.train experiment=production

inference:
	poetry run python submission.py

optimize:
	poetry run python scripts/optimize_thresholds.py

lint:
	poetry run ruff check .
	poetry run ruff format .
	poetry run mypy src

test:
	poetry run pytest tests/

docker-build:
	docker build -t aptos-training .

clean:
	rm -rf outputs/ multirun/ __pycache__/ .pytest_cache/ data/processed/tfrecords_*
