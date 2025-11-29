.PHONY: install folds tfrecords train experiment inference clean

install:
	poetry install

folds:
	poetry run python scripts/create_folds.py

tfrecords:
	poetry run python scripts/create_tfrecords.py

train:
	poetry run python src/train.py

experiment:
	poetry run python src/train.py experiment=debug

inference:
	poetry run python submission.py

optimize:
	poetry run python scripts/optimize_thresholds.py

clean:
	rm -rf outputs/ multirun/ __pycache__/ .pytest_cache/
