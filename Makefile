.PHONY: install folds train experiment inference clean lint test docker-build train-cv ensemble submit-final

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

# ==================== PRODUCTION PIPELINE ====================

train-cv:
	@echo "🚀 Starting 5-Fold Cross-Validation Training..."
	@echo "⏱️  Estimated time: 10-15 hours"
	poetry run python scripts/train_all_folds.py

ensemble:
	@echo "🔗 Generating ensemble predictions with optimized thresholds..."
	poetry run python scripts/ensemble_predictions.py

submit-final:
	@echo "🏆 Full Production Pipeline: Ensemble + Optimized Thresholds"
	poetry run python scripts/ensemble_predictions.py --test
	@echo "✅ Final submission saved to submission.csv"

# ==================== SINGLE FOLD COMMANDS ====================

inference:
	poetry run python -m src.inference --tta 8

inference-fast:
	poetry run python -m src.inference --tta 1

submit:
	@echo "🚀 Creating submission with TTA..."
	poetry run python -m src.inference --tta 8 --output submission.csv
	@echo "✅ Submission saved to submission.csv"

optimize:
	poetry run python scripts/optimize_thresholds.py

# ==================== DEV TOOLS ====================

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

