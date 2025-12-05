# Development Workflow

Ce document décrit les processus de développement, d'expérimentation et de soumission.

## 1. Git Workflow

- **main** : Branche stable, code prêt pour la production/soumission.
- **feat/nom-feature** : Nouvelles fonctionnalités.
- **exp/nom-experience** : Tests d'hypothèses.
- **fix/nom-bug** : Corrections de bugs.

**Règles de Commit** : Conventional Commits (`feat: ...`, `fix: ...`, `docs: ...`).

## 2. Experimentation Workflow

1. **Définir l'hypothèse** : Qu'est-ce qu'on teste ?
2. **Créer une Config** : `configs/experiment/ma_config.yaml`
3. **Lancer l'entraînement** :
   ```bash
   make train-test           # Test rapide sur M2
   make train-prod           # Production sur GPU
   ```
4. **Tracker** : Métriques envoyées à WandB.
5. **Analyser** : Comparer sur le dashboard WandB.

## 3. Makefile Commands

| Commande          | Description                         |
| ----------------- | ----------------------------------- |
| `make install`    | Installe les dépendances via Poetry |
| `make folds`      | Crée les folds de cross-validation  |
| `make train-test` | Entraînement rapide (M2/CPU)        |
| `make train-prod` | Entraînement production (GPU)       |
| `make lint`       | Vérifie le code (ruff, mypy)        |
| `make test`       | Lance les tests unitaires           |
| `make clean`      | Nettoie les outputs                 |

## 4. Kaggle Submission Workflow

1. **Model Export** : Sauvegarder `.keras` et uploader comme Kaggle Dataset.
2. **Inference Script** : `submission.py` contient tout le code d'inférence.
3. **Validation Locale** : Tester localement avant soumission.

## 5. CI/CD Pipeline

GitHub Actions vérifie à chaque push :

- **Quality Check** : `ruff check`, `ruff format`, `mypy`
- **Testing** : `pytest`
- **Environment** : `ubuntu-latest`, Python 3.11
