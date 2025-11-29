# Global Rules

- **Senior Engineer Persona:** Be direct, objective, and succinct. Omit pleasantries, apologies, and conversational filler.
- **Code Integrity:** Prioritize modular, DRY, and readable code. Strictly adhere to the idioms and conventions of the active language or framework.
- **Context Awareness:** Analyze the directory structure and existing patterns before acting to ensure consistency with the current project state.
- **Intentional Documentation:** Comment only to explain the **why** (business logic/reasoning) behind complex decisions, never the **what** (syntax).
- **Iterative Logic:** Break down complex tasks into atomic steps. Verify assumptions before implementation.

# Tech Stack & Tools (2025 Standards)

Ce document définit la stack technique officielle pour le projet APTOS 2019. L'objectif est la performance, la reproductibilité et la rapidité d'itération.

## Core

- **Language** : Python 3.11+ (Performance boost vs 3.9/3.10).
- **Package Manager** : `poetry` (Gestion déterministe des dépendances et environnements virtuels).
- **Version Control** : Git.
- **Data Versioning** : `DVC` (Data Version Control) pour tracker les versions du dataset.

## Deep Learning & Data Science

- **Framework** : `tensorflow` 2.16+ / `keras` 3.0+ (Backend agnostic, haute performance).
- **Data Manipulation** : `pandas` 2.2+ (Standard) & `numpy` 1.26+.
- **Computer Vision** :
  - `opencv-python-headless` : Traitement d'images bas niveau.
  - `albumentations` : Augmentations d'images SOTA (State of the Art).
- **Visualization** : `seaborn` & `matplotlib`.

## MLOps & Experimentation

- **Configuration** : `hydra-core` (Gestion dynamique des configs via YAML/CLI).
- **Tracking** : `wandb` (Weights & Biases) pour le suivi des expériences, métriques et artefacts (si Internet dispo en training).
- **Optimization** : `optuna` (Recherche d'hyperparamètres bayésienne).

## Code Quality (Modern Python)

- **Linter & Formatter** : `ruff` (Remplace Flake8, Black, Isort avec une vitesse x100).
- **Type Checking** : `mypy` (Typage statique strict).
- **Testing** : `pytest`.

## Kaggle Specifics

- **Kernel Environment** : Compatibilité avec les images Docker Kaggle (GPU/TPU).
- **Offline Mode** : Capacité à charger les dépendances via des datasets (`pip download` préalable).


# Architecture & Codebase Structure

L'architecture doit favoriser la modularité, la réutilisabilité et la clarté. Nous adoptons une structure de type "Package" avec une séparation claire des responsabilités.

## Directory Structure

```
.
├── configs/                # Configuration files (Hydra/YAML)
│   ├── experiment/         # Configs spécifiques aux expériences
│   ├── model/              # Configs des architectures modèles
│   ├── train/              # Configs d'entraînement (LR, Batch size...)
│   └── config.yaml         # Main config
├── data/                   # Données (GitIgnored)
│   ├── raw/                # Données originales (Immutable)
│   ├── processed/          # Données prétraitées (TFRecords, Cropped images)
│   └── external/           # Datasets externes (2015 EyePACS)
├── docs/                   # Documentation & Rules
├── notebooks/              # Jupyter Notebooks (Exploration & Prototyping)
│   └── naming_convention: 01_eda_initial.ipynb
├── src/                    # Source Code (Package Python)
│   ├── data/               # Data Loading & Processing
│   │   ├── loader.py       # tf.data.Dataset pipelines
│   │   └── preprocess.py   # Ben Graham, Resizing
│   ├── models/             # Model Definitions
│   │   ├── backbones.py    # EfficientNet, ResNeXt factories
│   │   ├── heads.py        # Custom heads (GeM, Regression)
│   │   └── losses.py       # KappaLoss, HuberLoss
│   ├── training/           # Training Loops & Callbacks
│   │   ├── trainer.py      # Main training class
│   │   └── callbacks.py    # Custom callbacks (SWA, Logging)
│   └── utils/              # Utilities
│       ├── metrics.py      # QWK implementation
│       └── seeding.py      # Reproducibility utils
├── outputs/                # Logs, Checkpoints (GitIgnored)
├── submissions/            # CSV files generated for submission
├── scripts/                # Utility scripts (Data prep, TFRecord conversion)
├── tests/                  # Unit tests
├── pyproject.toml          # Poetry dependencies
└── README.md
```

## Design Patterns

### 1. Config-Driven Development

Tout paramètre (hyperparamètre, chemin, choix d'architecture) doit être défini dans `configs/` et injecté via Hydra. **Aucun "Magic Number" dans le code.**

### 2. Factory Pattern

Utiliser des "Factories" pour instancier les modèles et les optimizers. Cela permet de changer d'architecture simplement en modifiant une ligne dans la config YAML.

- Exemple : `get_model(cfg.model.name)`

### 3. Pipeline Pattern (tf.data)

Le chargement des données doit être un pipeline asynchrone et optimisé avec `tf.data`.

- `load` -> `decode` -> `augment` -> `batch` -> `prefetch`.

### 4. Modularité

Le code dans `src/` doit être indépendant des Notebooks. Les Notebooks importent `src` pour exécuter des tâches, ils ne contiennent pas de logique métier complexe.


# Coding Conventions

## Naming Conventions

- **Variables/Functions** : `snake_case` (ex: `load_data`, `image_size`).
- **Classes** : `PascalCase` (ex: `EfficientNetModel`, `CustomDataGenerator`).
- **Constants** : `UPPER_CASE` (ex: `BATCH_SIZE`, `SEED`).
- **Private Members** : `_snake_case` (ex: `_internal_process`).
- **Experiment Names** : `[Date]_[Model]_[Resolution]_[Note]` (ex: `20231027_effb4_512_mixup`).

## Docstrings

Nous utilisons le **Google Style** pour les docstrings.

```python
def process_image(image: np.ndarray, size: int = 256) -> np.ndarray:
    """Resizes and normalizes an image.

    Args:
        image (np.ndarray): Input image array (H, W, C).
        size (int, optional): Target size. Defaults to 256.

    Returns:
        np.ndarray: Processed image.
    """
    ...
```

## Imports

Les imports doivent être triés (géré automatiquement par `ruff`).
Ordre :

1.  Standard Library (`os`, `sys`...)
2.  Third Party (`numpy`, `tensorflow`...)
3.  Local Application (`src.utils`...)

## Error Handling

- Préférer les exceptions explicites (`ValueError`, `FileNotFoundError`) aux `Exception` génériques.
- Utiliser des assertions pour vérifier les invariants (dimensions des tenseurs, types).

## Logging

- Ne pas utiliser `print()`. Utiliser le module `logging` ou `wandb.log()`.


# Definition of Done (DoD)

Une tâche ou une fonctionnalité est considérée comme "Terminée" uniquement si elle respecte les critères suivants.

## 1. Code Quality

- [ ] **Linting** : Le code passe `ruff check` sans erreurs.
- [ ] **Formatting** : Le code est formaté avec `ruff format`.
- [ ] **Typing** : Les fonctions critiques ont des type hints et passent `mypy` (ou au moins pas d'erreurs flagrantes).
- [ ] **Clean** : Pas de code mort, de `print()` de debug, ou de commentaires obsolètes.

## 2. Reproducibility

- [ ] **Seeding** : Tous les processus aléatoires (Numpy, TF, Python) sont fixés avec une seed globale.
- [ ] **Config** : L'expérience est entièrement reproductible à partir de son fichier de configuration YAML.
- [ ] **Environment** : Les dépendances sont lockées dans `poetry.lock`.
- [ ] **Cleanup** : Les fichiers temporaires et checkpoints non retenus sont supprimés.

## 3. Functionality & Testing

- [ ] **Unit Tests** : Les nouvelles fonctions utilitaires ont des tests unitaires (dans `tests/`).
- [ ] **Data Tests** : Vérification de la validité des données (pas de NaNs, dimensions correctes) et tests de régression sur un subset fixe.
- [ ] **Integration** : Le pipeline d'entraînement tourne de bout en bout (au moins 1 epoch) sans crash.
- [ ] **Inference** : Le script de soumission fonctionne en mode "Internet Off".

## 4. Documentation

- [ ] **Docstrings** : Les classes et fonctions publiques ont des docstrings explicites (Google Style).
- [ ] **README** : Si un nouveau module est ajouté, le README est mis à jour.
- [ ] **Results** : Les résultats de l'expérience (Score QWK, Loss) sont notés (dans WandB ou un log local).

## 5. Performance (Si applicable)

- [ ] **Metric** : L'amélioration du score QWK est vérifiée sur le set de Validation (CV).
- [ ] **Time** : Le temps d'entraînement ou d'inférence reste dans les limites acceptables (9h max pour inférence).


# Development Workflow

Ce document décrit les processus de développement, d'expérimentation et de soumission.

## 1. Git Workflow

Nous utilisons un **Feature Branch Workflow** simplifié.

- **main** : Branche stable, code prêt pour la production/soumission.
- **feat/nom-feature** : Pour le développement de nouvelles fonctionnalités (ex: `feat/mixup-augmentation`).
- **exp/nom-experience** : Pour tester une hypothèse spécifique (ex: `exp/efficientnet-b5`).
- **fix/nom-bug** : Pour les corrections de bugs.

**Règles de Commit** : Conventional Commits (`feat: ...`, `fix: ...`, `docs: ...`).

## 2. Experimentation Workflow

L'expérimentation doit être rigoureuse et traçable.

1.  **Définir l'hypothèse** : Qu'est-ce qu'on teste ? (ex: "Mixup améliore la généralisation").
2.  **Créer une Config** : Créer un fichier `configs/experiment/mixup.yaml` qui surcharge la config par défaut.
3.  **Lancer l'entraînement** :
    ```bash
    python src/train.py experiment=mixup
    ```
4.  **Tracker** : Les métriques (Loss, QWK, LR) sont envoyées automatiquement à WandB.
5.  **Analyser** : Comparer les courbes sur le dashboard WandB.
6.  **Conclure** : Si concluant, merger dans `main` ou garder comme candidat pour l'ensemble final.

## 3. Kaggle Submission Workflow

Le processus de soumission est critique car l'environnement est "Internet Off".

1.  **Freeze Dependencies** : Exporter les requirements ou utiliser les datasets Kaggle pour les libs non-standard.
2.  **Model Export** : Sauvegarder les poids (`.h5` ou `.keras`) et les uploader comme Kaggle Dataset.
3.  **Inference Script** :
    - Créer un script unique `submission.py` (ou un Notebook propre) qui contient tout le code nécessaire à l'inférence.
    - **Sync** : Utiliser un script de build pour packager `src/` en un dataset Kaggle ou utiliser `kaggle-cli` pour push le code.
    - Charger les modèles depuis le Dataset.
    - Exécuter sur le Test Set.
4.  **Validation Locale** : Toujours tester le script d'inférence localement sur un sous-ensemble du Train Set pour vérifier qu'il ne plante pas (Memory, Time).

## 4. Review Process

- Tout code mergé sur `main` doit être relu (Self-review ou Pair-review).
- Vérifier que le code respecte les standards (Ruff, Mypy).