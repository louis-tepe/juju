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
