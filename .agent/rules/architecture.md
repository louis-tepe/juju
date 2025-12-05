# Architecture & Design

## Overview (Pipeline)

```mermaid
graph TD
    A[Raw Data (Images + CSV)] -->|create_folds.py| B[Train Folds CSV]
    B --> C[Data Loader (tf.data)]
    C -->|Preprocess: Ben Graham| D[Processed Images]
    C -->|Augment: TF-Native or Albumentations| D
    D --> E[Model (EfficientNet + GeM)]
    E -->|Training (CrossEntropy)| F[Weights (.keras)]
    F -->|Submission Script| G[Inference (TTA + Thresholds)]
    G --> H[submission.csv]
```

## 1. Data Pipeline (`src/data`)

- **Loader**: `tf.data.Dataset` avec prefetch (pas de cache pour éviter les blocages M2).
- **Preprocessing** :
  - **Ben Graham** : Optimisation de la visibilité des lésions (Crop + Resize + Gaussian Blend).
  - **Mode rapide** : Resize simple sans Ben Graham (`use_ben_graham: false`).
- **Augmentation** :
  - **TF-Native** (par défaut test) : `tf.image` ops (flip, rot90, brightness). GPU-friendly.
  - **Albumentations** : Pour augmentations avancées (ShiftScaleRotate, HueSaturation).

## 2. Models (`src/models`)

- **Pattern Factory** : Instanciation dynamique des backbones via `factory.py`.
- **Pooling** : GeM (Generalized Mean Pooling) par défaut.
- **Head** : Classification multi-classe (5 neurones, softmax).

## Directory Structure

```
.
├── configs/                # Configuration files (Hydra/YAML)
│   ├── experiment/         # test.yaml, production.yaml
│   ├── model/              # efficientnet.yaml
│   ├── train/              # default.yaml
│   └── config.yaml         # Main config
├── data/                   # Données (GitIgnored)
│   ├── train_images/       # Images d'entraînement
│   ├── test_images/        # Images de test
│   └── *.csv               # Métadonnées
├── docs/                   # Documentation & Rules
├── src/                    # Source Code
│   ├── data/               # Data Loading & Processing
│   │   ├── loader.py       # tf.data.Dataset pipelines
│   │   └── preprocess.py   # Ben Graham, Resizing
│   ├── models/             # Model Definitions
│   ├── training/           # Training Loops & Callbacks
│   └── utils/              # Utilities
├── outputs/                # Logs, Checkpoints (GitIgnored)
├── scripts/                # Utility scripts
├── tests/                  # Unit tests
├── pyproject.toml          # Poetry dependencies
└── README.md
```

## Design Patterns

### 1. Config-Driven Development

Tout paramètre doit être défini dans `configs/` et injecté via Hydra. **Aucun "Magic Number" dans le code.**

### 2. Factory Pattern

Utiliser des "Factories" pour instancier les modèles. Changement d'architecture via config YAML.

### 3. Pipeline Pattern (tf.data)

Le chargement des données est un pipeline asynchrone optimisé :

- `load` -> `decode` -> `augment` -> `batch` -> `prefetch`.

### 4. Mode Test vs Production

| Aspect       | Test (M2)       | Production (GPU) |
| ------------ | --------------- | ---------------- |
| Image Size   | 224px           | 512px            |
| Batch Size   | 32              | 16               |
| Backbone     | EfficientNet-B0 | EfficientNet-B4  |
| Augmentation | TF-Native       | Albumentations   |
| Ben Graham   | Off             | On               |
