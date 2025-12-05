# Tech Stack & Tools (2025 Standards)

Stack technique officielle pour le projet APTOS 2019.

## Core

| Tool   | Version | Usage                   |
| ------ | ------- | ----------------------- |
| Python | 3.11+   | Langage principal       |
| Poetry | Latest  | Gestion des dépendances |
| Git    | Latest  | Version control         |

## Deep Learning

| Library          | Version | Usage                |
| ---------------- | ------- | -------------------- |
| TensorFlow       | 2.16+   | Framework DL         |
| Keras            | 3.0+    | API haut niveau      |
| tensorflow-metal | Latest  | Accélération M1/M2   |
| pandas           | 2.2+    | Manipulation données |
| numpy            | 1.26+   | Calcul numérique     |

## Computer Vision

| Library                | Usage                              |
| ---------------------- | ---------------------------------- |
| opencv-python-headless | Traitement d'images bas niveau     |
| albumentations         | Augmentations avancées (optionnel) |

## MLOps & Configuration

| Tool            | Usage                             |
| --------------- | --------------------------------- |
| hydra-core 1.3+ | Configuration dynamique YAML/CLI  |
| wandb           | Tracking expériences et métriques |

## Code Quality

| Tool   | Usage                                            |
| ------ | ------------------------------------------------ |
| ruff   | Linter + Formatter (remplace flake8/black/isort) |
| mypy   | Type checking statique                           |
| pytest | Tests unitaires                                  |

## Platform-Specific

### MacOS M1/M2

- `tensorflow-metal` pour accélération GPU Metal
- `use_native_augment: true` pour éviter les bottlenecks CPU
- Batch size réduit (32) pour stabilité

### Linux/CUDA GPU

- `use_mixed_precision: true` activé
- `use_xla: true` pour optimisation
- Batch size plus élevé (16-64)

## Kaggle Specifics

- Compatibilité avec images Docker Kaggle (GPU/TPU)
- Offline Mode : Chargement dépendances via datasets
- Script unique `submission.py` avec TTA
