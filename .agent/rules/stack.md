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
