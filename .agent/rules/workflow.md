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
