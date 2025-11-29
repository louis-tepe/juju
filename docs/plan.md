# Plan de Mise en Œuvre : APTOS 2019 Blindness Detection (Gold Strategy)

Ce document définit la stratégie optimale pour viser le **Top 1%** et maximiser le score **Quadratic Weighted Kappa (QWK)**.

## 1. Configuration de l'Environnement

- **Outils** : `pyenv` (3.10+), `poetry`.
- **Framework** : `tensorflow` / `keras` (XLA enabled pour vitesse).
- **Bibliothèques Avancées** :
  - `albumentations` : Augmentations complexes.
  - `tensorflow_addons` : Pour certaines losses/optimizers (si nécessaire).
  - `scikit-learn` : StratifiedGroupKFold.

## 2. Données & Stratégie de Dataset

Le volume de données est la clé. Le dataset 2019 est trop petit pour généraliser parfaitement.

- **Dataset Principal (2019)** : ~3.6k images.
- **Dataset Externe (2015 EyePACS)** : ~35k images (Kaggle Competition précédente).
  - _Stratégie_ : Utiliser 2015 pour le **Pre-training**, puis 2019 pour le **Fine-tuning**.
  - _Attention_ : Les distributions de classes et la qualité diffèrent. Il faudra peut-être sous-échantillonner la classe 0 du 2015.
- **Nettoyage & Anti-Leakage** :
  - Supprimer les images dupliquées ou de trop mauvaise qualité du dataset 2015.
  - **CRITIQUE** : Effectuer une dé-duplication croisée (Image Hashing) entre 2015 et 2019 pour s'assurer qu'aucune image de validation n'a été vue pendant le pre-training.

## 3. Pipeline de Prétraitement (Preprocessing)

- **Ben Graham's Preprocessing** :
  1.  **Circular Crop** : Suppression des bords noirs.
  2.  **Gaussian Blur** : Normalisation de la texture/couleur ($\alpha=4, \beta=-4, \gamma=128$).
- **Progressive Resizing** :
  - Phase 1 (Pre-train 2015) : 256x256.
  - Phase 2 (Fine-tune 2019) : 384x384 -> 512x512 (Tester 768x768 ou 1024x1024 sur un modèle si le temps d'inférence le permet).
  - _Gain_ : Convergence plus rapide et meilleure généralisation.

## 4. Augmentation Avancée

Pour combattre l'overfitting drastique sur 3.6k images.

- **Géométrique** : Rotation (360°), Flip, Zoom, Shear.
- **Pixel-level** : RandomBrightnessContrast, HueSaturationValue.
- **Regularization** :
  - **Mixup** : Mélange linéaire de 2 images et de leurs labels ($ \lambda x_i + (1-\lambda) x_j $).
  - **Cutmix** : Remplacer une zone de l'image par une zone d'une autre image.
  - _Note_ : Indispensable pour les gros backbones.

## 5. Stratégie de Modélisation (Ensemble Hétérogène)

Ne jamais miser sur un seul modèle.

### Architectures (Backbones)

Utiliser des modèles pré-entraînés sur ImageNet (NoisyStudent si dispo).

1.  **EfficientNet B4/B5** : Performance/Vitesse équilibrée.
2.  **SE-ResNeXt50 (32x4d)** : Architecture différente pour diversifier les erreurs.
3.  **InceptionResNetV2** : Souvent robuste sur les fonds d'œil.

### Tête du Modèle (Head) & Loss

- **Pooling** : Generalized Mean Pooling (GeM) souvent meilleur que GlobalAveragePooling.
- **Approche Hybride** :
  - Sortie : 1 neurone (Régression) pour l'ordre.
  - **Alternative** : Régression Ordinale (Coral Layer ou encodage multi-label `[1, 1, 0, 0, 0]`) pour stabiliser la convergence.
  - **Loss** : `Huber Loss` (robuste) ou une **SoftKappa Loss** (approximation différentiable du QWK).

## 6. Entraînement & Validation

- **Stratégie de Split** : **Stratified Group K-Fold (k=5)**.
  - _Critique_ : Grouper par `id_code` (Patient) si possible pour éviter le "Data Leakage" (œil gauche dans train, œil droit dans val).
- **Optimiseur** : Utiliser **RAdam** ou **Lookahead** pour éviter les minimums locaux précoces et stabiliser l'entraînement.
- **Training Schedule** :
  1.  **Warmup** : Entraîner seulement la Head (Backbone gelé) sur 2019.
  2.  **Pre-training** : Dé-geler et entraîner sur 2015 (LR faible).
  3.  **Fine-tuning** : Entraîner sur 2019 (LR très faible, Cosine Decay).
- **Stochastic Weight Averaging (SWA)** : À la fin du training, moyenner les poids des N dernières époques pour un modèle plus stable.

## 7. Post-Processing & Optimisation

- **Optimisation des Seuils** : Utiliser `scipy.optimize.minimize` sur les prédictions OOF (Out-Of-Fold) pour trouver les seuils qui maximisent le QWK.
- **Pseudo-Labeling** (Si temps de calcul permet) :
  1.  Entraîner l'ensemble.
  2.  Prédire sur le Test Set (Public).
  3.  Ajouter les prédictions confiantes au Train Set.
  4.  Re-finetuner.

## 8. Pipeline de Soumission (Inférence)

- **TTA (Test Time Augmentation)** : 4x ou 8x (Original, Flip H, Flip V, Rotate 90). Moyenne des prédictions.
- **Ensembling** : Moyenne pondérée des sorties des différents modèles (EfficientNet + ResNeXt + Inception).
- **Contraintes** : Vérifier que le temps d'inférence total < 9h (ajuster le nombre de TTA ou de modèles si nécessaire).

## Roadmap

1.  [ ] **Setup** : Env Poetry + Téléchargement Dataset 2015 (via Kaggle API ou script).
2.  [ ] **Data Pipeline** : Implémenter Ben Graham + Mixup/Cutmix + TFRecord (pour vitesse).
3.  [ ] **Model Factory** : Classe générique pour instancier différents backbones avec GeM Pooling.
4.  [ ] **Training Loop** : Boucle custom (GradientTape) ou Callbacks pour gérer Mixup et SWA.
5.  [ ] **Optimization** : Script de recherche de seuils.
6.  [ ] **Submission** : Kernel d'inférence optimisé.
