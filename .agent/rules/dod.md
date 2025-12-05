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
- [ ] **CI/CD** : Le pipeline GitHub Actions passe au vert.
- [ ] **Time** : Le temps d'entraînement ou d'inférence reste dans les limites acceptables (9h max pour inférence).
