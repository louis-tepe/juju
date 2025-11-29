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
