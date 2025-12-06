#!/usr/bin/env python3
"""
Ensemble predictions from multiple fold models.
Supports both averaging probabilities and applying optimized thresholds.

Usage:
    poetry run python scripts/ensemble_predictions.py [--test]
"""
import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import keras
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

from src.inference import load_and_preprocess, tta_predict
# Import custom layers/losses so they're registered before model loading
from src.models import layers  # noqa: F401 - registers GeMPooling2D, ScaleLayer
from src.models import losses  # noqa: F401 - registers CategoricalFocalCrossentropy


def optimize_thresholds(y_true, y_pred_proba):
    """
    Optimize decision thresholds for ordinal classification.
    Converts probabilities to expected value, then finds optimal cutoffs.
    """
    # Convert probabilities to expected value (ordinal score)
    ordinal_scores = np.sum(y_pred_proba * np.arange(5), axis=1)
    
    def neg_qwk(thresholds):
        thresholds = np.sort(thresholds)
        preds = np.digitize(ordinal_scores, thresholds)
        return -cohen_kappa_score(y_true, preds, weights='quadratic')
    
    # Initial thresholds (evenly spaced)
    init_thresholds = [0.5, 1.5, 2.5, 3.5]
    
    result = minimize(
        neg_qwk,
        init_thresholds,
        method='Nelder-Mead',
        options={'maxiter': 1000}
    )
    
    return np.sort(result.x), -result.fun


def load_oof_predictions(output_dir: Path, n_folds: int = 5):
    """Load and combine OOF predictions from all folds."""
    all_oof = []
    
    for fold in range(n_folds):
        fold_dir = output_dir / f"fold_{fold}"
        oof_path = fold_dir / "oof.csv"
        
        if not oof_path.exists():
            print(f"⚠️ Missing OOF for fold {fold}: {oof_path}")
            continue
        
        oof_df = pd.read_csv(oof_path)
        oof_df['fold'] = fold
        all_oof.append(oof_df)
    
    if not all_oof:
        raise FileNotFoundError("No OOF predictions found!")
    
    return pd.concat(all_oof, ignore_index=True)


def ensemble_test_predictions(
    output_dir: Path,
    test_csv: str,
    test_images: str,
    image_size: int = 512,
    use_tta: bool = True,
    n_folds: int = 5
):
    """Generate ensemble predictions on test set."""
    test_df = pd.read_csv(test_csv)
    print(f"Found {len(test_df)} test samples")
    
    # Load all fold models
    models = []
    for fold in range(n_folds):
        model_path = output_dir / f"fold_{fold}" / "model_best.keras"
        if model_path.exists():
            print(f"Loading model from {model_path}")
            models.append(keras.models.load_model(
                model_path,
                custom_objects={
                    'GeMPooling2D': layers.GeMPooling2D,
                    'ScaleLayer': layers.ScaleLayer,
                    'CategoricalFocalCrossentropy': losses.CategoricalFocalCrossentropy,
                    'KappaLoss': losses.KappaLoss,
                }
            ))
        else:
            print(f"⚠️ Missing model for fold {fold}")
    
    if not models:
        raise FileNotFoundError("No models found!")
    
    print(f"Loaded {len(models)} models for ensemble")
    
    # Predict
    all_preds = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        img_path = os.path.join(test_images, f"{row['id_code']}.png")
        
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            all_preds.append(np.zeros(5))
            continue
        
        img = load_and_preprocess(img_path, image_size, use_ben_graham=True)
        img = np.expand_dims(img, axis=0)
        
        # Average predictions from all models
        fold_preds = []
        for model in models:
            if use_tta:
                pred = tta_predict(model, img, n_augments=8)
            else:
                pred = model.predict(img, verbose=0)
            fold_preds.append(pred)
        
        avg_pred = np.mean(fold_preds, axis=0)
        all_preds.append(avg_pred.flatten())
    
    return test_df['id_code'].values, np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Ensemble predictions")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory with fold outputs")
    parser.add_argument("--test", action="store_true", help="Generate test predictions (requires test images)")
    parser.add_argument("--test_csv", type=str, default="data/test.csv")
    parser.add_argument("--test_images", type=str, default="data/test_images")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--no_tta", action="store_true", help="Disable TTA for faster inference")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Step 1: Load OOF predictions and optimize thresholds
    print("\n" + "=" * 60)
    print("STEP 1: Loading OOF predictions and optimizing thresholds")
    print("=" * 60)
    
    oof_df = load_oof_predictions(output_dir)
    print(f"Loaded {len(oof_df)} OOF predictions")
    
    # Extract probability columns if present
    prob_cols = [col for col in oof_df.columns if col.startswith('prob_')]
    
    if prob_cols:
        y_true = oof_df['diagnosis'].values
        y_pred_proba = oof_df[prob_cols].values
        
        thresholds, qwk = optimize_thresholds(y_true, y_pred_proba)
        print(f"\n📊 Optimized Thresholds: {thresholds}")
        print(f"📊 OOF QWK with optimized thresholds: {qwk:.4f}")
        
        # Save thresholds
        with open("thresholds.json", "w") as f:
            json.dump({"thresholds": thresholds.tolist(), "qwk": qwk}, f, indent=2)
        print("✅ Saved thresholds to thresholds.json")
    else:
        print("⚠️ No probability columns found in OOF. Using argmax.")
        thresholds = None
    
    # Step 2: Generate test predictions if requested
    if args.test:
        print("\n" + "=" * 60)
        print("STEP 2: Generating ensemble test predictions")
        print("=" * 60)
        
        id_codes, test_proba = ensemble_test_predictions(
            output_dir,
            args.test_csv,
            args.test_images,
            args.image_size,
            use_tta=not args.no_tta
        )
        
        # Apply thresholds or argmax
        if thresholds is not None:
            ordinal_scores = np.sum(test_proba * np.arange(5), axis=1)
            predictions = np.digitize(ordinal_scores, thresholds)
        else:
            predictions = np.argmax(test_proba, axis=1)
        
        # Create submission
        submission = pd.DataFrame({
            'id_code': id_codes,
            'diagnosis': predictions
        })
        submission.to_csv("submission.csv", index=False)
        print(f"\n✅ Saved submission to submission.csv")
        print(f"Prediction distribution: {pd.Series(predictions).value_counts().sort_index().to_dict()}")
    
    print("\n" + "=" * 60)
    print("✅ ENSEMBLE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
