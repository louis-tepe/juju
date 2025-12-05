"""
Inference script with Test-Time Augmentation (TTA) and threshold optimization.

Usage:
    poetry run python -m src.inference --model model_best.keras --output submission.csv
"""
import argparse
import json
import os

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


def tta_predict(model, image, n_augments=8):
    """
    Test-Time Augmentation: predict on augmented versions of the image.
    
    Augmentations:
    - Original
    - Horizontal flip
    - Vertical flip
    - 90° rotations (3)
    - Horizontal + Vertical flip
    - 90° + Horizontal flip
    
    Returns averaged prediction.
    """
    preds = []
    
    # Original
    preds.append(model.predict(image, verbose=0))
    
    if n_augments >= 2:
        # Horizontal flip
        preds.append(model.predict(tf.image.flip_left_right(image), verbose=0))
    
    if n_augments >= 3:
        # Vertical flip
        preds.append(model.predict(tf.image.flip_up_down(image), verbose=0))
    
    if n_augments >= 4:
        # 90° rotation
        preds.append(model.predict(tf.image.rot90(image, k=1), verbose=0))
    
    if n_augments >= 5:
        # 180° rotation
        preds.append(model.predict(tf.image.rot90(image, k=2), verbose=0))
    
    if n_augments >= 6:
        # 270° rotation
        preds.append(model.predict(tf.image.rot90(image, k=3), verbose=0))
    
    if n_augments >= 7:
        # H + V flip
        preds.append(model.predict(tf.image.flip_left_right(tf.image.flip_up_down(image)), verbose=0))
    
    if n_augments >= 8:
        # 90° + H flip
        preds.append(model.predict(tf.image.flip_left_right(tf.image.rot90(image, k=1)), verbose=0))
    
    # Average predictions
    return np.mean(preds, axis=0)


def apply_thresholds(raw_preds, thresholds):
    """Apply optimized thresholds to raw regression predictions."""
    thresholds = np.sort(thresholds)
    return np.digitize(raw_preds, thresholds)


def load_and_preprocess(path, image_size, use_ben_graham=True):
    """Load and preprocess a single image."""
    import cv2
    from src.data.preprocess import crop_image_from_gray
    
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if use_ben_graham:
        img = crop_image_from_gray(img)
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)
    else:
        img = cv2.resize(img, (image_size, image_size))
    
    # EfficientNet preprocessing
    img = img.astype(np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Inference with TTA")
    parser.add_argument("--model", type=str, default="model_best.keras", help="Path to trained model")
    parser.add_argument("--test_csv", type=str, default="data/test.csv", help="Path to test CSV")
    parser.add_argument("--test_images", type=str, default="data/test_images", help="Path to test images dir")
    parser.add_argument("--thresholds", type=str, default="thresholds.json", help="Path to thresholds JSON")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output submission file")
    parser.add_argument("--image_size", type=int, default=380, help="Image size")
    parser.add_argument("--tta", type=int, default=8, help="Number of TTA augmentations (1=no TTA)")
    parser.add_argument("--use_ben_graham", action="store_true", default=True, help="Use Ben Graham preprocessing")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    model = keras.models.load_model(args.model)
    
    # Check if regression mode (single output)
    output_shape = model.output_shape[-1]
    is_regression = output_shape == 1
    print(f"Model output: {'REGRESSION' if is_regression else 'CLASSIFICATION'} (shape: {output_shape})")
    
    # Load thresholds if regression mode
    thresholds = None
    if is_regression and os.path.exists(args.thresholds):
        with open(args.thresholds, 'r') as f:
            thresholds = json.load(f)["thresholds"]
        print(f"Loaded thresholds: {thresholds}")
    
    # Load test data
    test_df = pd.read_csv(args.test_csv)
    print(f"Found {len(test_df)} test samples")
    
    predictions = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
        img_path = os.path.join(args.test_images, f"{row['id_code']}.png")
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            predictions.append(0)
            continue
        
        img = load_and_preprocess(img_path, args.image_size, args.use_ben_graham)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Predict with TTA
        if args.tta > 1:
            pred = tta_predict(model, img, n_augments=args.tta)
        else:
            pred = model.predict(img, verbose=0)
        
        # Convert to class
        if is_regression:
            raw_pred = pred.flatten()[0]
            if thresholds:
                final_pred = apply_thresholds([raw_pred], thresholds)[0]
            else:
                final_pred = int(np.clip(np.round(raw_pred), 0, 4))
        else:
            final_pred = int(np.argmax(pred, axis=-1)[0])
        
        predictions.append(final_pred)
    
    # Create submission
    submission_df = pd.DataFrame({
        'id_code': test_df['id_code'],
        'diagnosis': predictions
    })
    submission_df.to_csv(args.output, index=False)
    print(f"\nSaved submission to {args.output}")
    print(f"Prediction distribution: {pd.Series(predictions).value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
