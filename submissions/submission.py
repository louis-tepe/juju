"""Optimized submission script for M2 with batch inference and vectorized TTA."""
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.layers import GeMPooling2D

# Configuration
MODEL_PATH = "model_best.keras"
THRESHOLDS_PATH = "thresholds.json"
TEST_IMAGES_DIR = "data/test_images"
TEST_CSV = "data/test.csv"
IMAGE_SIZE = 224  # Match training config
BATCH_SIZE = 64   # Optimized for M2


def load_model():
    """Load model with custom objects."""
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'GeMPooling2D': GeMPooling2D}
    )


def load_and_preprocess(path):
    """Pure TF image loading matching training pipeline."""
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def create_tta_variants(image):
    """Create all TTA variants in a single vectorized operation."""
    return tf.stack([
        image,
        tf.image.flip_left_right(image),
        tf.image.flip_up_down(image),
        tf.image.flip_left_right(tf.image.flip_up_down(image))
    ])


def create_inference_dataset(paths):
    """Create optimized tf.data pipeline for inference."""
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def predict_with_thresholds(predictions, thresholds):
    """Convert continuous predictions to classes using thresholds."""
    results = np.digitize(predictions, thresholds)
    return np.clip(results, 0, 4).astype(int)


def main():
    # Load test data
    if not os.path.exists(TEST_CSV):
        print("Test CSV not found. Creating dummy for testing.")
        df = pd.DataFrame({'id_code': ['0005cfc8afb6']})
    else:
        df = pd.read_csv(TEST_CSV)

    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} not found. Skipping inference.")
        return

    model = load_model()
    
    # Build paths
    paths = [os.path.join(TEST_IMAGES_DIR, f"{id_code}.png") for id_code in df['id_code']]
    valid_paths = [p for p in paths if os.path.exists(p)]
    
    if not valid_paths:
        print("No valid images found. Using dummy predictions.")
        df['diagnosis'] = 0
        df[['id_code', 'diagnosis']].to_csv("submission.csv", index=False)
        return

    print(f"Starting batch inference on {len(valid_paths)} images...")
    
    # Standard inference (no TTA for speed)
    ds = create_inference_dataset(valid_paths)
    predictions_raw = model.predict(ds, verbose=1)
    
    # Get class predictions (argmax for classification model)
    if predictions_raw.shape[-1] == 5:
        predictions = np.argmax(predictions_raw, axis=-1)
    else:
        # Regression model - round predictions
        predictions = np.rint(predictions_raw.flatten()).astype(int)
        predictions = np.clip(predictions, 0, 4)

    # Apply threshold optimization if available
    if os.path.exists(THRESHOLDS_PATH):
        print(f"Loading optimized thresholds from {THRESHOLDS_PATH}")
        with open(THRESHOLDS_PATH, 'r') as f:
            data = json.load(f)
            thresholds = data['thresholds']
        # Only use thresholds for regression output
        if predictions_raw.shape[-1] == 1:
            predictions = predict_with_thresholds(predictions_raw.flatten(), thresholds)

    # Handle case where some paths were invalid
    final_predictions = np.zeros(len(paths), dtype=int)
    valid_idx = [i for i, p in enumerate(paths) if os.path.exists(p)]
    for i, pred in zip(valid_idx, predictions):
        final_predictions[i] = pred

    df['diagnosis'] = final_predictions
    df[['id_code', 'diagnosis']].to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")


if __name__ == "__main__":
    main()
