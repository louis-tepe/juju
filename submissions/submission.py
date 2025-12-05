import json
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.preprocess import load_and_preprocess_image
from src.models.layers import GeMPooling2D

# Configuration
MODEL_PATH = "model_best.keras"
THRESHOLDS_PATH = "thresholds.json"
TEST_IMAGES_DIR = "data/test_images"
TEST_CSV = "data/test.csv"
IMAGE_SIZE = 256 # Must match training
BATCH_SIZE = 32
TTA_STEPS = 4 # Original, Flip LR, Flip UD, Both

def load_model():
    # Load model with custom objects
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'GeMPooling2D': GeMPooling2D}
    )
    return model

def process_image(path):
    image = load_and_preprocess_image(path, size=IMAGE_SIZE)
    return image

def tta_predict(model, image):
    # Create TTA batch
    # 1. Original
    # 2. Flip Left-Right
    # 3. Flip Up-Down
    # 4. Rotate 180 (Flip LR + UD)

    img1 = image
    img2 = cv2.flip(image, 1)
    img3 = cv2.flip(image, 0)
    img4 = cv2.flip(image, -1)

    batch = np.stack([img1, img2, img3, img4])
    preds = model.predict(batch, verbose=0)

    return np.mean(preds)

def predict_with_thresholds(predictions, thresholds):
    predictions = np.array(predictions)
    results = []
    for pred in predictions:
        if pred < thresholds[0]:
            results.append(0)
        elif pred >= thresholds[0] and pred < thresholds[1]:
            results.append(1)
        elif pred >= thresholds[1] and pred < thresholds[2]:
            results.append(2)
        elif pred >= thresholds[2] and pred < thresholds[3]:
            results.append(3)
        else:
            results.append(4)
    return np.array(results).astype(int)

def main():
    if not os.path.exists(TEST_CSV):
        print("Test CSV not found. Creating dummy for testing.")
        df = pd.DataFrame({'id_code': ['0005cfc8afb6']}) # Example
    else:
        df = pd.read_csv(TEST_CSV)

    if not os.path.exists(MODEL_PATH):
        print(f"Model {MODEL_PATH} not found. Skipping inference.")
        return

    model = load_model()
    predictions = []

    print(f"Starting inference on {len(df)} images...")

    for idx, row in df.iterrows():
        img_path = os.path.join(TEST_IMAGES_DIR, f"{row['id_code']}.png")
        if not os.path.exists(img_path):
             # Fallback for dummy run
             img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        else:
            img = process_image(img_path)

        pred = tta_predict(model, img)
        predictions.append(pred)

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df)}")

    # Post-processing
    predictions = np.array(predictions)

    if os.path.exists(THRESHOLDS_PATH):
        print(f"Loading optimized thresholds from {THRESHOLDS_PATH}")
        with open(THRESHOLDS_PATH, 'r') as f:
            data = json.load(f)
            thresholds = data['thresholds']
        predictions_final = predict_with_thresholds(predictions, thresholds)
    else:
        print("Using standard rounding.")
        predictions_rounded = np.rint(predictions).astype(int)
        predictions_final = np.clip(predictions_rounded, 0, 4)

    df['diagnosis'] = predictions_final
    df[['id_code', 'diagnosis']].to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    main()
