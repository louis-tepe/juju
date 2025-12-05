import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def create_folds(input_csv, output_csv, n_splits=5, seed=42):
    df = pd.read_csv(input_csv)

    # Ensure we have diagnosis
    if 'diagnosis' not in df.columns:
        print("Error: 'diagnosis' column missing")
        return

    # Ensure we have id_code for groups (assuming one image per patient for now,
    # but if duplicate patients exist, we need a patient_id.
    # In APTOS 2019, id_code is unique per image.
    # If we had patient ID, we would use it for groups.
    # For now, we treat each image as a group or just use StratifiedKFold if groups are singletons.
    # However, the prompt mentioned "Grouper par id_code (Patient)".
    # Usually id_code IS the image.
    # If there's no patient mapping, we fallback to StratifiedKFold.
    # Let's assume id_code is unique and just use StratifiedKFold logic
    # but implemented via StratifiedGroupKFold with groups=id_code is same as StratifiedKFold.

    df['fold'] = -1

    X = df['id_code']
    y = df['diagnosis']
    groups = df['id_code'] # Singleton groups

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        df.loc[val_idx, 'fold'] = fold

    df.to_csv(output_csv, index=False)
    print(f"Saved folds to {output_csv}")
    print(df['fold'].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/train.csv")
    parser.add_argument("--output", type=str, default="data/train_folds.csv")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if os.path.exists(args.input):
        create_folds(args.input, args.output, args.splits, args.seed)
    else:
        print(f"Input file {args.input} not found.")
