import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

class OptimizedRounder:
    def __init__(self):
        self.coef_ = [0.5, 1.5, 2.5, 3.5]

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        # Sort coefficients to ensure monotonic thresholds
        coef = np.sort(coef)

        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        coef = np.sort(coef)

        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p.astype(int)

    def coefficients(self):
        return np.sort(self.coef_['x'])

from functools import partial


def main(input_path, output_path):
    df = pd.read_csv(input_path)

    # Filter valid columns
    if 'y_true' not in df.columns or 'y_pred' not in df.columns:
        print("Error: CSV must contain 'y_true' and 'y_pred'")
        return

    X = df['y_pred'].values
    y = df['y_true'].values

    opt = OptimizedRounder()
    opt.fit(X, y)
    coefficients = opt.coefficients()

    print(f"Optimized Thresholds: {coefficients}")

    # Validate
    y_pred_opt = opt.predict(X, coefficients)
    score = quadratic_weighted_kappa(y, y_pred_opt)
    print(f"Optimized QWK: {score:.4f}")

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump({"thresholds": coefficients.tolist()}, f)
    print(f"Saved thresholds to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="oof.csv", help="Path to OOF CSV")
    parser.add_argument("--output", type=str, default="thresholds.json", help="Output JSON path")
    args = parser.parse_args()

    main(args.input, args.output)
