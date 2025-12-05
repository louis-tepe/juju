from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculates the Quadratic Weighted Kappa.
    y_true: 1D array of true labels.
    y_pred: 1D array of predicted labels.
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def get_optimal_thresholds(y_true, y_pred_continuous):
    """
    Finds optimal thresholds to convert continuous predictions to ordinal labels
    maximizing QWK. This is a placeholder for the optimization step.
    """
    # Basic rounding for default
    return [0.5, 1.5, 2.5, 3.5]
