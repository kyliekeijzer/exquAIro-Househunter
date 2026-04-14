import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, accuracy_score, \
    roc_auc_score


def compute_metrics(y_true, y_pred, model=None):
    """
    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        model:  Fitted statsmodels OLS model (required for AIC/BIC)
    """
    mse = mean_squared_error(y_true, y_pred)

    metrics = {
        "r2":   round(r2_score(y_true, y_pred), 4),
        "mse":  round(mse, 4),
        "rmse": round(np.sqrt(mse), 4),
        "mae":  round(mean_absolute_error(y_true, y_pred), 4),
    }

    # AIC/BIC are model-level stats — only available from statsmodels
    if model is not None:
        metrics["aic"] = np.round(model.aic, 4)
        metrics["bic"] = np.round(model.bic, 4)

    return metrics


def compute_classification_metrics(y_true, y_pred, y_prob=None, model=None):
    """
    Args:
        y_true: Actual target values (binary 0/1)
        y_pred: Predicted class labels (0/1)
        y_prob: Predicted probabilities (required for AUC-ROC)
        model:  Fitted statsmodels Logit model (required for AIC/BIC)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy":     round(accuracy_score(y_true, y_pred), 4),
        "sensitivity":  round(tp / (tp + fn), 4),  # True Positive Rate / Recall
        "specificity":  round(tn / (tn + fp), 4),  # True Negative Rate
        "confusion_matrix": cm,
    }

    if y_prob is not None:
        metrics["auc_roc"] = round(roc_auc_score(y_true, y_prob), 4)

    if model is not None:
        metrics["aic"] = np.round(model.aic, 4)
        metrics["bic"] = np.round(model.bic, 4)

    return metrics

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true-y_pred)/y_true))*100
