import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from Scripts.performance_metrics_functions import compute_metrics, compute_classification_metrics


def perform_linear_regression(train_data_X, train_data_Y,
                              test_data_X, test_data_Y):
    """
        Performs linear regression using statsmodels and returns performance metrics.

        Args:
            train_data_X: Training features (DataFrame or ndarray)
            train_data_Y: Training target (Series or ndarray)
            test_data_X:  Test features (DataFrame or ndarray)
            test_data_Y:  Test target (Series or ndarray)

        Returns:
            dict with the fitted model, summary, and performance metrics
        """
    # Add constant (intercept) to features
    X_train = sm.add_constant(train_data_X)
    X_test = sm.add_constant(test_data_X)

    # Fit model
    model = sm.OLS(train_data_Y, X_train).fit()

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return {
        "model": model,
        "train_predictions": y_train_pred,
        "test_predictions": y_test_pred,
        "summary": model.summary(),  # full OLS summary table
        "coefficients": model.params.to_dict(),
        "p_values": model.pvalues.to_dict(),  # significance of each feature
        "train_metrics": compute_metrics(train_data_Y, y_train_pred, model),
        "test_metrics": compute_metrics(test_data_Y, y_test_pred, model),
    }

# Logistic regression
def perform_logistic_regression(train_data_X, train_data_Y,
                                test_data_X, test_data_Y):
    """
    Performs logistic regression using statsmodels and returns performance metrics.

    Args:
        train_data_X: Training features (DataFrame or ndarray)
        train_data_Y: Training target (Series or ndarray) — binary (0/1)
        test_data_X:  Test features (DataFrame or ndarray)
        test_data_Y:  Test target (Series or ndarray)

    Returns:
        dict with the fitted model, summary, and performance metrics
    """
    # Add constant (intercept) to features
    X_train = sm.add_constant(train_data_X)
    X_test = sm.add_constant(test_data_X)

    # Fit model
    model = sm.Logit(train_data_Y, X_train).fit()

    # Predicted probabilities
    y_train_prob = model.predict(X_train)
    y_test_prob = model.predict(X_test)

    # Predicted classes (threshold = 0.5)
    y_train_pred = (y_train_prob >= 0.5).astype(int)
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    # Odds Ratios with 95% CI (exponentiate coefficients and confidence intervals)
    conf_int = model.conf_int()
    odds_ratios = pd.DataFrame({
        "OR":    np.exp(model.params),
        "CI_lower": np.exp(conf_int[0]),
        "CI_upper": np.exp(conf_int[1]),
    }).round(4)

    return {
        "model": model,
        "train_predictions": y_train_pred,
        "test_predictions": y_test_pred,
        "train_probs": y_train_prob,
        "test_probs": y_test_prob,
        "summary": model.summary(),
        "coefficients": model.params.to_dict(),
        "p_values": model.pvalues.to_dict(),
        "odds_ratios": odds_ratios,
        "train_metrics": {
            **compute_classification_metrics(train_data_Y, y_train_pred, y_train_prob, model),
            **compute_metrics(train_data_Y, y_train_prob, model),
        },
        "test_metrics": {
            **compute_classification_metrics(test_data_Y, y_test_pred, y_test_prob, model),
            **compute_metrics(test_data_Y, y_test_prob, model),
        },
    }

def perform_decision_tree(train_data_X, train_data_Y,
                          test_data_X, test_data_Y,
                          max_depth=None, random_state=42):
    """
    Performs decision tree classification using sklearn and returns performance metrics.

    Args:
        train_data_X:  Training features (DataFrame or ndarray)
        train_data_Y:  Training target (Series or ndarray) — binary (0/1)
        test_data_X:   Test features (DataFrame or ndarray)
        test_data_Y:   Test target (Series or ndarray)
        max_depth:     Maximum depth of the tree (None = unlimited)
        random_state:  Random seed for reproducibility

    Returns:
        dict with the fitted model, predictions, and performance metrics
    """
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(train_data_X, train_data_Y)

    # Predicted classes
    y_train_pred = model.predict(train_data_X)
    y_test_pred  = model.predict(test_data_X)

    # Predicted probabilities (for AUC-ROC)
    y_train_prob = model.predict_proba(train_data_X)[:, 1]
    y_test_prob  = model.predict_proba(test_data_X)[:, 1]

    # Feature importances
    feature_importances = pd.Series(
        model.feature_importances_, index=train_data_X.columns
    ).sort_values(ascending=False).round(4)

    return {
        "model": model,
        "train_predictions": y_train_pred,
        "test_predictions":  y_test_pred,
        "train_probs": y_train_prob,
        "test_probs":  y_test_prob,
        "feature_importances": feature_importances,
        "train_metrics": {
            **compute_classification_metrics(train_data_Y, y_train_pred, y_train_prob),
            **compute_metrics(train_data_Y, y_train_prob),
        },
        "test_metrics": {
            **compute_classification_metrics(test_data_Y, y_test_pred, y_test_prob),
            **compute_metrics(test_data_Y, y_test_prob),
        },
    }