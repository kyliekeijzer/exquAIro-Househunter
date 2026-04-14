import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.tree import plot_tree

def plot_predicted_vs_actual(y_test, y_test_pred, save_plot=False, title=None):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(y_test, y_test_pred, alpha=0.4, marker='o')
    ax.plot((0, max(y_test)), (0, max(y_test)), 'r-', label='Real=Predicted')
    ax.set_xlabel('Real values')
    ax.set_ylabel('Predictions')
    ax.legend(loc='upper left')

    if title:
        ax.set_title(title)

    if save_plot:
        filename = f"{title}.png" if title else "predicted_vs_actual.png"
        fig.savefig(filename, dpi=300)

    plt.close(fig)
    return fig


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues,
                          save_plot=False, title=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, interpolation='nearest', cmap=cmap)

    if title:
        ax.set_title(title)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()

    if save_plot:
        filename = f"{title}.png" if title else "confusion_matrix.png"
        fig.savefig(filename, dpi=300)

    plt.close(fig)
    return fig

def plot_roc_curve(y_true, y_prob, save_plot=False, title=None):
    """
    Plots the ROC curve for a binary classifier.

    Args:
        y_true: Actual binary target values (0/1)
        y_prob: Predicted probabilities for the positive class
        save_plot: Whether to save the plot to a file
        title: Plot title and filename if saved
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Random classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.legend(loc='lower right')

    if title:
        ax.set_title(title)

    if save_plot:
        filename = f"{title}.png" if title else "roc_curve.png"
        fig.savefig(filename, dpi=300)

    plt.close(fig)
    return fig

def plot_decision_tree(model, feature_names, class_names=None,
                       max_depth=20, save_plot=False, title=None):
    """
    Plots the decision tree structure.

    Args:
        model:         Fitted DecisionTreeClassifier
        feature_names: List of feature names (e.g. X_train.columns.tolist())
        class_names:   List of class labels (e.g. ["Low Price", "High Price"])
        max_depth:     How many levels to display (None = full tree, can get large)
        save_plot:     Whether to save the plot to a file
        title:         Plot title and filename if saved
    """
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        max_depth=max_depth,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax,
    )

    if title:
        ax.set_title(title)

    if save_plot:
        filename = f"{title}.png" if title else "decision_tree.png"
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    plt.close(fig)
    return fig