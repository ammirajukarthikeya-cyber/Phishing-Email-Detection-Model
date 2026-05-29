"""
Utility functions for display formatting and visualization.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# --- Color Constants ----------------------------------------------------------

COLORS = {
    "header": "\033[95m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "bold": "\033[1m",
    "underline": "\033[4m",
    "end": "\033[0m",
}


def print_banner():
    """Print the application banner."""
    banner = f"""
{COLORS['cyan']}{COLORS['bold']}
 +==============================================================+
 |        PHISHING EMAIL DETECTION MODEL                        |
 |        --------------------------------------                |
 |        Machine Learning Classification System                |
 +==============================================================+
{COLORS['end']}"""
    print(banner)


def print_section(title, icon=">>"):
    """Print a formatted section header."""
    sep = "-" * (len(title) + 4)
    print(f"\n{COLORS['yellow']}{COLORS['bold']}{icon} {title}{COLORS['end']}")
    print(f"{COLORS['yellow']}{sep}{COLORS['end']}")


def print_metric(name, value, color="green"):
    """Print a formatted metric."""
    c = COLORS.get(color, COLORS["green"])
    print(f"  {COLORS['bold']}{name}:{COLORS['end']} {c}{value}{COLORS['end']}")


def plot_confusion_matrix(y_true, y_pred, output_dir="outputs"):
    """
    Plot and save a styled confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_dir: Directory to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Safe", "Phishing"]

    plt.figure(figsize=(8, 6))
    sns.set_style("darkgrid")

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=2,
        linecolor="white",
        annot_kws={"size": 18, "weight": "bold"},
        cbar_kws={"label": "Count"},
    )

    plt.title("Confusion Matrix - Phishing Detection", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Predicted Label", fontsize=13, fontweight="bold")
    plt.ylabel("True Label", fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Confusion matrix saved to: {filepath}")


def plot_feature_importance(importances, feature_names, top_n=15, output_dir="outputs"):
    """
    Plot and save the top N most important features.

    Args:
        importances: Feature importance values from the model.
        feature_names: Names of the features.
        top_n: Number of top features to display.
        output_dir: Directory to save the plot.
    """
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_n))

    bars = plt.barh(range(top_n), top_importances[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
    plt.yticks(range(top_n), top_features[::-1], fontsize=11)
    plt.xlabel("Importance Score", fontsize=13, fontweight="bold")
    plt.title(f"Top {top_n} Feature Importances", fontsize=16, fontweight="bold", pad=15)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Feature importance saved to: {filepath}")


def display_results(y_true, y_pred, importances=None, feature_names=None, output_dir="outputs"):
    """
    Display full evaluation results: accuracy, classification report,
    confusion matrix, and feature importances.
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Safe", "Phishing"])

    # Accuracy
    print_section("Model Accuracy", "[ACCURACY]")
    color = "green" if acc >= 0.9 else "yellow" if acc >= 0.8 else "red"
    print_metric("Accuracy", f"{acc:.4f}  ({acc * 100:.2f}%)", color)

    # Classification Report
    print_section("Classification Report", "[REPORT]")
    print(report)

    # Confusion Matrix
    print_section("Confusion Matrix", "[MATRIX]")
    cm = confusion_matrix(y_true, y_pred)
    print(f"                  Predicted")
    print(f"                Safe    Phishing")
    print(f"  Actual Safe   {cm[0][0]:<8}{cm[0][1]}")
    print(f"  Actual Phish  {cm[1][0]:<8}{cm[1][1]}")
    plot_confusion_matrix(y_true, y_pred, output_dir)

    # Feature Importance
    if importances is not None and feature_names is not None:
        print_section("Feature Importance", "[FEATURES]")
        plot_feature_importance(importances, feature_names, output_dir=output_dir)
        top_5_idx = np.argsort(importances)[::-1][:5]
        for i, idx in enumerate(top_5_idx, 1):
            print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
