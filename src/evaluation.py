from typing import Dict, Any
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, RocCurveDisplay, classification_report
)
from sklearn.inspection import permutation_importance


def evaluate_models(results: Dict[str, Any], X_test, y_test) -> pd.DataFrame:
    """
    Evaluate multiple trained models on the test set and generate comparison metrics.
    Returns a DataFrame with metrics: AUC, Accuracy, F1, Precision, Recall.
    """
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/tables", exist_ok=True)

    print("\n===== MODEL EVALUATION SUMMARY =====\n")
    summary = []

    for name, grid in results.items():
        best_model = grid.best_estimator_
        proba = best_model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "Model": name,
            "AUC": roc_auc_score(y_test, proba),
            "Accuracy": accuracy_score(y_test, preds),
            "F1": f1_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
        }
        summary.append(metrics)

        # Print classification report
        print(f"Model: {name}")
        print(classification_report(y_test, preds))
        print(f"AUC-ROC: {metrics['AUC']:.4f}")
        print("-" * 60)

        # Confusion matrix visualization
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", color="black")
        plt.tight_layout()
        plt.savefig(f"reports/figures/confusion_matrix_{name.replace(' ', '_')}.png")
        plt.close(fig)

        # ROC Curve
        RocCurveDisplay.from_estimator(best_model, X_test, y_test)
        plt.title(f"ROC Curve - {name}")
        plt.tight_layout()
        plt.savefig(f"reports/figures/roc_curve_{name.replace(' ', '_')}.png")
        plt.close()

    df_summary = pd.DataFrame(summary).sort_values(by="AUC", ascending=False)
    df_summary.to_csv("reports/tables/model_comparison.csv", index=False)
    print("\nSaved model comparison metrics → reports/tables/model_comparison.csv\n")

    return df_summary

def compute_permutation_importance(best_model, X_test, y_test, n_repeats=10) -> pd.DataFrame:
    """
    Computes permutation feature importance for the best trained model.
    Handles pipelines with one-hot encoders and ensures arrays match in length.
    """
    os.makedirs("reports/tables", exist_ok=True)
    print("\nCalculating permutation importance...")

    r = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring="roc_auc"
    )

    # Safely extract feature names
    try:
        names = best_model.named_steps["prep"].get_feature_names_out()
    except Exception:
        names = [f"feature_{i}" for i in range(len(r.importances_mean))]

    # Handle mismatched lengths
    n = len(r.importances_mean)
    if len(names) > n:
        names = names[:n]
    elif len(names) < n:
        names += [f"extra_{i}" for i in range(n - len(names))]

    imp = pd.DataFrame({
        "Feature": names,
        "Importance": r.importances_mean
    }).sort_values("Importance", ascending=False)

    imp.to_csv("reports/tables/permutation_importance.csv", index=False)
    print("Saved feature importance → reports/tables/permutation_importance.csv\n")

    return imp

def plot_feature_importance(model, feature_names):
    """
    Plots feature importances for tree-based models like Random Forest or XGBoost.
    """
    os.makedirs("reports/figures", exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(importances)), importances[indices], color="skyblue")
        plt.xticks(range(len(importances)),
                   [feature_names[i] for i in indices],
                   rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("reports/figures/feature_importance.png")
        plt.show()
    else:
        print("This model does not support feature importances.")
