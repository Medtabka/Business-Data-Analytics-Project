import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def explain_model(model_path, sample_data_path):
    """Generates SHAP explainability plots and saves top churn-driving features."""


    model = joblib.load(model_path)
    df = pd.read_csv(sample_data_path)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()


    if len(X) > 500:
        X = X.sample(500, random_state=42)

    # Preprocess features
    X_preprocessed = model.named_steps["prep"].transform(X)
    feature_names = model.named_steps["prep"].get_feature_names_out()

    # Compute SHAP values
    explainer = shap.Explainer(model.named_steps["clf"], X_preprocessed)
    shap_values = explainer(X_preprocessed, check_additivity=False)

    # Handle multi-class outputs
    if hasattr(shap_values, "values"):
        if isinstance(shap_values.values, list):
            shap_values = shap_values[1]
        elif shap_values.values.ndim == 3:
            shap_values = shap_values[..., 1]

    shap.summary_plot(
        shap_values,
        features=X_preprocessed,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig("reports/figures/shap_summary.png")

    # Save single-customer waterfall plot
    try:
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        plt.savefig("reports/figures/shap_customer_example.png")
    except Exception as e:
        print(f"Skipping waterfall plot: {e}")

    # Compute average absolute SHAP importance
    abs_importance = np.abs(shap_values).mean(axis=0)

    # Ensure lengths match before creating DataFrame
    min_len = min(len(feature_names), len(abs_importance))
    feature_names = feature_names[:min_len]
    abs_importance = abs_importance[:min_len]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsSHAP": abs_importance
    }).sort_values("MeanAbsSHAP", ascending=False)
