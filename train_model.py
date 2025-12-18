import os
import pandas as pd
import joblib
from src.config import Windows, Columns
from src.data_prep import load_and_clean
from src.labeling import build_datasets
from src.features import split_train_test, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.modeling import train_models
from src.evaluation import (
    evaluate_models,
    compute_permutation_importance,
    plot_feature_importance
)


def main(data_path: str):

    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/tables", exist_ok=True)
    os.makedirs("models/artifacts", exist_ok=True)

    print("\nStarting Churn Prediction Training Pipeline...\n")


    # 2. LOAD & PREPARE DATA

    print("Loading and cleaning dataset...")
    df = load_and_clean(data_path)

    print("Building labeled dataset windows...")
    windows = Windows()
    main_ds = build_datasets(df, windows)

    print("Splitting train/test data...")
    train_df, test_df = split_train_test(main_ds, test_size=0.25, random_state=42)

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df["Churn"]
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df["Churn"]


    # 3. TRAIN MULTIPLE MODELS (LogReg, RF, XGB)

    print("\nTraining models (Logistic Regression, Random Forest, XGBoost)...")
    results = train_models(X_train, y_train)


    # 4. EVALUATE ALL MODELS

    print("\nEvaluating models on test data...")
    summary = evaluate_models(results, X_test, y_test)
    summary.to_csv(r"reports\model_comparison.csv", index=False)
    print("\nModel comparison results saved → reports/tables/model_comparison.csv\n")


    # 5. SELECT BEST MODEL BY AUC SCORE

    best_name = summary.iloc[0]["Model"]
    best_model = results[best_name].best_estimator_
    print(f"Best model based on AUC: {best_name}")

    # Save trained model
    model_path = r"models/artifacts/best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Saved best model → {model_path}")


    # 6. FEATURE IMPORTANCE & INTERPRETABILITY

    print("\nCalculating permutation importances...")
    imp = compute_permutation_importance(best_model, X_test, y_test)
    print(imp.head())

    print("\nPlotting feature importances (for tree-based models)...")
    try:
        feature_names = best_model.named_steps["prep"].get_feature_names_out()
        plot_feature_importance(best_model.named_steps["clf"], feature_names)

    except Exception as e:
        print(f"Skipping feature importance plot (reason: {e})")

    print("\nTraining pipeline complete.")
    print("All outputs are saved inside the 'reports/' and 'models/' folders.")

    # 7. SHAP EXPLAINABILITY
    from src.explainability import explain_model
    main_ds.to_csv("data/processed/main_ds.csv", index=False)
    explain_model(model_path,"data/processed/main_ds.csv")



if __name__ == "__main__":

    data_path = r"data\cleaned\training_data\cleaned_online_retail_sales.csv"

    print(f"Using dataset from: {data_path}")
    main(data_path)

