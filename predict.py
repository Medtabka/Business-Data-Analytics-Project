import pandas as pd
import joblib

model = joblib.load(r"models\artifacts\best_model.pkl")

data = pd.read_csv(r"data\cleaned\test_data\cleaned_test_sales_data.csv")

X = data.drop(columns=["CustomerID", "Churn"], errors="ignore")

data["Churn_Probability"] = model.predict_proba(X)[:, 1]
data["Churn_Prediction"] = model.predict(X)

data = data.sort_values(by="Churn_Probability", ascending=False)

output_path = r"reports\tables\churn_predictions.csv"
data.to_csv(output_path, index=False)

print(f"Batch churn prediction completed. File saved at: {output_path}")

