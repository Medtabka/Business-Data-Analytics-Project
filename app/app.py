import sys
import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES

@st.cache_resource
def load_model():
    return joblib.load("models/artifacts/best_model.pkl")

model = load_model()

# Title
st.title("Customer Churn Prediction Dashboard")
st.markdown("Predict churn probability, explore insights, and visualize feature impacts.")

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Home", "Predict Churn", "Explainability"])

if page == "Home":
    st.header("Overview")
    st.write("""
    This Streamlit app demonstrates customer churn prediction using machine learning.
    You can upload your dataset, predict churn, and explore model explainability via SHAP.
    """)

elif page == "Predict Churn":
    st.header("Predict Customer Churn")
    st.write("Upload a CSV or enter customer details manually below:")

    uploaded = st.file_uploader("Upload CSV file", type=["csv", "xlsx"])
    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.write("File uploaded successfully")
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        preds = model.predict_proba(X)[:, 1]
        df["Churn_Probability (%)"] = (preds * 100).round(2)
        st.dataframe(df.head())
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "churn_predictions.csv", "text/csv")

    st.divider()
    st.subheader("Manual Prediction")
    inputs = {}
    for col in NUMERIC_FEATURES:
        inputs[col] = st.number_input(col, min_value=0.0, value=100.0)
    for col in CATEGORICAL_FEATURES:
        inputs[col] = st.text_input(col, "Unknown")

    if st.button("Predict"):
        X_new = pd.DataFrame([inputs])
        prob = model.predict_proba(X_new)[:, 1][0]
        st.success(f"Churn Probability: {prob*100:.2f}%")

elif page == "Explainability":
    st.header("SHAP Explainability")
    st.write("Upload your customer dataset and view global & local churn explanations.")

    uploaded_exp = st.file_uploader("Upload CSV for SHAP analysis", type=["csv"])

    if uploaded_exp:
        df_exp = pd.read_csv(uploaded_exp)
        st.success("File uploaded!")

        needed = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        missing = [c for c in needed if c not in df_exp.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            X = df_exp[needed].copy()

            st.write("Select a customer row index to explain:")

            row_to_explain = st.number_input(
                "Row index", min_value=0, max_value=len(X)-1, value=0
            )

            if st.button("Explain this customer"):
                try:
                    with st.spinner("Computing SHAP explanation..."):

                        prep = model.named_steps["prep"]
                        clf = model.named_steps["clf"]

                        Xp = prep.transform(X)
                        if hasattr(Xp, "toarray"):
                            Xp_dense = Xp.toarray()
                        else:
                            Xp_dense = np.asarray(Xp)

                        feature_names = prep.get_feature_names_out()


                        @st.cache_resource
                        def make_explainer(_clf, reference):
                            return shap.Explainer(clf, reference)

                        # use first 300 samples as reference
                        reference_data = Xp_dense[: min(300, len(Xp_dense))]
                        explainer = make_explainer(clf, reference_data)

                        shap_vals = explainer(Xp_dense, check_additivity=False)

                        sv = getattr(shap_vals, "values", shap_vals)

                        if sv.ndim == 3:
                            sv = sv[:, :, 1]


                        exp_val = explainer.expected_value
                        if isinstance(exp_val, (list, np.ndarray)):
                            base_val = exp_val[-1]
                        else:
                            base_val = exp_val

                        idx = int(row_to_explain)

                        row_expl = shap.Explanation(
                            values=sv[idx],
                            base_values=base_val,
                            data=Xp_dense[idx],
                            feature_names=feature_names
                        )

                        st.subheader(f"Local explanation for customer row {idx}")

                        fig, ax = plt.subplots(figsize=(10, 5))
                        shap.plots.waterfall(row_expl, show=False)
                        st.pyplot(fig)

                        contrib = (
                            pd.DataFrame({"Feature": feature_names, "SHAP Value": sv[idx]})
                            .assign(Impact=lambda d: d["SHAP Value"].abs())
                            .sort_values("Impact", ascending=False)
                            .head(10)
                        )
                        st.subheader("Top factors influencing this prediction")
                        st.dataframe(contrib)

                        st.success("SHAP explanation generated successfully!")

                except Exception as e:
                    st.error(f"Failed to compute SHAP: {e}")

