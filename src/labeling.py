from datetime import timedelta
import pandas as pd
from .config import Windows, Columns

def window_filter(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    m = df[Columns.invoice_date].between(start, end, inclusive="both")
    return df.loc[m].copy()

def aggregate_features(df_feat: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    g = df_feat.groupby(Columns.customer_id)
    agg = g.agg(
        FirstPurchase=(Columns.invoice_date, "min"),
        LastPurchase=(Columns.invoice_date, "max"),
        Frequency=(Columns.invoice_no, "nunique"),
        Monetary=("LineAmount", "sum"),
        UniqueProducts=(Columns.stock_code, "nunique"),
        TotalQty=(Columns.quantity, "sum"),
        TotalLines=("InvoiceNo", "count"),
        Country=(Columns.country, lambda x: x.mode().iloc[0] if len(x.mode()) else "Unknown"),
    ).reset_index()

    agg["RecencyDays"] = (asof - agg["LastPurchase"]).dt.days
    agg["TenureDays"] = (asof - agg["FirstPurchase"]).dt.days
    agg["AvgBasketValue"] = agg["Monetary"] / agg["Frequency"].clip(lower=1)
    agg["AvgBasketSize"] = agg["TotalQty"] / agg["Frequency"].clip(lower=1)
    agg["ReturnsRatio"] = 0.0
    agg["Momentum30d"] = 0.0
    return agg

def label_churn(df_label: pd.DataFrame, label_start: pd.Timestamp, label_end: pd.Timestamp) -> pd.DataFrame:
    purchases = df_label.groupby(Columns.customer_id).size().rename("FuturePurchases").reset_index()
    purchases["Churn"] = (purchases["FuturePurchases"] == 0).astype(int)
    return purchases[[Columns.customer_id, "Churn"]]

def build_datasets(df: pd.DataFrame, w: Windows):
    feature_end = pd.Timestamp(w.feature_end_main)
    feature_start = df[Columns.invoice_date].min().normalize()
    label_start = feature_end + pd.Timedelta(days=1)
    label_end = feature_end + pd.Timedelta(days=w.label_days)

    df_feat = window_filter(df, feature_start, feature_end)
    X_main = aggregate_features(df_feat, asof=feature_end)

    df_label = window_filter(df, label_start, label_end)
    y_main = label_churn(df_label, label_start, label_end)

    main = X_main.merge(y_main, on=Columns.customer_id, how="left").fillna({"Churn": 1})
    return main
