import numpy as np
import pandas as pd

def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", utc=False)
    for c in ("Quantity", "UnitPrice"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_amount(df: pd.DataFrame) -> pd.Series:
    return df["Quantity"] * df["UnitPrice"]

def is_cancelled_invoice(s: pd.Series) -> pd.Series:
    return s.astype(str).str.startswith("C", na=False)

def iqr_winsorize(s: pd.Series, q1=0.25, q3=0.75, k=3.0) -> pd.Series:
    x = s.copy()
    Q1, Q3 = x.quantile([q1, q3])
    iqr = Q3 - Q1
    low, high = Q1 - k * iqr, Q3 + k * iqr
    return x.clip(lower=low, upper=high)
