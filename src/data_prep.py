from typing import Tuple
import pandas as pd
from .config import Columns
from .utils import enforce_dtypes, compute_amount, is_cancelled_invoice

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df = enforce_dtypes(df)
    df = df.dropna(subset=[Columns.customer_id, Columns.invoice_date, Columns.quantity, Columns.unit_price])
    df = df[~is_cancelled_invoice(df[Columns.invoice_no])]
    df = df[(df[Columns.quantity] > 0) & (df[Columns.unit_price] > 0)]
    df["LineAmount"] = compute_amount(df)
    df = df.drop_duplicates()
    return df
