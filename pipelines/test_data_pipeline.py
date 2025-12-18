import pandas as pd
from datetime import datetime
import os

def make_churn_data(inp, outp):
    print("Loading data...")
    data = pd.read_csv(inp)


    before_len = len(data)
    data = data.dropna(subset=["CustomerID"])
    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)]
    print(f"Cleaned {before_len - len(data)} rows")


    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data["LineAmount"] = data["Quantity"] * data["UnitPrice"]


    cutoff = data["InvoiceDate"].max()
    print(f"Cutoff date: {cutoff.date()}")

    g = data.groupby("CustomerID").agg({
        "InvoiceDate": ["min", "max"],
        "InvoiceNo": "nunique",
        "Quantity": "sum",
        "StockCode": "nunique",
        "LineAmount": "sum",
        "Country": lambda x: x.mode().iloc[0] if len(x.mode()) else "Unknown"
    })

    g.columns = [
        "first_buy", "last_buy", "freq", "total_qty",
        "unique_items", "money", "country"
    ]
    g = g.reset_index()


    g["recency"] = (cutoff - g["last_buy"]).dt.days
    g["tenure"] = (cutoff - g["first_buy"]).dt.days
    g["avg_basket_val"] = g["money"] / g["freq"].clip(lower=1)
    g["avg_basket_size"] = g["total_qty"] / g["freq"].clip(lower=1)
    g["total_lines"] = g["freq"]
    g["returns_ratio"] = 0.0
    g["momentum_30d"] = 0.0


    os.makedirs(os.path.dirname(outp), exist_ok=True)
    g.to_csv(outp, index=False)
    print(f"Saved file to {outp}")

if __name__ == "__main__":

    infile = r"data\raw\test_sales_data.csv"
    outfile = r"data\cleaned\test_data\cleaned_test_sales_data.csv"

    make_churn_data(infile, outfile)
