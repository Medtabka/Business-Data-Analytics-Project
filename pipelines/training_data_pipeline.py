import pandas as pd
import os

def clean_sales_data(inp, outp):
    ext = os.path.splitext(inp)[-1].lower()
    print("Reading data...")


    if ext == ".csv":
        df = pd.read_csv(inp)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(inp, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


    before = len(df)
    df = df[df["CustomerID"].notna()]
    df = df[df["Quantity"] > 0]


    os.makedirs(os.path.dirname(outp), exist_ok=True)
    if outp.endswith(".csv"):
        df.to_csv(outp, index=False)
    else:
        df.to_excel(outp, index=False, engine="openpyxl")

    print(f"Cleaned data saved to {outp} ({before - len(df)} rows removed)")

if __name__ == "__main__":     

    infile = r"data\raw\online_retail_sales.csv"
    outfile = r"data\cleaned\training_data\cleaned_online_retail_sales.csv"

    clean_sales_data(infile, outfile)
