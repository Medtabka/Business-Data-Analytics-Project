from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class Windows:
    feature_end_main: datetime = datetime(2011, 9, 9)
    label_days: int = 90
    train_feature_end: datetime = datetime(2011, 6, 1)

@dataclass(frozen=True)
class Columns:
    invoice_no: str = "InvoiceNo"
    stock_code: str = "StockCode"
    description: str = "Description"
    quantity: str = "Quantity"
    invoice_date: str = "InvoiceDate"
    unit_price: str = "UnitPrice"
    customer_id: str = "CustomerID"
    country: str = "Country"

DATE_FORMAT = "%Y-%m-%d"
RANDOM_STATE = 42

