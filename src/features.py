from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC_FEATURES = [
    "RecencyDays", "Frequency", "Monetary", "UniqueProducts", "TotalQty", "TotalLines",
    "TenureDays", "AvgBasketValue", "AvgBasketSize", "ReturnsRatio", "Momentum30d"
]
CATEGORICAL_FEATURES = ["Country"]

def split_train_test(main: pd.DataFrame, test_size=0.25, random_state=42):
    return train_test_split(main, test_size=test_size, stratify=main["Churn"], random_state=random_state)

