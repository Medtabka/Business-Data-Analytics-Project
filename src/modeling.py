from typing import Dict, Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from .features import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def make_preprocessor():
    """Builds a preprocessing pipeline for numeric and categorical features."""
    num = Pipeline([("scaler", StandardScaler())])
    cat = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))])

    return ColumnTransformer([
        ("num", num, NUMERIC_FEATURES),
        ("cat", cat, CATEGORICAL_FEATURES),
    ])

def build_grids():
    """Defines model pipelines and their parameter grids for tuning."""
    lr = LogisticRegression(max_iter=200, class_weight="balanced", solver="liblinear")
    rf = RandomForestClassifier(
        n_estimators=400, random_state=42, class_weight="balanced"
    )
    xgb = XGBClassifier(
        random_state=42,
        tree_method="hist",
        eval_metric="auc",
        n_estimators=500
    )

    grids = {
        "Logistic Regression": {
            "model": Pipeline([
                ("prep", make_preprocessor()),
                ("clf", lr)
            ]),
            "param_grid": {"clf__C": [0.1, 1.0, 10.0]},
        },
        "Random Forest": {
            "model": Pipeline([
                ("prep", make_preprocessor()),
                ("clf", rf)
            ]),
            "param_grid": {
                "clf__max_depth": [6, 10, None],
                "clf__min_samples_split": [2, 10],
                "clf__min_samples_leaf": [1, 5],
            },
        },
        "XGBoost": {
            "model": Pipeline([
                ("prep", make_preprocessor()),
                ("clf", xgb)
            ]),
            "param_grid": {
                "clf__max_depth": [4, 6, 8],
                "clf__learning_rate": [0.05, 0.1],
                "clf__subsample": [0.7, 1.0],
                "clf__colsample_bytree": [0.7, 1.0],
                "clf__scale_pos_weight": [1.0, 2.0, 3.0],
            },
        },
    }
    return grids

def train_models(X: pd.DataFrame, y: pd.Series, cv_splits=5) -> Dict[str, Any]:
    """Trains multiple models using GridSearchCV and returns fitted objects."""
    results = {}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for name, spec in build_grids().items():
        print(f"\nTraining model: {name}")
        gs = GridSearchCV(
            estimator=spec["model"],
            param_grid=spec["param_grid"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        gs.fit(X, y)
        results[name] = gs
        print(f"Completed: {name}")
        print("-" * 60)

    return results
