#################################
# Imports
#################################
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error
except Exception:  # pragma: no cover
    root_mean_squared_error = None
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import dump, load as joblib_load


#################################
# Pipeline Construction
#################################
def build_pipeline(model_name: str, numeric_cols, categorical_cols) -> Pipeline:
    model_name = model_name.lower()

    if model_name == "linear":
        model = LinearRegression()
    elif model_name == "ridge":
        model = Ridge(alpha=1.0, random_state=42)
    else:
        raise ValueError("Unsupported model. Choose 'linear' or 'ridge'.")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_cols)),
            ("cat", categorical_transformer, list(categorical_cols))
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipe


#################################
# Column Type Detection
#################################
def _split_columns(X: pd.DataFrame):
    numeric_cols = list(X.select_dtypes(include=["number"]).columns)
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


#################################
# Model Evaluation
#################################
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    if root_mean_squared_error is not None:
        rmse = root_mean_squared_error(y_true, y_pred)
    else:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


#################################
# Main Training & Evaluation Runner
#################################
def run(csv_path: Path, target: str, test_size: float, random_state: int, model_name: str,
    save_model: Optional[Path] = None, save_metrics: Optional[Path] = None) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")

    y = df[target]
    X = df.drop(columns=[target])

    feature_columns = list(X.columns)
    numeric_cols, categorical_cols = _split_columns(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipe = build_pipeline(model_name, numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    metrics = evaluate(y_test, preds)

    if save_model:
        bundle = {
            "pipeline": pipe,
            "feature_columns": feature_columns,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "model_name": model_name,
            "target": target,
        }
        dump(bundle, save_model)
    if save_metrics:
        save_metrics.write_text(pd.Series(metrics).to_json(indent=2))

    return metrics


#################################
# Convenience Wrapper for Training
#################################
def train_and_save(csv_path: Path = Path("nexus_hours_regression_data.csv"),
                   target: str = "hours",
                   model_name: str = "linear",
                   test_size: float = 0.2,
                   random_state: int = 42,
                   save_model: Path = Path("model.joblib")) -> Dict[str, float]:
    metrics = run(
        csv_path=csv_path,
        target=target,
        test_size=test_size,
        random_state=random_state,
        model_name=model_name,
        save_model=save_model,
        save_metrics=None,
    )
    return metrics


#################################
# Model Loading
#################################
def load_model_bundle(path: Path) -> Dict[str, Any]:
    return joblib_load(path)


#################################
# Prediction Helper
#################################
def predict_with_model(bundle: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    feature_columns = bundle["feature_columns"]
    pipe: Pipeline = bundle["pipeline"]

    X = df.reindex(columns=feature_columns)
    return pipe.predict(X)


#################################
# CLI Entry Point
#################################
def main():
    parser = argparse.ArgumentParser(description="Run regression on nexus hours dataset.")
    parser.add_argument("--csv", type=str, default="nexus_hours_regression_data.csv",
                        help="Path to CSV file (default: nexus_hours_regression_data.csv)")
    parser.add_argument("--target", type=str, default="hours",
                        help="Target column name (default: hours)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test size fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--model", type=str, default="linear", choices=["linear", "ridge"],
                        help="Model to use (default: linear)")
    parser.add_argument("--save-model", type=str, default=None,
                        help="Optional path to save trained model (joblib)")
    parser.add_argument("--save-metrics", type=str, default=None,
                        help="Optional path to save metrics (json)")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    save_model = Path(args.save_model) if args.save_model else None
    save_metrics = Path(args.save_metrics) if args.save_metrics else None

    metrics = run(
        csv_path=csv_path,
        target=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        model_name=args.model,
        save_model=save_model,
        save_metrics=save_metrics,
    )

    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"- {k.upper()}: {v:.4f}")


if __name__ == "__main__":
    main()
