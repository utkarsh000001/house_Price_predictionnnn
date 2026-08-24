"""
Trains and evaluates multiple regression models on the housing dataset,
picks the best one by test-set R², and saves it along with its metrics
and feature importances.

Fixes vs. original:
- Path bug: original looked for 'Dataset/house_data.csv' relative to
  BASE_DIR, but the file lived at the repo root. This version resolves
  the path correctly no matter where the script is run from.
- Adds an actual train/test split + metrics (R², MAE, RMSE) instead of
  just fitting silently.
- Adds a house_age feature (derived from YearBuilt) since "how old is
  this house" is usually more predictive than the raw build year.
- Compares Linear Regression, Random Forest, and Gradient Boosting,
  and keeps the best performer instead of hardcoding Random Forest.
- Saves the model with joblib (sklearn's recommended format, faster
  and safer than raw pickle for numpy-heavy objects).
- Saves metrics.json and feature_importance.json so the app can
  display them instead of just outputting a single number.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "house_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "Model")
FEATURES = ["SquareFeet", "Bedrooms", "Bathrooms", "HouseAge", "Neighborhood_Score"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["HouseAge"] = 2026 - df["YearBuilt"]
    X = df[FEATURES]
    y = df["Price"]
    return X, y


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "r2": round(r2_score(y_test, preds), 4),
        "mae": round(mean_absolute_error(y_test, preds), 2),
        "rmse": round(root_mean_squared_error(y_test, preds), 2),
    }


def train_and_save_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }

    results = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        print(f"{name}: {results[name]}")

    best_name = max(results, key=lambda n: results[n]["r2"])
    best_model = candidates[best_name]
    print(f"\n🏆 Best model: {best_name} (R²={results[best_name]['r2']})")

    # Feature importance (RF/GB expose it directly; for Linear use |coef|)
    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(FEATURES, best_model.feature_importances_.round(4)))
    else:
        coefs = np.abs(best_model.coef_)
        importances = dict(zip(FEATURES, (coefs / coefs.sum()).round(4)))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "house_model.joblib"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "all_results": results}, f, indent=2)

    with open(os.path.join(MODEL_DIR, "feature_importance.json"), "w") as f:
        json.dump({k: float(v) for k, v in importances.items()}, f, indent=2)

    print(f"🎉 Saved model, metrics, and feature importances to {MODEL_DIR}")


if __name__ == "__main__":
    train_and_save_model()
