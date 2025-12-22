import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ========= CONFIG =========
RATINGS_CSV = "nba2k26_ratings.csv"
STATS_CSV   = "nba_2425_stats.csv"  # or your actual stats filename

RAW_PLAYER_COL = "Player"   # this is correct based on your columns

# Use the advanced metrics that actually exist in your CSV
FEATURE_COLS_CANDIDATES = [
    "PER",
    "TS%",
    "USG%",
    "OWS",
    "DWS",
    "WS",
    "WS/48",
    "OBPM",
    "DBPM",
    "BPM",
    "VORP",
]
# ==========================

def clean_name_series(s: pd.Series) -> pd.Series:
    """Lowercase, strip accents, remove dots/apostrophes, trim."""
    return (
        s.astype(str)
         .str.lower()
         .str.normalize("NFKD")
         .str.encode("ascii", errors="ignore")
         .str.decode("utf-8")
         .str.replace(r"\.", "", regex=True)
         .str.replace(r"'", "", regex=True)
         .str.strip()
    )


def load_and_merge(ratings_path: str, stats_path: str) -> pd.DataFrame:
    # Load CSVs
    ratings = pd.read_csv(ratings_path)
    stats = pd.read_csv(stats_path)

    # Clean names
    ratings["name_key"] = clean_name_series(ratings["Name"])
    stats["name_key"] = clean_name_series(stats[RAW_PLAYER_COL])

    # Merge on cleaned name
    merged = pd.merge(
        ratings,
        stats,
        on="name_key",
        how="inner",
        suffixes=("_2k", "_stats")
    )

    print(f"Loaded {len(ratings)} 2K players, {len(stats)} stats rows.")
    print(f"Merged dataset has {len(merged)} matched players.")
    return merged


def build_features_and_target(merged: pd.DataFrame):
    # Keep only features that actually exist in the merged dataset
    feature_cols = [c for c in FEATURE_COLS_CANDIDATES if c in merged.columns]
    if not feature_cols:
        raise ValueError(
            f"None of the expected feature columns were found. "
            f"Available columns include: {list(merged.columns)[:40]}"
        )

    print("Using feature columns:", feature_cols)

    X = merged[feature_cols].astype(float)
    y = merged["Overall_Rating"].astype(float)

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, feature_cols


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=42
        )
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = {
            "model": model,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse
        }
        print(f"\n=== {name} ===")
        print(f"R2:   {r2:.3f}")
        print(f"MAE:  {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")

    return results


def plot_feature_importance(rf_model: RandomForestRegressor, feature_cols):
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    plt.figure(figsize=(6, 4))
    plt.bar(
        range(len(sorted_idx)),
        importances[sorted_idx],
        tick_label=np.array(feature_cols)[sorted_idx]
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.show()


def plot_pred_vs_actual(model, X_test, y_test):
    y_pred = model.predict(X_test)

    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--")
    plt.xlabel("Actual 2K Rating")
    plt.ylabel("Predicted 2K Rating")
    plt.title("Predicted vs Actual 2K Ratings")
    plt.tight_layout()
    plt.show()


def main():
    merged = load_and_merge(RATINGS_CSV, STATS_CSV)
    X_train, X_test, y_train, y_test, feature_cols = build_features_and_target(merged)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Choose a model for plots (RandomForest by default)
    rf = results["RandomForest"]["model"]
    plot_feature_importance(rf, feature_cols)
    plot_pred_vs_actual(rf, X_test, y_test)


if __name__ == "__main__":
    main()

