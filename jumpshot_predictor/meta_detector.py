from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parent
SHOTS_PATH = DATA_DIR / "shots.csv"
MODEL_PATH = DATA_DIR / "jumpshot_model.pkl"
GREEN_MODEL_PATH = DATA_DIR / "jumpshot_green_model.pkl"
LEADERBOARD_PATH = DATA_DIR / "exploit_leaderboard.csv"
PATCH_IMPACT_PATH = DATA_DIR / "patch_impact.csv"
PATCH_ALERTS_PATH = DATA_DIR / "patch_alerts.csv"
CHART_PATH = DATA_DIR / "exploit_chart.png"

CAT_COLS = ["base", "release1", "release2", "speed", "patch"]
NUM_COLS = ["player_3pt"]
EXPLOIT_COLS = ["base", "release1", "release2", "speed"]


def load_shots(path: Path = SHOTS_PATH) -> pd.DataFrame:
    """Load tracked shot sessions and derive real make/green percentages.

    Args:
        path: CSV path containing tracked shot sessions.

    Returns:
        Dataframe with make_pct and green_pct columns.
    """
    shots = pd.read_csv(path)
    required = {"attempts", "makes", "greens"}
    missing = required - set(shots.columns)
    if missing:
        raise ValueError(f"shots.csv missing required columns: {', '.join(sorted(missing))}")
    if shots["greens"].isna().any():
        raise ValueError("greens must be tracked for every session")

    shots["make_pct"] = shots["makes"] / shots["attempts"]
    shots["green_pct"] = shots["greens"] / shots["attempts"]
    return shots


def build_model() -> Pipeline:
    """Build the XGBoost preprocessing and regression pipeline.

    Returns:
        Configured scikit-learn pipeline.
    """
    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError(
            "XGBoost could not be loaded. On macOS, install OpenMP with "
            "`brew install libomp`, then rerun the training script."
        ) from exc

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocess = ColumnTransformer(
        [
            ("cat", encoder, CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ]
    )

    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=120,
        max_depth=2,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ("preprocess", preprocess),
            ("xgb", xgb),
        ]
    )


def train_baseline(shots: pd.DataFrame, target: str = "make_pct") -> tuple[Pipeline, dict[str, float]]:
    """Train a weighted baseline model for a target percentage.

    Args:
        shots: Shot session dataframe.
        target: Target column to predict.

    Returns:
        Fitted model and evaluation metrics.
    """
    X = shots[CAT_COLS + NUM_COLS]
    y = shots[target]

    gss = GroupShuffleSplit(test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=shots["date"]))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    weights = shots.iloc[train_idx]["attempts"]

    model = build_model()
    model.fit(X_train, y_train, xgb__sample_weight=weights)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    metrics = {
        "train_r2": r2_score(y_train, train_pred),
        "test_r2": r2_score(y_test, test_pred),
        "test_mae": mean_absolute_error(y_test, test_pred),
    }
    return model, metrics


def build_exploit_leaderboard(shots: pd.DataFrame) -> pd.DataFrame:
    """Build exploit and latency leaderboard rows from model residuals.

    Args:
        shots: Shot session dataframe with residual columns.

    Returns:
        Ranked leaderboard dataframe.
    """
    exploits = (
        shots.groupby(EXPLOIT_COLS)
        .agg(
            total_attempts=("attempts", "sum"),
            total_makes=("makes", "sum"),
            avg_residual=("residual", "mean"),
            avg_green_residual=("green_residual", "mean"),
            residual_std=("residual", "std"),
            n_sessions=("date", "count"),
            total_greens=("greens", "sum"),
        )
        .reset_index()
    )
    exploits = exploits[exploits["total_attempts"] >= 50].copy()
    exploits["exploit_score"] = (
        exploits["avg_residual"]
        * np.log1p(exploits["total_attempts"])
        * np.sqrt(exploits["n_sessions"])
    )
    exploits["make_pct"] = exploits["total_makes"] / exploits["total_attempts"]
    exploits["green_pct"] = exploits["total_greens"] / exploits["total_attempts"]
    exploits["latency_score"] = (
        exploits["green_pct"]
        / exploits["make_pct"].replace(0, np.nan)
    ) * np.log1p(exploits["total_attempts"])
    exploits["latency_score"] = exploits["latency_score"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return exploits.sort_values("exploit_score", ascending=False)


def build_patch_impact(shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build patch impact rows and significant patch alerts.

    Args:
        shots: Shot session dataframe with residual columns.

    Returns:
        Full patch impact dataframe and alert dataframe.
    """
    patch_perf = (
        shots.groupby(["patch"] + EXPLOIT_COLS)
        .agg(
            mean_make_pct=("make_pct", "mean"),
            mean_green_pct=("green_pct", "mean"),
            total_attempts=("attempts", "sum"),
            avg_residual=("residual", "mean"),
            avg_green_residual=("green_residual", "mean"),
            total_greens=("greens", "sum"),
        )
        .reset_index()
        .sort_values("patch")
    )

    grouped = patch_perf.groupby(EXPLOIT_COLS)["mean_make_pct"]
    patch_perf["rolling_mean"] = grouped.transform(lambda x: x.shift(1).rolling(2).mean())
    patch_perf["rolling_std"] = grouped.transform(lambda x: x.shift(1).rolling(2).std())
    patch_perf["z_score"] = (
        patch_perf["mean_make_pct"] - patch_perf["rolling_mean"]
    ) / patch_perf["rolling_std"]

    patch_alerts = patch_perf[
        (patch_perf["total_attempts"] >= 50) & (patch_perf["z_score"].abs() > 2)
    ]
    return patch_perf, patch_alerts


def save_chart(exploits: pd.DataFrame, path: Path = CHART_PATH) -> None:
    """Save a leaderboard chart image.

    Args:
        exploits: Leaderboard dataframe.
        path: PNG path to write.

    Returns:
        None.
    """
    top_exploits = exploits.nlargest(10, "exploit_score")
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top_exploits)), top_exploits["exploit_score"], color="#246bfe")
    plt.yticks(
        range(len(top_exploits)),
        [
            f"{row['base']}/{row['release1']}/{row['release2']}"
            for _, row in top_exploits.iterrows()
        ],
    )
    plt.xlabel("Exploit Score")
    plt.title("Top 10 Meta Jumpshots")
    plt.axvline(0, color="#d94b4b", linestyle="--")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def run() -> dict[str, float]:
    """Train make and green models, refresh leaderboard outputs, and save charts.

    Returns:
        Evaluation metrics for both target models.
    """
    shots = load_shots()
    print("Real green counts loaded from shots.csv")
    print(f"Loaded {len(shots)} shot sessions")

    model, metrics = train_baseline(shots, target="make_pct")
    joblib.dump(model, MODEL_PATH)
    print("Make% baseline model trained + saved")
    print(f"Make Train R2: {metrics['train_r2']:.4f}")
    print(f"Make Test  R2: {metrics['test_r2']:.4f}")
    print(f"Make Test MAE: {metrics['test_mae']:.4f}")

    model_green, metrics_green = train_baseline(shots, target="green_pct")
    joblib.dump(model_green, GREEN_MODEL_PATH)
    print("Green% baseline model trained + saved")
    print(f"Green Train R2: {metrics_green['train_r2']:.4f}")
    print(f"Green Test  R2: {metrics_green['test_r2']:.4f}")
    print(f"Green Test MAE: {metrics_green['test_mae']:.4f}")

    shots["pred_make"] = model.predict(shots[CAT_COLS + NUM_COLS])
    shots["residual"] = shots["make_pct"] - shots["pred_make"]
    shots["pred_green"] = model_green.predict(shots[CAT_COLS + NUM_COLS])
    shots["green_residual"] = shots["green_pct"] - shots["pred_green"]

    exploits = build_exploit_leaderboard(shots)
    print("\nEXPLOIT LEADERBOARD (top 5)")
    print(
        exploits.nlargest(5, "exploit_score")[
            [
                "base",
                "release1",
                "release2",
                "speed",
                "avg_residual",
                "avg_green_residual",
                "exploit_score",
                "latency_score",
                "total_attempts",
                "total_greens",
            ]
        ]
    )
    exploits.to_csv(LEADERBOARD_PATH, index=False)

    patch_perf, patch_alerts = build_patch_impact(shots)
    patch_perf.to_csv(PATCH_IMPACT_PATH, index=False)
    patch_alerts.to_csv(PATCH_ALERTS_PATH, index=False)

    print("\nPATCH IMPACT (significant stealth changes)")
    print(
        patch_alerts[
            ["patch", "base", "release1", "release2", "mean_make_pct", "z_score"]
        ]
    )
    save_chart(exploits)
    print(
        "Files saved: exploit_leaderboard.csv, patch_impact.csv, "
        "patch_alerts.csv, jumpshot_model.pkl, jumpshot_green_model.pkl, exploit_chart.png"
    )
    return {
        "make_train_r2": metrics["train_r2"],
        "make_test_r2": metrics["test_r2"],
        "make_test_mae": metrics["test_mae"],
        "green_train_r2": metrics_green["train_r2"],
        "green_test_r2": metrics_green["test_r2"],
        "green_test_mae": metrics_green["test_mae"],
    }


if __name__ == "__main__":
    run()
