from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = Path(__file__).resolve().parent
SHOTS_PATH = DATA_DIR / "shots.csv"
MODEL_PATH = DATA_DIR / "jumpshot_model.pkl"
LEADERBOARD_PATH = DATA_DIR / "exploit_leaderboard.csv"
PATCH_IMPACT_PATH = DATA_DIR / "patch_impact.csv"
PATCH_ALERTS_PATH = DATA_DIR / "patch_alerts.csv"
CHART_PATH = DATA_DIR / "exploit_chart.png"

CAT_COLS = ["base", "release1", "release2", "speed", "patch"]
NUM_COLS = ["player_3pt"]
EXPLOIT_COLS = ["base", "release1", "release2", "speed"]


def load_shots(path: Path = SHOTS_PATH) -> pd.DataFrame:
    shots = pd.read_csv(path)
    shots["make_pct"] = shots["makes"] / shots["attempts"]
    shots["green_pct"] = shots["make_pct"] * (
        0.65 + (shots["player_3pt"] - 85) * 0.003
    )
    shots["greens"] = shots["greens"].fillna(
        (shots["green_pct"] * shots["attempts"]).round()
    ).astype(int)
    return shots


def build_model() -> Pipeline:
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


def train_baseline(shots: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    X = shots[CAT_COLS + NUM_COLS]
    y = shots["make_pct"]

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
    exploits = (
        shots.groupby(EXPLOIT_COLS)
        .agg(
            total_attempts=("attempts", "sum"),
            avg_residual=("residual", "mean"),
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
    return exploits.sort_values("exploit_score", ascending=False)


def build_patch_impact(shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patch_perf = (
        shots.groupby(["patch"] + EXPLOIT_COLS)
        .agg(
            mean_make_pct=("make_pct", "mean"),
            total_attempts=("attempts", "sum"),
            avg_residual=("residual", "mean"),
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
    shots = load_shots()
    print(f"Est. greens calculated (92 3PT -> ~75% of makes)")
    print(f"Loaded {len(shots)} shot sessions")

    model, metrics = train_baseline(shots)
    joblib.dump(model, MODEL_PATH)
    print("Baseline model trained + saved")
    print(f"Train R2: {metrics['train_r2']:.4f}")
    print(f"Test  R2: {metrics['test_r2']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")

    shots["pred_make"] = model.predict(shots[CAT_COLS + NUM_COLS])
    shots["residual"] = shots["make_pct"] - shots["pred_make"]

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
                "exploit_score",
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
        "patch_alerts.csv, jumpshot_model.pkl, exploit_chart.png"
    )
    return metrics


if __name__ == "__main__":
    run()
